# # Deploy a Langchain RAG as a service on AWS EC2

# This is an example of easily deploying [Langchain's Quickstart RAG app](https://python.langchain.com/docs/use_cases/question_answering/quickstart)
# as a service on AWS EC2 using Runhouse.

# ## Setup credentials and dependencies

# Optionally, set up a virtual environment:
# ```shell
# $ conda create -n langchain-rag python=3.9.15
# $ conda activate langchain-rag
# ```
# Install Runhouse, the only library needed to run this script locally:
# ```shell
# $ pip install runhouse[aws]
# ```

# We'll be launching an AWS EC2 instance via [SkyPilot](https://github.com/skypilot-org/skypilot), so we need to
# make sure our AWS credentials are set up:
# ```shell
# $ aws configure
# $ sky check
# ```

# We'll be hitting OpenAI's API, so we need to set up our OpenAI API key:
# ```shell
# $ export OPENAI_API_KEY=<your openai key>
# ```
#
# ## Setting up a class for your app
#
# We import `runhouse`, because that's all that's needed to run the script locally.
# The actual torch and transformers imports can happen within the functions
# that will be sent to the Runhouse cluster; we don't need those locally.
from typing import List

import runhouse as rh

# Next, we define a class that will hold the Langchain App and allow us to send requests to it.
# You'll notice this class inherits from `rh.Module`.
# This is a Runhouse class that allows you to
# run code in your class on a remote machine.
#
# Learn more in the [Runhouse docs on functions and modules](/docs/tutorials/api-modules).
class LangchainRAG:
    def __init__(self, urls: List[str]):
        super().__init__()

        from langchain_community.document_loaders import WebBaseLoader
        from langchain_community.vectorstores import Chroma
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import PromptTemplate
        from langchain_core.runnables import RunnablePassthrough
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        self.urls = urls

        # Load, chunk and index the contents of the blog.
        loader = WebBaseLoader(
            web_paths=urls,
        )
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        splits = text_splitter.split_documents(docs)
        vectorstore = Chroma.from_documents(
            documents=splits, embedding=OpenAIEmbeddings()
        )

        # Retrieve and generate using the relevant snippets of the blog.
        retriever = vectorstore.as_retriever()
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

        template = """Use the following pieces of context to answer the question at the end.
        If you don't know the answer, specifically say that you aren't sure, but say the closest thing you
        were able to find. Use maximum five sentences to explain.

        {context}

        Question: {question}

        Helpful Answer:"""
        custom_rag_prompt = PromptTemplate.from_template(template)

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        self.rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | custom_rag_prompt
            | llm
            | StrOutputParser()
        )

    def invoke(self, user_prompt: str):
        return self.rag_chain.invoke(user_prompt)


# ## Setting up Runhouse primitives
#
# Now, we define the main function that will run locally when we run this script, and set up
# our Runhouse module on a remote cluster. First, we create a cluster with the desired instance type and provider.
# Our `instance_type` here is defined as `CPU:2`, which is the accelerator type and count that we need. In this
# case, it is a simple CPU cluster. We could alternatively specify a specific AWS instance type,
# such as `p3.2xlarge` or `g4dn.xlarge`.
#
# Learn more in the [Runhouse docs on clusters](/docs/tutorials/api-clusters).
#
# :::note{.info title="Note"}
# Make sure that your code runs within a `if __name__ == "__main__":` block, as shown below. Otherwise,
# the script code will run when Runhouse attempts to run code remotely.
# :::
if __name__ == "__main__":
    # Note: Runhouse also supports custom domains secured automatically with HTTPS so you can use your own domain name
    # when sharing an endpoint. Check out our docs on [using custom domains](https://www.run.house/docs/main/en/api/python/cluster#using-a-custom-domain)
    # for more information.
    cluster = rh.cluster(
        name="rh-serving-cpu",
        instance_type="CPU:2",
        provider="aws",
        server_connection_type="tls",
        open_ports=[443],
    ).up_if_not()

    # Next, we define the environment for our module. This includes the required dependencies that need
    # to be installed on the remote machine, as well as any secrets that need to be synced up from local to remote.
    # Passing `openai` to the `secrets` parameter will load the OpenAI API key we set up earlier.
    #
    # Learn more in the [Runhouse docs on envs](/docs/tutorials/api-envs).
    env = rh.env(
        name="langchain_rag_env",
        reqs=[
            "langchain",
            "langchain-community",
            "langchainhub",
            "langchain-openai",
            "chromadb",
            "bs4",
        ],
        secrets=["openai"],
    )

    # Finally, we define our module and run it on the remote cluster. We construct it normally and then call
    # `get_or_to` to run it on the remote cluster. Using `get_or_to` allows us to load the exiting Module
    # by the name `basic_rag_app` if it was already put on the cluster. If we want to update the module each
    # time we run this script, we can use `to` instead of `get_or_to`.
    #
    # Note that we also pass the `env` object to the `get_or_to` method, which will ensure that the environment is
    # set up on the remote machine before the module is run.
    urls = (
        "https://www.nyc.gov/site/hpd/services-and-information/tenants-rights-and-responsibilities.page",
        "https://www.nyc.gov/content/tenantprotection/pages/covid19-home-quarantine",
        "https://www.nyc.gov/content/tenantprotection/pages/new-protections-for-all-tenants",
    )

    RemoteLangchainRAG = rh.module(LangchainRAG, name="basic_rag_app").get_or_to(
        cluster, env=env
    )

    rag_app = RemoteLangchainRAG(urls)

    # After this app is set up, it maintains state on the remote cluster. We can repeatedly call the `invoke`
    # method, and the module functions as a remote service.
    user_input = input("Ask a question about NYC tenants rights and responsibilities: ")
    print(rag_app.invoke(user_input))

    # ## Standing up an endpoint
    # We can call the model via an HTTP request, which calls directly into the module's `invoke` method:
    # ```python
    #   base_url = f"{rag_app.endpoint()}/invoke"
    #   encoded_prompt = urllib.parse.quote(user_input)
    #   resp = requests.get(f"{base_url}?user_prompt={encoded_prompt}")
    #   print(resp.json())
    # ```

    # And we can also call it via cURL:
    # ```python
    # print(
    #     f"curl {base_url}?user_prompt={encoded_prompt} -X GET -d "
    #     "-H 'Content-Type: application/json'"
    # )
    # ```
