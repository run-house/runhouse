# # RAG App with Vector Embedding and LLM Generation
#
# This example defines a retrieval augmented generation (RAG) app that references text from websites
# to enrich the response from an LLM. Depending on the URLs you use to feed the application database,
# you'll be able to answer questions more intelligently with explicit context.
#
# You could, for example, pass in the URL of a new startup and then ask the app to answer questions
# about that startup using information on their site. Or pass in several specialized gardening websites and
# ask nuanced questions about horticulture.
#
# ## Example Overview
# Deploy a FastAPI app that is able to create and store embeddings from text on public website URLs,
# and generate answers to questions using related context from stored websites and an open source LLM.
#
# #### Indexing:
# - Send a list of URL paths to the application via a POST endpoint
# - Use [LangChain](https://python.langchain.com/v0.1/docs/modules/data_connection/document_transformers/)
#   to **parse text from URLs**
# - **Create vector embeddings from split text** with [Sentence Transformers (SBERT)](https://sbert.net/index.html)
# - Store embeddings in a **vector database** ([LanceDB](https://lancedb.com/))
#
# #### Retrieval and generation:
# - Send question text via a GET endpoint on the FastAPI application
# - Create a vector embedding from the text and **retrieve related docs** from database
# - **Construct an LLM prompt** from documents and original question
# - Generate a response using an **LLM (Llama 3)**
# - **Output response** with source URLs and question input
#
# ![Graphic displaying the steps of indexing data and the retrieval and generation process](https://runhouse-tutorials.s3.amazonaws.com/indexing-retrieval-generation.png)
#
# Note: that some of the steps in this example could be accomplished more simply with platforms like OpenAI and
# tools such as LangChain, but we break out the components explicitly to fully illustrate each step and make the
# example easily adaptible to other use cases. Swap out components as you see fit!
#
# ### What does Runhouse enable?
# Runhouse allows you to turn complex operations such as preprocessing and inference into independent services.
# Servicifying with Runhouse enables:
# - **Decoupling**: AI/ML or compute-heavy heavy tasks can be separated from your main application.
#   This keeps the FastAPI app light and allows your service to scale independently.
# - **Multi-cloud**: Host each service on remote machies from any cloud provider (or even on the same GPU).
#   Take advantage of unused cloud credits and avoid platform dependence.
# - **Sharing**: Running on GPUs can be expensive, especially if they aren't fully utilized. By sharing services within
#   your organization, you can substantially cut costs.
#
# ### Why FastAPI?
# We choose FastAPI as our platform because of its popularity and simplicity. However, we could
# easily use any other *Python-based* platform with Runhouse. Streamlit, Flask, `<your_favorite>`, we got you!
#
# ## Setup credentials and dependencies
#
# To ensure that Runhouse is able to managed deploying services to your cloud provider (AWS in our case)
# you may need to follow initial setup steps. Please visit the AWS section of
# our [Installation Guide](https://www.run.house/docs/installation)
#
# Additionally, we'll be downloading the Llama 3 model from Hugging Face, so we need to set up our Hugging Face token:
# ```shell
# $ export HF_TOKEN=<your huggingface token>
# ```
#
# Make sure to sign the waiver on the [Hugging Face model page](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)
# so that you can access it.
#
# ## FastAPI RAG App Setup
# First, we'll import necessary packages and initialize variables used in the application. The `URLEmbedder` and
# `LlamaModel` classes that will be sent to Runhouse are available in the `app/modules` folder in this source code.
from contextlib import asynccontextmanager
from typing import Dict, List

import dotenv
import lancedb

import runhouse as rh

from fastapi import Body, FastAPI, HTTPException

from app.modules.embedding import Item, URLEmbedder
from app.modules.llm import LlamaModel

dotenv.load_dotenv()

EMBEDDER, TABLE, LLM = None, None, None

DEBUG = True  # In DEBUG mode we will always override Runhouse modules

# Configuration options for our remote cluster
CLUSTER_NAME = "rh-xa10g"  # Allows the cluster to be reused
INSTANCE_TYPE = "A10G:1"  # A10G GPU to handle LLM iinference
CLOUD_PROVIDER = "aws"  # Alternatively "gcp", "azure", or "cheapest"

# Template to be used in the LLM generation phase of the RAG app
PROMPT_TEMPLATE = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use three sentences maximum and keep the answer as concise as possible.
Always say "thanks for asking!" at the end of the answer.

{context}

Question: {question}

Helpful Answer: """


# ### Initialize Embedder, LanceDB Database, and LLM Service
# We'll use the `lifespan` argument of the FastAPI app to initialize our embedding service,
# vector database, and LLM service when the application is first created.
@asynccontextmanager
async def lifespan(app):
    global EMBEDDER, TABLE, LLM

    EMBEDDER = load_embedder()
    TABLE = load_table()
    LLM = load_llm()

    yield


# ### Create Vector Embedding Service as a Runhouse Module
# This method, run durig initialization, will provision a remote machine (A10G on AWS in this case)
# and deploy our `URLEmbedder` to that machine. The `env` is essentially the worker environment that
# will run the module.
#
# The Python packages required by the embedding service (`langchain` etc.) are defined on the Runhouse `env` so that
# they can be properly installed on the cluster. Also note that they are imported inside class methods. Those packages
# do not need to be installed locally.
def load_embedder():
    """Launch an A10G and send the embedding service to it."""
    # TODO: On production how does this know our AWS credentials to deploy or find the cluster?
    # Is there a good way to pass those in through a Docker container?
    # We *could* update the Dockerfile but is that a little janky?

    # TODO: If we DO replace the existing embedder, we should del the instance and run garbage
    # collection

    # TODO:
    # rh.login(YOUR_TOKEN)
    # rh.sky(aws_key=AWS_KEY)
    # rh.sky(creds="/path/to/folder/with/creds/in/repo")
    # rh.login(YOUR_TOKEN, AWS_CREDENTIALS="/path/in/local/files") # like how boto does it.

    cluster = rh.cluster(
        CLUSTER_NAME, instance_type=INSTANCE_TYPE, provider=CLOUD_PROVIDER
    ).up_if_not()

    env = rh.env(
        name="embedder_env",
        reqs=[
            "langchain",
            "langchain-community",
            "langchain_text_splitters",
            "langchainhub",
            "bs4",
            "sentence_transformers",
            "torch",
        ],
        compute={"GPU": 1},
    )

    module_name = "url_embedder"
    remote_url_embedder = cluster.get(module_name, default=None, remote=True)
    if DEBUG or remote_url_embedder is None:
        RemoteEmbedder = rh.module(URLEmbedder).to(
            system=cluster, env=env, name="URLEmbedder"
        )
        remote_url_embedder = RemoteEmbedder(
            model_name_or_path="BAAI/bge-large-en-v1.5", device="cuda", name=module_name
        )
    return remote_url_embedder


# ### Initialize LanceDB vector database
# We'll be using open source [LanceDB](https://lancedb.com/) to create an embedded database to store
# the URL embeddings and perform vector search for the retrieval phase. You could alternatively try
# Chroma, Pinecone, Weaviate, or even MongoDB.
def load_table():
    db = lancedb.connect("/tmp/db")
    return db.create_table("rag-table", schema=Item.to_arrow_schema(), exist_ok=True)


# ### Load RAG LLM Inference Service with Runhouse
# Deploy an open LLM, Llama 3 in this case, to a GPU on the cloud provider of your choice.
# We will use vLLM to serve the model due to it's high performance and throughput but there
# are many other options such as HuggingFace Transforms and TGI.
#
# Here we leverage the same A10G cluster we used for the embedding service, but you could also spin
# up a new remote machine specifically for the LLM service. Alternatively, use a proprietary model
# like ChatGPT or Claude.
def load_llm():
    """Use the existing A10G cluster to run an LLM inference service"""
    # Specifying the same name will reuse our embedding service cluster
    cluster = rh.cluster(
        CLUSTER_NAME, instance_type=INSTANCE_TYPE, provider=CLOUD_PROVIDER
    ).up_if_not()

    env = rh.env(
        reqs=["torch", "vllm"],
        secrets=["huggingface"],  # Needed to download Llama 3 from HuggingFace
        name="llama3_inference_env",
    )

    module_name = "llama_model"
    # First check for an instance of the LlamaModel stored on the cluster
    remote_llm = cluster.get(module_name, default=None, remote=True)
    if DEBUG or remote_llm is None:
        # If not found (or debugging) sync up the model and create a fresh instance
        RemoteLlama = rh.module(LlamaModel).to(
            system=cluster, env=env, name="LlamaModel"
        )
        remote_llm = RemoteLlama(name=module_name)
    return remote_llm


### Initialize the FastAPI application
# Before defining endpoints, we'll initialize the application and set the lifespan events defined
# above. This will load in the various services we've defined on start-up.
app = FastAPI(lifespan=lifespan)


# Add an endpoint to check on our app health. This is a minimal example intended to only
# show if the application is up and running or down.
@app.get("/health")
def health_check():
    return {"status": "healthy"}


# ### Vector Embedding POST Endpoint
# To illustrate tehe flexibility of FastAPI, we're allowing embeddings to be added to your database
# via a POST endpoint. This method will use the embedder service to create database entries with the
# source, content, and vector embeddings for chunks of text from a provided list of URLs.
@app.post("/embeddings")
async def generate_embeddings(paths: List[str] = Body([]), kwargs: Dict = Body({})):
    """Generate embeddings for the URL and write to DB."""
    try:
        items = await EMBEDDER.embed_docs(
            paths,
            normalize_embeddings=True,
            run_async=True,
            stream_logs=False,
            **kwargs,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to embed URLs: {str(e)}")

    items = [Item(**item) for item in items]
    TABLE.add(items)
    return {"status": "success"}


# ## Retrieval augmented generation (RAG) steps:
# Now that we've defined our services and created an endpoint to populate data for retrieval, the remaining
# components of the application will focus on the generative phases of the RAG app.

# ### Retrieval with Sentence Transformers and LanceDB
# In the retrieval phase we'll first use the Embedder service to create an embedding from the input text
# to search our LanceDB vector database with. LanceDB is optimized for vector searches in this manner.
async def retrieve_documents(text: str, limit: int) -> List[Item]:
    """Retrieve documents from vector DB related to input text"""
    try:
        # Encode the input text into a vector
        vector = await EMBEDDER.encode_text(
            text, normalize_embeddings=True, run_async=True, stream_logs=False
        )
        # Search LanceDB for nearest neighbors to the vector embed
        results = TABLE.search(vector).limit(limit).to_pydantic(Item)
        return results

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve documents: {str(e)}"
        )


# ### Format an Augmented Prompt
# To leverage the documents retrieved from the previous step, we'll format a prompt that provides text from
# related documents as "context" for the LLM prompt. This allows a general purpose LLM (like Llama) to provide
# more specific responses to a particular question.
async def format_prompt(text: str, docs: List[Item]) -> str:
    """Retrieve documents from vector DB related to input text"""
    context = "\n".join([doc.page_content for doc in docs])
    prompt = PROMPT_TEMPLATE.format(question=text, context=context)
    return prompt


# ### Generation GET endpoint
# Using the methods above, this endpoint will run inference on our LLM to generate a response to a question.
# The results are enhanced by first retrieving related documents from the source URLs fed into the POST endpoint.
# Content from the fetched documents is then formatted into the text prompt sent to our self-hosted LLM.
# We'll be using a generic prompt template to illustrate how many "chat" tools work behind the scenes.
@app.get("/generate")
async def generate_response(text: str, limit: int = 4):
    """Generate a response to a question using an LLM with context from our database"""
    if not text:
        return {"error": "Question text is missing"}

    try:
        # Retrieve related documents from vector DB
        documents = await retrieve_documents(text, limit)

        # List of sources from retrieved documents
        sources = set([doc.url for doc in documents])

        # Create a prompt using the documents and search text
        prompt = await format_prompt(text, documents)

        response = await LLM.generate(
            prompt=prompt, temperature=0.8, top_p=0.95, max_tokens=100
        )

        return {"question": text, "response": response, "sources": sources}

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to generate response: {str(e)}"
        )


# ## Run the FastAPI App Locally
# Use the following command to run the app from your terminal:
#
# ```shell
# $ fastapi run app/main.py
# ```
#
# After a few minutes, you can navigate to `http://127.0.0.1/health` to check that your application is running.
#
# You'll see something like:
# ```json
# { "status": "healthy" }
# ```
#
# To debug the application, we recommend you use `fastapi dev`. This will trigger
# automatic re-deployments from any changes to your code. Be sure to set `DEBUG` to `True` to
# override instances of the embedding and LLM services with updated versions.
#
# ### Example cURL Command to Add Embeddings
# To populate the LanceDB database with vector embeddings for use in the RAG app, you can send a HTTP request
# to the `/embeddings` POST endpoint. Let's say you have a question about bears. You could send a cURL
# command with a list of URLs including essential bear information:
#
# ```shell
# curl --header "Content-Type: application/json" \
#   --request POST \
#   --data '{"paths":["https://www.nps.gov/yell/planyourvisit/safety.htm", "https://en.wikipedia.org/wiki/Adventures_of_the_Gummi_Bears"]}' \
#   http://127.0.0.1:8000/embeddings
# ```
#
# Alternatively, we recommend a tool like [Postman](https://www.postman.com/) to test HTTP APIs.
#
# ### Test the Generation Endpoint
# Open your browser and send a prompt to your locally running RAG app by appending your question
# to the URL as a query param, e.g. `?text=Does%20yellowstone%20have%20gummi%20bears%3F`
#
# ```text
# "http://127.0.0.1/generate?text=Does%20yellowstone%20have%20gummi%20bears%3F"
# ```
#
# The `LlamaModel` will need to load on the initial call and may take a few minutes to generate a
# response. Subsequent calls will generally take less than a second to generate.
#
# Example output:
#
# ```json
# {
#   "question": "Does yellowstone have gummi bears?",
#   "response": [
#     " No, Yellowstone is bear country, not gummi bear country. Thanks for asking! "
#   ],
#   "sources": [
#     "https://www.nps.gov/yell/planyourvisit/safety.htm",
#     "https://en.wikipedia.org/wiki/Adventures_of_the_Gummi_Bears"
#   ]
# }
# ```
#
# ## Deploying to Production
# There are any methods to deploy a FastAPI application to a production environment. With some modifications
# to the logic of the app, setting `DEBUG` to `False`, and deploying to a public IP, this example
# could easily serve as the backend to a RAG app.
#
# We won't go into depth on a specific method, but here are a few things to consider:
# - **Cloud credentials**: Runhouse uses SkyPilot to provision remote machines on various cloud providers. You'll
#   need to ensure that you have the appropriate permissios available on your production environment.
# - **ENV Variables**: This example passes your Hugging Face token to a Runhouse `env` to grant permissions to use
#   Llama 3. Make sure you handle this on your server as well.
# - **Cluster lifecycle**: Running GPUs can be costly. `sky` commands make it easy to manage remote clusters locally
#   but you may also want to monitor your cloud provider to avoid unused GPUs running up a bill.
#
# If you're running into any problems using Runhouse in production, please reach out to our team at
# [team@run.house](mailto:team@run.house). We'd be happy to set up a time to help you debug live.
# Additionally, you can chat with us directly on [Discord](https://discord.com/invite/RnhB6589Hs)
