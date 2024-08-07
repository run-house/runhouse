# # Deploy a FastAPI RAG App with Runhouse Modules

# This example is for a RAG app that references text from websites to enrich the response from an LLM.
# You could, for example, pass in the URL of a new startup and then ask the app to answer questions
# about that startup using information on their site. Or pass in the website for every single company
# linked to from YC and summarize what sort of AI tool each one is.

# ## Example Overview
#
# -
# -
# -
#
# MAYBE AN IMAGE OF THE ARCHITECTURE HERE?
#
# Note that some of the steps in this example could be accomplished more simply with tools
# such as LangChain, but we try to break out the explicit parts so that it's more
# adaptible to whatever your use case is.
#
# ### Where does Runhouse come in?
# Runhouse allows you to decouple your GPU tasks such as preprocessing and inference.
# Multi-cloud means you can host your CPU server in one place and each service anywhere else
# on any cloud provider (or even on the same GPU). This  ....
#
# ### Why FastAPI?
# We choose FastAPI as our platform because of its popularity and simplicity. However, we could
# easily use *any* over Python-based platform with Runhouse. Elixir, PHP, you name it.

# ## Setup credentials and dependencies
#
# TK
#
#
#
#

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

# ## Initialize the Embedder and LanceDB Database
# We'll use the `lifespan` argument of the FastAPI app to initialize both
# our embedding service and vector database when the application is first
# created.
@asynccontextmanager
async def lifespan(app):
    global EMBEDDER, TABLE, LLM

    EMBEDDER = load_embedder()
    TABLE = load_table()
    LLM = load_llm()

    yield


# ## Create the Embedding Service as a Runhouse Module
# This method, run durig initialization, will provision a remote machine (A10G on AWS) in this case
# and deploy our `URLEmbedder`
def load_embedder():
    """Launch an A10G and send the embedding service to it."""
    # On production how does this know our AWS credentials to deploy or find the cluster?
    # Is there a good way to pass those in through a Docker container?
    # We *could* update the Dockerfile but is that a little janky?
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
        ],
        compute={"GPU": 1},
    )

    # Change from `.get_or_to` to `.to` to deploy changes while debugging application
    RemoteEmbedder = rh.module(URLEmbedder).to(
        system=cluster, env=env, name="doc_embedder"
    )
    remote_url_embedder = RemoteEmbedder(
        model_name_or_path="BAAI/bge-large-en-v1.5",
        device="cuda",
    )
    return remote_url_embedder


# ## Initialize LanceDB vector database
# We'll be using LanceDB to create a local vector database to store the URL embeddings and perform searchs
# for related documents. You could alternatively use a service like MongoDB (locally or hosted)
# to improve the modurality of your application.
def load_table():
    db = lancedb.connect("/tmp/db")
    return db.create_table("rag-table", schema=Item.to_arrow_schema(), exist_ok=True)


# ## Create an LLM Inference Service with Runhouse
# Deploy an open LLM, Llama 3.1 in this case, to a GPU on the cloud provider of your choice.
# We will use vLLM to serve the model due to it's high performance and throughput but there
# are many other options such as HuggingFace Transforms and TGI.
#
# Here we leverage the same A10G cluster we used for the embedding service, but you could also spin
# up a new remote machine specifically for the LLM service. Alternatively, used a closed model
# like ChatGPT or Claude
def load_llm():
    """Use the existing A10G cluster to run an LLM inference service"""
    cluster = rh.cluster(
        CLUSTER_NAME, instance_type=INSTANCE_TYPE, provider=CLOUD_PROVIDER
    ).up_if_not()

    env = rh.env(
        reqs=["vllm==0.2.7"],  # >=0.3.0 causes Pydantic version error
        secrets=["huggingface"],  # Needed to download Llama 3 from HuggingFace
        name="llama3_inference_env",
    )

    RemoteLlama = rh.module(LlamaModel).get_or_to(
        system=cluster, env=env, name="llama3_8b_model"
    )
    remote_llm = RemoteLlama()
    return remote_llm


# Initialize the FastAPI application
app = FastAPI(lifespan=lifespan)


# Add an endpoint to check on our app health. This is a minimal example intended to only
# show if the application is up and running or down.
@app.get("/health")
def health_check():
    return {"status": "healthy"}


# ##
# In this example, we're allowing embeddings to be added to your database via a POST endpoint to
# illustrate tehe flexibility of FastAPI.
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


async def format_prompt(text: str, docs: List[Item]) -> str:
    """Retrieve documents from vector DB related to input text"""
    context = "\n".join([doc.page_content for doc in docs])
    prompt = PROMPT_TEMPLATE.format(question=text, context=context)
    return prompt


@app.get("/generate")
async def generate_response(text: str, limit: int = 5):
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
            status_code=500, detail=f"Failed to load embeddings: {str(e)}"
        )


# ## Run the FastAPI app locally
# Use the following command to run the app from your terminal:
#
# ```shell
# fastapi run app/main.py
# ```
#
# To debug the application, we recommend you use `fastapi dev`. This will enable
# automatic re-deployments based on changes to your code. Be sure to use `rh.module().to` in
# place of `rh.module().get_or_to` to redeploy any changes to your Runhouse Module as well.

# ## Deploy Locally via Docker
#
#
# ### Build the Docker Image
#
# ```shell
# docker build -t myimage .
# ```
#
# ### Start the Docker Container
# ```shell
# docker run -d --name mycontainer -p 80:80 myimage
# ```
#
# Now you can go to `http://127.0.0.1/health` to check on your application.
#
# You'll see:
# ```
# { "status": "healthy" }
# ```
#
# ## Deploy the FastAPI App to a Remote Machine
#
# For production use cases, you'll want to deploy your app to a VM where it can be
# easily consumed at a public URL (or privately, if you prefer). This would, for example,
# enable you to use the RAG app as the backend for a website. Or you could build a UI on
# top of the FastAPI code we've started.
#
# [TODO]: How to put your AWS config inside the container? Maybe include in the Dockerfile??
#

# {
#   "question": "How is Runhouse better than LangChain?",
#   "response": [
#     " Runhouse is better than LangChain because it allows you to rapidly iterate your AI on your own infrastructure, whereas LangChain is more focused on building context-aware, reasoning applications with its flexible framework. Runhouse provides a modular home for ML infra, allowing you to browse, manage, and grow your ML stack as a living set of shared services and compute, whereas LangChain is more focused on building LLM apps. Thanks for asking!"
#   ],
#   "sources": [
#     "https://www.run.house",
#     "https://www.langchain.com"
#   ]
# }
