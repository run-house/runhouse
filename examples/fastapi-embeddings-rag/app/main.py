# # RAG App with Vector Embedding and LLM Generation
#
# This example defines a retrieval augmented generation (RAG) app that references text from websites
# to enrich the response from an LLM. Depending on the URLs you use to populate the vector database,
# you'll be able to answer questions more intelligently with relevant context.
#
# ## Example Overview
# Deploy a FastAPI app that is able to create and store embeddings from text on public website URLs,
# and generate answers to questions using related context from stored websites and an open source LLM.
#
# #### Indexing:
# - **Send a list of URL paths** to the application via a POST endpoint
# - Use [LangChain](https://python.langchain.com/v0.1/docs/modules/data_connection/document_transformers/)
#   to **parse text from URLs**
# - **Create vector embeddings from split text** with [Sentence Transformers (SBERT)](https://sbert.net/index.html)
# - Store embeddings in a **vector database** ([LanceDB](https://lancedb.com/))
#
# #### Retrieval and generation:
# - **Send question text** via a GET endpoint on the FastAPI application
# - Create a vector embedding from the text and **retrieve related docs** from database
# - **Construct an LLM prompt** from documents and original question
# - Generate a response using an **LLM (Llama 3)**
# - **Output response** with source URLs and question input
#
# ![Graphic displaying the steps of indexing data and the retrieval and generation process](https://runhouse-tutorials.s3.amazonaws.com/indexing-retrieval-generation.png)
#
# Note: Some of the steps in this example could also be accomplished with platforms like OpenAI and
# tools such as LangChain, but we break out the components explicitly to fully illustrate each step and make the
# example easily adaptible to other use cases. Swap out components as you see fit!
#
# ### What does Kubetorch enable?
# Kubetorch allows you to turn complex operations such as preprocessing and inference into independent services.
# By decoupling accelerated compute tasks from your main application, you can keep the FastAPI app
# light and allows each service to scale independently.
#

# ## FastAPI RAG App Setup
# First, we'll import necessary packages and initialize variables used in the application. The `URLEmbedder` and
# `LlamaModel` classes that will be sent to remote compute are available in the `app/modules` folder in this source code.
from contextlib import asynccontextmanager
from typing import Dict, List

import kubetorch as kt

import lancedb

from fastapi import Body, FastAPI, HTTPException

from app.modules.embedding import Item, URLEmbedder
from app.modules.llm import LlamaModel

EMBEDDER, TABLE, LLM = None, None, None

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


# ### Create Vector Embedding Service
# This method, run during initialization, will provision remote compute and deploy an embedding service.
#
# The Python packages required by the embedding service (`langchain` etc.) are defined on the image, which
# is a base Docker image and any additional commands to run such as pip installs.
def load_embedder():
    """Launch an A10G and send the embedding service to it."""
    img = kt.images.pytorch().pip_install(
        [
            "langchain",
            "langchain-community",
            "langchain_text_splitters",
            "langchainhub",
            "bs4",
            "sentence_transformers",
        ],
    )
    compute = kt.compute(gpus="L4:1", image=img)

    init_args = dict(model_name_or_path="BAAI/bge-large-en-v1.5", device="cuda")

    remote_url_embedder = (
        kt.cls(URLEmbedder)
        .to(compute, init_args)
        .distribute("auto", num_replicas=(0, 4))
    )

    return remote_url_embedder


# ### Initialize LanceDB Vector Database
# We'll be using open source [LanceDB](https://lancedb.com/) to create an embedded database to store
# the URL embeddings and perform vector search for the retrieval phase. You could alternatively try
# Chroma, Pinecone, Weaviate, or even MongoDB.
def load_table():
    # Initialize LanceDB database directly on the FastAPI app's machine
    db = lancedb.connect("/tmp/db")
    return db.create_table("rag-table", schema=Item.to_arrow_schema(), exist_ok=True)


# ### Load RAG LLM Inference Service
# Deploy an open LLM, Llama 3 in this case, to 1 or more GPUs in the cloud.
# We will use vLLM to serve the model due to it's high performance.
#
def load_llm():
    img = kt.images.pytorch().pip_install(["vllm==0.5.4"]).sync_secrets(["huggingface"])

    compute = kt.Compute(gpus="L4:1", image=img)
    remote_llm = (
        kt.cls(LlamaModel).to(system=compute).distribute("auto", num_replicas=(0, 4))
    )
    return remote_llm


### Initialize the FastAPI Application
# Before defining endpoints, we'll initialize the application and set the lifespan events defined
# above. This will load in the various services we've defined on start-up.
app = FastAPI(lifespan=lifespan)

# Add an endpoint to check on our app health. This is a minimal example intended to only
# show if the application is up and running or down.
@app.get("/health")
def health_check():
    return {"status": "healthy"}


# ### Vector Embedding POST Endpoint
# To illustrate the flexibility of FastAPI, we're allowing embeddings to be added to your database
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


# ## Retrieval Augmented Generation (RAG) Steps:
# Now that we've defined our services and created an endpoint to populate data for retrieval, the remaining
# components of the application will focus on the generative phases of the RAG app.

# ### Retrieval with Sentence Transformers and LanceDB
# In the retrieval phase we'll first use the Embedder service to create an embedding from input text
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
# related documents as "context" for the LLM. This allows a general purpose LLM (like Llama) to provide
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

        # Send prompt with optional sampling parameters for vLLM
        # More info: https://github.com/vllm-project/vllm/blob/main/vllm/sampling_params.py
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
# This may take a while due to initialization logic in the lifespan.
#
# You'll see something like:
# ```json
# { "status": "healthy" }
# ```
#
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
# response. Subsequent calls will generally take less than a second.
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
