from contextlib import asynccontextmanager
from typing import Dict

import dotenv
import lancedb

import runhouse as rh

from fastapi import Body, FastAPI, HTTPException

from app.modules.embedding import Item, URLEmbedder

dotenv.load_dotenv()

EMBEDDER, TABLE = None, None

# ## Initialize the Embedder and LanceDB Database
# We'll use the `lifespan` argument of the FastAPI app to initialize both
# our embedding service and vector database when the application is first
# created.


@asynccontextmanager
async def lifespan(app):
    global EMBEDDER, TABLE

    EMBEDDER = load_embedder()
    TABLE = load_table()

    yield


# ## Create the Embedding Service as a Runhouse Module


def load_embedder():
    """Launch an A10G and send the embedding service to it."""
    cluster = rh.cluster("rh-xa10g", instance_type="A10G:1", provider="aws").up_if_not()

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

    # Change from `.get_or_to` `.to` to deploy changes while debugging application
    RemoteEmbedder = rh.module(URLEmbedder).to(system=cluster, env=env)
    remote_url_embedder = RemoteEmbedder(
        model_name_or_path="BAAI/bge-large-en-v1.5",
        device="cuda",
        name="doc_embedder",
    )
    return remote_url_embedder


def load_table():
    db = lancedb.connect("/tmp/db")
    return db.create_table("rag-table", schema=Item.to_arrow_schema(), exist_ok=True)


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/embedding")
async def generate_embeddings(url: str = Body(...), kwargs: Dict = Body({})):
    """Generate embeddings for the URL and write to DB."""
    try:
        items = await EMBEDDER.embed_doc(
            url, normalize_embeddings=True, run_async=True, stream_logs=False, **kwargs
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to embed URLs: {str(e)}")

    items = [Item(**item) for item in items]
    TABLE.add(items)
    return {"status": "success"}


@app.get("/embedding")
async def load_embedding(text: str):
    try:
        vector = await EMBEDDER.encode_text(
            text, normalize_embeddings=True, run_async=True, stream_logs=False
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to encode text: {str(e)}")

    try:
        results = TABLE.search(vector).limit(1).to_pydantic(Item)
        if not results:
            return []

        return results[0].page_content

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to load embeddings: {str(e)}"
        )


@app.get("/search")
def find_nearest_neighbors(url: str):
    try:
        results = TABLE.search(url).limit(2).to_pandas()
        if not results:
            return []

        return results

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to find nearest neighbor: {str(e)}"
        )


## Deploy the FastAPI app

if __name__ == "__main__":
    import uvicorn

    DEFAULT_APP_HOST = "0.0.0.0"
    DEFAULT_APP_PORT = 8000

    uvicorn.run(app, host=DEFAULT_APP_HOST, port=DEFAULT_APP_PORT)


# To debug the application, we recommend you use the FastAPI CLI. This will enable
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
#
#
#
