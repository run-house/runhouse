from contextlib import asynccontextmanager
from typing import Dict

import dotenv
import lancedb

import runhouse as rh

from fastapi import Body, FastAPI, HTTPException
from modules.embedding import Item, URLEmbedder

dotenv.load_dotenv()

EMBEDDER, TABLE = None, None


@asynccontextmanager
async def lifespan(app):
    global EMBEDDER, TABLE

    EMBEDDER = load_embedder()
    TABLE = load_table()

    yield


def load_embedder():
    """Launch an A10G and send the embedding service to it."""
    cluster = rh.cluster("rh-xa10g", instance_type="A10G:1", provider="aws").up_if_not()

    env = rh.env(
        name="embedder_env",
        reqs=[
            "langchain",
            "langchain-community",
            "langchainhub",
            "bs4",
            "sentence_transformers",
            "fake_useragent",
        ],
        compute={"GPU": 1},
    )

    RemoteEmbedder = rh.module(URLEmbedder).get_or_to(system=cluster, env=env)
    remote_url_embedder = RemoteEmbedder(
        model_name_or_path="BAAI/bge-large-en-v1.5",
        device="cuda",
        name="doc_embedder",
    )
    return remote_url_embedder


def load_table():
    db = lancedb.connect("/tmp/db")
    return db.create_table("my-table", schema=Item.to_arrow_schema(), exist_ok=True)


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/embedding")
async def generate_embeddings(url: str = Body(...), kwargs: Dict = Body({})):
    """Generate embeddings for the URL and write to DB."""
    try:
        embedding, embed_time, download_time = await EMBEDDER.embed_doc(
            url, normalize_embeddings=True, run_async=True, stream_logs=False, **kwargs
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to embed URLs: {str(e)}")

    # TODO not working
    TABLE.add(Item(url=url, vector=embedding))
    return {"status": "success"}


@app.get("/embedding")
def load_embedding(url: str):
    try:
        results = TABLE.search(url).limit(1).to_pydantic(Item)
        if not results:
            return []

        return results[0].text

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
