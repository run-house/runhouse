import time
from typing import List

import runhouse as rh


class EmbeddingMapper:
    def __init__(self, model_name: str, model_kwargs: dict, encode_kwargs: dict):

        from langchain.embeddings import HuggingFaceBgeEmbeddings

        self.model_name = model_name
        self.model_kwargs = model_kwargs
        self.encode_kwargs = encode_kwargs

        self.model = HuggingFaceBgeEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )

    def embed_into_vector_store(self, urls: List[str]):
        from bs4 import BeautifulSoup as Soup
        from langchain.vectorstores import LanceDB
        from langchain_community.document_loaders.recursive_url_loader import (
            RecursiveUrlLoader,
        )
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        # Load, chunk and index the contents of the blog.

        # Time loading of URLS
        start_time = time.time()
        # loader = WebBaseLoader(
        #     web_paths=urls,
        # )
        loader = RecursiveUrlLoader(
            url=urls[0], max_depth=5, extractor=lambda x: Soup(x, "html.parser").text
        )
        docs = loader.load()
        print(f"Loaded {len(docs)} docs.")
        print(f"Time to load URLs: {time.time() - start_time}")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        splits = text_splitter.split_documents(docs)

        # Time embedding
        start_time = time.time()

        self.vectorstore = LanceDB(embedding=self.model)
        self.vectorstore.add_documents(splits)
        print(f"Time to embed: {time.time() - start_time}")


if __name__ == "__main__":
    model_name = "BAAI/bge-large-en-v1.5"
    model_kwargs = {"device": "cuda"}
    encode_kwargs = {
        "normalize_embeddings": True
    }  # set True to compute cosine similarity

    cluster = rh.cluster("rh-a10g4", instance_type="A10G:4").save()

    env = rh.env(
        name="langchain_embed_env",
        reqs=[
            "langchain",
            "langchain-community",
            "langchainhub",
            "lancedb",
            "bs4",
            "sentence_transformers",
        ],
        secrets=["huggingface"],
    )

    mapper_class = rh.module(EmbeddingMapper).to(cluster, env=env)

    model = mapper_class(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )

    model.embed_into_vector_store(urls=["https://js.langchain.com/docs/"])
