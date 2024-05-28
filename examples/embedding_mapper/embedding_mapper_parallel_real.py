import time

import runhouse as rh


def partition_list(lst, num_chunks):
    chunks = []
    chunk_size = len(lst) // num_chunks
    for i in range(num_chunks):
        start = i * chunk_size
        end = (i + 1) * chunk_size
        chunks.append(lst[start:end])

    if len(chunks) > num_chunks:
        chunks[-2].extend(chunks[-1])
        chunks = chunks[:-1]

    return chunks


class EmbeddingMapper:
    def embed_url_recursively(self, url: str, mapper, max_depth: int = 5):
        from langchain_community.document_loaders.recursive_url_loader import (
            RecursiveUrlLoader,
        )

        loader = RecursiveUrlLoader(url=url, max_depth=max_depth)
        print(f"Loading {url} recursively...")
        docs = list(loader.load())
        print(f"Loaded {len(docs)} docs.")

        print("Embedding docs...")
        # Partition into chunks and embed
        chunked_docs = partition_list(docs, num_chunks=3)
        mapper.map(chunked_docs, method="embed_docs")


class DocEmbedder:
    def __init__(self):
        self.model = None
        self.vectorstore = None

    def initialize_model(self):
        if self.model is None:
            from langchain.embeddings import HuggingFaceBgeEmbeddings

            model_name = "BAAI/bge-large-en-v1.5"
            model_kwargs = {"device": "cuda"}
            encode_kwargs = {
                "normalize_embeddings": True
            }  # set True to compute cosine similarity

            self.model = HuggingFaceBgeEmbeddings(
                model_name=model_name,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs,
            )

    def embed_docs(self, docs):
        from langchain.vectorstores import LanceDB
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        self.initialize_model()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        splits = text_splitter.split_documents(docs)

        # This connects to our existing LanceDB

        # Time embedding
        start_time = time.time()
        self.vectorstore = LanceDB(embedding=self.model)
        self.vectorstore.add_documents(splits)
        print(f"Time to embed {len(docs)} docs: {time.time() - start_time}")


if __name__ == "__main__":

    cluster = rh.cluster("rh-a10g", instance_type="A10G:4").save().up_if_not()
    cluster.restart_server()

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

    remote_module = rh.module(EmbeddingMapper)().to(system=cluster, env=env)
    remote_doc_embedder = rh.module(DocEmbedder)().to(system=cluster, env=env)
    remote_mapper = rh.mapper(remote_doc_embedder).to(system=cluster, env=env)
    remote_mapper.add_replicas(3)

    remote_module.embed_url_recursively("https://js.langchain.com/docs/", remote_mapper)
