import runhouse as rh


class EmbeddingMapper:
    def embed_url_recursively(self, url: str, max_depth: int = 5):
        from langchain_community.document_loaders.recursive_url_loader import (
            RecursiveUrlLoader,
        )

        loader = RecursiveUrlLoader(url=url, max_depth=max_depth)
        print(f"Loading {url} recursively...")
        docs = list(loader.load())
        print(f"Loaded {len(docs)} docs.")

        print("Spinning up mapper...")
        # This breaks
        # remote_embedder = rh.module(DocEmbedder, env="langchain_embed_env")()
        # This breaks
        # remote_embedder = rh.module(DocEmbedder)()
        # This breaks
        # remote_embedder = rh.module(DocEmbedder)().to(self.system, name="doc_embedder", env=self.env)
        # remote_embedder = rh.module(DocEmbedder)().to(system=self.system, name="doc_embedder", env=self.env)
        # These all break because constructing before calling .to constructs in base_env which doesn't exist.
        # All our docker tests pass because we construct base_env in the cluster fixture.
        remote_embedder = rh.module(DocEmbedder, env=self.env)()

        mapper = rh.mapper(remote_embedder)
        mapper.add_replicas(3)

        print("Embedding docs...")
        mapper.map(docs, method="embed_docs")


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
        self.vectorstore = LanceDB(embedding=self.model)
        self.vectorstore.add_documents(splits)


if __name__ == "__main__":

    cluster = rh.cluster("rh-a10g4", instance_type="A10G:4").save().up_if_not()
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

    remote_module.embed_url_recursively("https://js.langchain.com/docs/")
