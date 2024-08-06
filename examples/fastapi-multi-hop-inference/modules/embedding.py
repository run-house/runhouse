import time

from lancedb.pydantic import LanceModel, Vector


class Item(LanceModel):
    url: str
    vector: Vector(1024)


class URLEmbedder:
    def __init__(self, **model_kwargs):
        import torch
        from sentence_transformers import SentenceTransformer

        self.model = torch.compile(SentenceTransformer(**model_kwargs))

    def embed_doc(self, url: str, **embed_kwargs):
        from langchain_community.document_loaders import WebBaseLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        start = time.time()
        docs = WebBaseLoader(
            web_paths=[url],
        ).load()
        splits = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        ).split_documents(docs)
        splits_as_str = [doc.page_content for doc in splits]
        downloaded = time.time()
        embedding = self.model.encode(splits_as_str, **embed_kwargs)
        return embedding, downloaded - start, time.time() - downloaded
