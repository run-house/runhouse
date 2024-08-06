from lancedb.pydantic import LanceModel, Vector


class Item(LanceModel):
    url: str
    page_content: str
    vector: Vector(1024)


class URLEmbedder:
    def __init__(self, **model_kwargs):
        import torch
        from sentence_transformers import SentenceTransformer

        self.model = torch.compile(SentenceTransformer(**model_kwargs))

    def encode_text(self, text: str, **embed_kwargs):
        embeddings = self.model.encode([text], **embed_kwargs)

        return embeddings[0]

    def embed_doc(self, url: str, **embed_kwargs):
        from langchain_community.document_loaders import WebBaseLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        docs = WebBaseLoader(
            web_paths=[url],
        ).load()
        split_docs = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        ).split_documents(docs)
        splits_as_str = [doc.page_content for doc in split_docs]
        embeddings = self.model.encode(splits_as_str, **embed_kwargs)
        items = [
            {"url": url, "page_content": doc.page_content, "vector": embeddings[index]}
            for index, doc in enumerate(split_docs)
        ]

        return items
