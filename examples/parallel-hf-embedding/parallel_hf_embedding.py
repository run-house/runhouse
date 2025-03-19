# # Embarrassingly parallel GPU Jobs: Batch Embeddings

# This example demonstrates how to use primitives to embed a large number of websites in parallel.
# We use a [BGE large model from Hugging Face](https://huggingface.co/BAAI/bge-large-en-v1.5) and load it via
# the `SentenceTransformer` class from the `huggingface` library.
#
# ## Some utility functions
#
# We import `kubetorch` and other utility libraries; only the ones that are needed to run the script locally.
# Imports of libraries that are needed on the remote machine (in this case, the `huggingface` dependencies)
# can happen within the functions that will be sent to the remote compute.
import time
from functools import partial
from multiprocessing.pool import ThreadPool
from urllib.parse import urljoin, urlparse

import kubetorch as kt

import requests
import torch
from bs4 import BeautifulSoup

# Then, we define an `extract_urls` function that will extract all URLs from a given URL, recursively up to a
# maximum depth. This'll be a useful helper function that we'll use to collect our list of URLs to embed.
def _extract_urls_helper(url, visited, original_url, max_depth=1, current_depth=1):
    """
    Extracts all URLs from a given URL, recursively up to a maximum depth.
    """
    if (
        url in visited
        or urlparse(url).netloc != urlparse(original_url).netloc
        or "redirect" in url
    ):
        return []

    visited.add(url)

    urls = [url]

    if current_depth <= max_depth:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        for link in soup.find_all("a"):
            href = link.get("href")
            if href:
                # Ignore links within the same page
                if href.startswith("#"):
                    continue

                if not href.startswith("http"):
                    href = urljoin(url, href)

                parsed_href = urlparse(href)
                if bool(parsed_href.scheme) and bool(parsed_href.netloc):
                    urls.extend(
                        _extract_urls_helper(
                            href, visited, original_url, max_depth, current_depth + 1
                        )
                    )
    return urls


def extract_urls(url, max_depth=1):
    visited = set()
    return _extract_urls_helper(
        url, visited, original_url=url, max_depth=max_depth, current_depth=1
    )


# ## Setting up the URL Embedder
#
# Next, we define a class that will hold the model and the logic to extract a document from a URL and embed it.
# Later, we'll instantiate this class with `kt.cls` and send it to the  remote compute.
class URLEmbedder:
    def __init__(self, **model_kwargs):
        from sentence_transformers import SentenceTransformer

        self.model = torch.compile(SentenceTransformer(**model_kwargs))

    def embed_docs(self, url: str, **embed_kwargs):
        from langchain_community.document_loaders import WebBaseLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        docs = WebBaseLoader(web_paths=[url]).load()
        splits = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        ).split_documents(docs)
        splits_as_str = [doc.page_content for doc in splits]
        embedding = self.model.encode(splits_as_str, **embed_kwargs)
        return embedding


# ## Launch compute and dispatch for execution
# Now, we define the main function that will run locally when we run this script, and set up
# our module on a remote compute.
if __name__ == "__main__":
    # Start by recursively extracting all children URLs from the given URL.
    url_to_recursively_embed = "https://en.wikipedia.org/wiki/Poker"
    start_time = time.time()
    urls = extract_urls(url_to_recursively_embed, max_depth=2)
    print(f"Time to extract {len(urls)} URLs: {time.time() - start_time}")

    # Now, we define compute with the desired resources. Each replica will run with one A10 GPU.
    # This is one major advantage of Kubetorch: you can use arbitrary resources from Kubernetes as if it were one opaque cluster,
    # and send things to it from your local machine. This is especially useful for embarrassingly parallel tasks
    # like this one.
    num_replicas = 4  # Number of models to load side by side

    img = kt.images.ubuntu().pip_install(
        [
            "langchain",
            "langchain-community",
            "langchainhub",
            "bs4",
            "sentence_transformers",
            "fake_useragent",
        ]
    )
    compute = kt.Compute(gpus="A10G:1", image=img)

    init_args = dict(
        model_name_or_path="BAAI/bge-large-en-v1.5",
        device="cuda",
    )

    # By distributing the service as a pool, we're creating a number of replicas which will be routed to by the
    # one service object we hold here.
    embedder = (
        kt.cls(URLEmbedder)
        .to(compute, init_args=init_args)
        .distribute(
            "pool", num_nodes=num_replicas, replicas_per_node=1, max_concurrency=32
        )
    )

    # ## Calling the modules in parallel
    # We'll simply use the `embed_docs` function on the remote module to embed all the URLs in parallel. Note that
    # we can call this function exactly as if it were a local module.
    with ThreadPool(num_replicas) as pool:
        embed = partial(
            embedder.embed_docs, normalize_embeddings=True, stream_logs=False
        )
        pool.map(embed, urls)
