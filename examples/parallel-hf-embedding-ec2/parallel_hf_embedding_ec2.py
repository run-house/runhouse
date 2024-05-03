# # An embarrassingly parallel embedding task with Hugging Face models on AWS EC2

# This example demonstrates how to use Runhouse primitives to embed a large number of websites in parallel.
# We use a [BGE large model from Hugging Face](https://huggingface.co/BAAI/bge-large-en-v1.5) and load it via
# the `HuggingFaceBgeEmbeddings` class from the `langchain` library. We load 4 models onto an instance from AWS EC2
# that contains 4 A10G GPUs, and can make parallel calls to these models with URLs to embed.
#
# ## Setup credentials and dependencies
#
# Optionally, set up a virtual environment:
# ```shell
# $ conda create -n parallel-embed python=3.9.15
# $ conda activate parallel-embed
#
# ```
# Install the few required dependencies:
# ```shell
# $ pip install -r requirements.txt
# ```
#
# We'll be launching an AWS EC2 instance via [SkyPilot](https://github.com/skypilot-org/skypilot), so we need to
# make sure our AWS credentials are set up:
# ```shell
# $ aws configure
# $ sky check
# ```
# We'll be downloading the model from Hugging Face, so we need to set up our Hugging Face token:
# ```shell
# $ export HF_TOKEN=<your huggingface token>
# ```
#
# ## Some utility functions
#
# We import `runhouse` and other utility libraries; only the ones that are needed to run the script locally.
# Imports of libraries that are needed on the remote machine (in this case, the `langchain` dependencies)
# can happen within the functions that will be sent to the Runhouse cluster.

import asyncio
import time
from typing import List
from urllib.parse import urljoin, urlparse

import requests

import runhouse as rh
from bs4 import BeautifulSoup

# Then, we define some utility functions that we'll use for our embedding task.
def _extract_urls_helper(url, visited, original_url, max_depth=1, current_depth=1):
    """
    Extracts all URLs from a given URL, recursively up to a maximum depth.
    """
    if url in visited:
        return []

    visited.add(url)

    if urlparse(url).netloc != urlparse(original_url).netloc:
        return []

    if "redirect" in url:
        return []

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
# Later, we'll instantiate this class with `rh.module` and send it to the Runhouse cluster. Then, we can call
# the functions on this class remotely.
#
# Learn more in the [Runhouse docs on functions and modules](/docs/tutorials/api-modules).
class URLEmbedder:
    def __init__(self, gpu_number: int):
        self.model = None
        self.vectorstore = None
        self.gpu_number = gpu_number

    def initialize_model(self):
        if self.model is None:
            from langchain.embeddings import HuggingFaceBgeEmbeddings

            model_name = "BAAI/bge-large-en-v1.5"
            model_kwargs = {"device": f"cuda:{self.gpu_number}"}
            encode_kwargs = {
                "normalize_embeddings": True
            }  # set True to compute cosine similarity

            self.model = HuggingFaceBgeEmbeddings(
                model_name=model_name,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs,
            )

    def embed_docs(self, urls: List[str]):
        from langchain_community.document_loaders import WebBaseLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        self.initialize_model()

        # Occasionally, the loader will fail to load the URLs, so we catch the exception and return None.
        loader = WebBaseLoader(
            web_paths=urls,
        )
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        splits = text_splitter.split_documents(docs)
        splits_as_str = [doc.page_content for doc in splits]

        # Time the actual embedding
        start_time = time.time()
        embeddings = self.model.embed_documents(splits_as_str)
        print(
            f"Time to embed {len(splits_as_str)} text chunks: {time.time() - start_time}"
        )
        return embeddings


# ## Setting up Runhouse primitives
#
# Now, we define the main function that will run locally when we run this script, and set up
# our Runhouse module on a remote cluster. First, we create a cluster with the desired instance type and provider.
# Our `instance_type` here is defined as `A10G:4`, which is the accelerator type and count that we need. We could
# alternatively specify a specific AWS instance type, such as `p3.2xlarge` or `g4dn.xlarge`. If the cluster was
# already up (e.g. if we had run this script before), we just bring it up if it's not already up.
#
# Learn more in the [Runhouse docs on clusters](/docs/tutorials/api-clusters).
#
# :::note{.info title="Note"}
# Make sure that your code runs within a `if __name__ == "__main__":` block, as shown below. Otherwise,
# the script code will run when Runhouse attempts to import your code remotely.
# :::
async def main():
    cluster = rh.cluster("rh-a10g", instance_type="A10G:4").save().up_if_not()

    # We set up some parameters for our embedding task.
    num_replicas = 4  # Number of models to load side by side
    num_parallel_tasks = 128  # Number of parallel calls to make to the replicas
    url_to_recursively_embed = "https://en.wikipedia.org/wiki/Poker"

    # We recursively extract all children URLs from the given URL.
    start_time = time.time()
    urls = extract_urls(url_to_recursively_embed, max_depth=2)
    print(f"Time to extract {len(urls)} URLs: {time.time() - start_time}")

    # Generally, when using Runhouse, you would initialize an env with `rh.env`, and send your module to
    # that env. Each env runs in a *separate process* on the cluster. In this case, we want to have 4 copies of the
    # embedding model in separate processes, because we have 4 GPUs. We can do this by creating 4 separate envs
    # and 4 separate modules, each sent to a separate env. We do this in a loop here, with a list of dependencies
    # that we need remotely to run the module.
    #
    # Note that we first construct the module locally, then send it to the cluster with `get_or_to`. The instance
    # returned by `get_or_to` functions exactly the same as a local instance of the module, but when we call a
    # function (like `initialize_model`) on it, the function is run on the remote machine.
    start_time = time.time()
    replicas = []
    for i in range(num_replicas):
        env = rh.env(
            name=f"langchain_embed_env_{i}",
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
        local_url_embedder_module = rh.module(URLEmbedder, name=f"doc_embedder_{i}")(
            gpu_number=i
        )
        remote_url_embedder_module = local_url_embedder_module.get_or_to(
            system=cluster, env=env
        )
        remote_url_embedder_module.initialize_model()
        replicas.append(remote_url_embedder_module)
    print(f"Time to initialize {num_replicas} replicas: {time.time() - start_time}")

    # ## Calling the Runhouse modules in parallel
    # We'll simply use the `embed_docs` function on the remote module to embed all the URLs in parallel. Note that
    # we can call this function exactly as if it were a local module. In this case, because we are using the
    # `asyncio` library to make parallel calls, we need to use a special `run_async=True` argument to the
    # Runhouse function. This tells Runhouse to return a coroutine that we can await on, rather than making
    # a blocking network call to the server.
    start_time = time.time()
    futs = [
        asyncio.create_task(
            replicas[i % num_replicas].embed_docs([urls[i]], run_async=True)
        )
        for i in range(len(urls))
    ]

    all_embeddings = []
    failures = 0
    task_results = await asyncio.gather(*futs)
    for res in task_results:
        if res is not None:
            all_embeddings.extend(res)
        else:
            print("An embedding call failed.")
            failures += 1

    print(f"Received {len(all_embeddings)} total embeddings, with {failures} failures.")
    print(
        f"Embedded {len(urls)} docs across {num_replicas} replicas with {num_parallel_tasks} concurrent calls: {time.time() - start_time}"
    )


if __name__ == "__main__":
    asyncio.run(main())
