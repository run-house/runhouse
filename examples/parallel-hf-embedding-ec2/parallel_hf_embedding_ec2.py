# # Embarrassingly parallel GPU Jobs: Batch Embeddings

# This example demonstrates how to use Runhouse primitives to embed a large number of websites in parallel.
# We use a [BGE large model from Hugging Face](https://huggingface.co/BAAI/bge-large-en-v1.5) and load it via
# the `SentenceTransformer` class from the `huggingface` library.
#
# ## Setup credentials and dependencies
#
# Optionally, set up a virtual environment:
# ```shell
# $ conda create -n parallel-embed python=3.9.15
# $ conda activate parallel-embed
# ```
# Install the required dependencies:
# ```shell
# $ pip install "runhouse[aws]" torch beautifulsoup4 tqdm
# ```
#
# We'll be launching an AWS EC2 instance via [SkyPilot](https://github.com/skypilot-org/skypilot), so we need to
# make sure our AWS credentials are set up:
# ```shell
# $ aws configure
# $ sky check
# ```
#
# ## Some utility functions
#
# We import `runhouse` and other utility libraries; only the ones that are needed to run the script locally.
# Imports of libraries that are needed on the remote machine (in this case, the `huggingface` dependencies)
# can happen within the functions that will be sent to the Runhouse cluster.

import time
from functools import partial
from multiprocessing.pool import ThreadPool
from urllib.parse import urljoin, urlparse

import requests
import runhouse as rh
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
# Later, we'll instantiate this class with `rh.module` and send it to the Runhouse cluster. Then, we can call
# the functions on this class and they'll run on the remote machine.
#
# Learn more in the [Runhouse docs on functions and modules](/docs/tutorials/api-modules).
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


# :::note{.info title="Note"}
# Make sure that any code in your Python file that's meant to only run locally runs within
# a `if __name__ == "__main__":` block, as shown below. Otherwise, the script code will run
# when Runhouse attempts to import your code remotely.
# :::
if __name__ == "__main__":
    # ## Setting up Runhouse primitives
    #
    # Now, we define the main function that will run locally when we run this script, and set up
    # our Runhouse module on a remote cluster.

    # We set up some parameters for our embedding task.
    num_replicas = 4  # Number of models to load side by side
    url_to_recursively_embed = "https://en.wikipedia.org/wiki/Poker"

    # We recursively extract all children URLs from the given URL.
    start_time = time.time()
    urls = extract_urls(url_to_recursively_embed, max_depth=2)
    print(f"Time to extract {len(urls)} URLs: {time.time() - start_time}")

    # First, we create a cluster with the desired instance type and provider.
    # Our `instance_type` here is defined as `A10G:1`, which is the accelerator type and count we need We could
    # alternatively specify a specific AWS instance type, such as `p3.2xlarge` or `g4dn.xlarge`. However, we
    # provision `num_replicas` number of these instances. This gives us one Runhouse cluster that has
    # several separate GPU machines that it can access.
    #
    # This is one major advantage of Runhouse: you can use a multinode machine as if it were one opaque cluster,
    # and send things to it from your local machine. This is especially useful for embarrassingly parallel tasks
    # like this one. Note that it is also far easier to provision several A10G:1 machines as spot instances
    # than it is to provision a single A10G:4 machine, which is why we do it this way.
    #
    # Note that if the cluster was
    # already up (e.g. if we had run this script before), the code would just bring it up instead of creating a
    # new one, since we have given it a unique name `"rh-4xa10g"`.
    #
    # Learn more in the [Runhouse docs on clusters](/docs/tutorials/api-clusters).
    start_time = time.time()
    cluster = rh.cluster(
        f"rh-{num_replicas}xa10g",
        instance_type="A10G:1",
        num_nodes=num_replicas,
        spot=True,
        default_env=rh.env(
            reqs=[
                "langchain",
                "langchain-community",
                "langchainhub",
                "bs4",
                "sentence_transformers",
                "fake_useragent",
            ]
        ),
    ).up_if_not()

    # Generally, when using Runhouse, you would initialize an env with `rh.env`, and send your module to
    # that env. Each env runs in a *separate process* on the cluster. In this case, we want to have 4 copies of the
    # embedding model in separate processes, because we have 4 GPUs. We can do this by creating 4 separate envs
    # and 4 separate modules, each sent to a separate env. We do this in a loop here, with a list of dependencies
    # that we need remotely to run the module.
    #
    # In this case, each `env` is also on a separate machine, but we could also provision an A10G:4 instance,
    # and send all 4 envs to the same machine. Each env runs within a separate process on the machine, so they
    # won't interfere with each other.
    #
    # Note that we send the `URLEmbedder` class to the cluster, and then can construct our modules using the
    # returned "remote class" instead of the normal local class. These instances are then actually constructed
    # on the cluster, and any methods called on these instances would run on the cluster.
    RemoteURLEmbedder = rh.module(URLEmbedder).to(
        cluster, rh.process(name="langchain_embed_env")
    )
    remote_url_embedder = RemoteURLEmbedder(
        model_name_or_path="BAAI/bge-large-en-v1.5",
        device="cuda",
        name="doc_embedder",
    )
    embedder_pool = remote_url_embedder.distribute(
        "queue", replicas=num_replicas, replicas_per_node=1, max_concurrency=32
    )

    # ## Calling the Runhouse modules in parallel
    # We'll simply use the `embed_docs` function on the remote module to embed all the URLs in parallel. Note that
    # we can call this function exactly as if it were a local module. The semaphore and asyncio logic allows us
    # to run all the functions in parallel, up to a maximum total concurrency.
    with ThreadPool(num_replicas) as pool:
        embed = partial(
            embedder_pool.embed_docs, normalize_embeddings=True, stream_logs=False
        )
        pool.map(embed, urls)
