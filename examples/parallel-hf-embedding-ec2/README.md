# An embarrassingly parallel embedding task with Hugging Face models on AWS EC2

This example demonstrates how to use Runhouse primitives to embed a large number of websites in parallel.
We use a [BGE large model from Hugging Face](https://huggingface.co/BAAI/bge-large-en-v1.5) and load it via
the `SentenceTransformer` class from the `huggingface` library.

## Setup credentials and dependencies

Optionally, set up a virtual environment:
```shell
$ conda create -n parallel-embed python=3.9.15
$ conda activate parallel-embed
```
Install the few required dependencies:
```shell
$ pip install -r requirements.txt
```
We'll be launching an AWS EC2 instance via [SkyPilot](https://github.com/skypilot-org/skypilot), so we need to
make sure our AWS credentials are set up:
```shell
$ aws configure
$ sky check
```

## Some utility functions

We import `runhouse` and other utility libraries; only the ones that are needed to run the script locally.
Imports of libraries that are needed on the remote machine (in this case, the `huggingface` dependencies)
can happen within the functions that will be sent to the Runhouse cluster.
