# Run several Hugging Face embedding models on AWS EC2 using Runhouse & Langchain

This example demonstrates how to use Runhouse primitives to embed a large number of websites in parallel.
We use a [BGE large model from Hugging Face](https://huggingface.co/BAAI/bge-large-en-v1.5) and load it via
the `HuggingFaceBgeEmbeddings` class from the `langchain` library. We load 4 models onto an instance from AWS EC2
that contains 4 A10G GPUs, and can make parallel calls to these models with URLs to embed.

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
We'll be downloading the model from Hugging Face, so we need to set up our Hugging Face token:
```shell
$ export HF_TOKEN=<your huggingface token>
```

After that, you can just run the example:
```shell
$ python llama3_ec2.py
```
