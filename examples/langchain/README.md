# Deploy a Langchain RAG as a service on AWS EC2

This is an example of easily deploying [Langchain's Quickstart RAG app](https://python.langchain.com/docs/use_cases/question_answering/quickstart)
as a service on AWS EC2 using Runhouse.

## Setup credentials and dependencies

Optionally, set up a virtual environment:
```shell
$ conda create -n langchain-rag python=3.9.15
$ conda activate langchain-rag
```
Install Runhouse, the only library needed to run this script locally:
```shell
$ pip install "runhouse[aws]"
```

We'll be launching an AWS EC2 instance via [SkyPilot](https://github.com/skypilot-org/skypilot), so we need to
make sure our AWS credentials are set up:
```shell
$ aws configure
$ sky check
```

We'll be hitting Open AI's API, so we need to set up our OpenAI API key:
```shell
$ export OPENAI_API_KEY=<your openai key>
```

After that, you can just run the example:
```shell
$ python langchain_rag.py
```
