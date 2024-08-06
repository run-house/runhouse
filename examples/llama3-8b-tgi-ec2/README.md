# Deploy Llama 3 8B with TGI on AWS EC2
This example demonstrates how to deploy a Meta Llama 3 8B model from Hugging Face with
[TGI](https://huggingface.co/docs/text-generation-inference/messages_api) on AWS EC2 using Runhouse.


Make sure to sign the waiver on the [Hugging Face model page](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) so that you can access it.

## Setup credentials and dependencies
Install the required dependencies:
```shell
$ pip install -r requirements.txt
```

We'll be launching an AWS EC2 instance via [SkyPilot](https://github.com/skypilot-org/skypilot), so we need to make sure our AWS credentials are set up:
```shell
$ aws configure
$ sky check
```
