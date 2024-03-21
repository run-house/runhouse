# Deploy Llama 2 7B Model with TGI on AWS EC2

See a more [rich explanation](https://www.run.house/examples/llama-tgi-inference-on-aws-ec2)
of this example on our site.

This example demonstrates how to deploy a
[Llama 7B model](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) using
[TGI](https://huggingface.co/docs/text-generation-inference/messages_api) on AWS EC2 using Runhouse.

## Setup credentials and dependencies
Install the required dependencies:
```shell
$ pip install -r requirements.txt
```

We'll be launching an AWS EC2 instance via [SkyPilot](https://github.com/skypilot-org/skypilot), so we need to make
sure our AWS credentials are set up:
```shell
$ aws configure
$ sky check
```
