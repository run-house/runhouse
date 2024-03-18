# Deploy Mistral's 7B Model with TGI on AWS EC2

See a more [rich explanation](https://www.run.house/examples/mistral-tgi-inference-on-aws-ec2)
of this example on our site.

This example demonstrates how to deploy a
[TGI model](https://huggingface.co/docs/text-generation-inference/messages_api) on AWS EC2 using Runhouse.
This example draws inspiration from
[Huggingface's tutorial on AWS SageMaker](https://huggingface.co/blog/text-generation-inference-on-inferentia2).
Zephyr is a 7B fine-tuned version of [Mistral's 7B-v0.1 model](https://huggingface.co/mistralai/Mistral-7B-v0.1).

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
