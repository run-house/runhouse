# Deploy Mistral's 7B Model with TGI on AWS EC2

See a more [rich explanation](https://www.run.house/examples/mistral-tgi-inference-on-aws-ec2)
of this example on our site.

This example demonstrates how to deploy a [Mistral's 7B-v0.1 model](https://huggingface.co/mistralai/Mistral-7B-v0.1)
using [TGI](https://huggingface.co/docs/text-generation-inference/index) on AWS EC2 with Runhouse.

We'll use the [Messages API](https://huggingface.co/docs/text-generation-inference/messages_api) to send the prompt
to the model, and the OpenAI python client for making the requests.

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
