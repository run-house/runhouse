# Deploy Llama 2 7B Model with TGI on AWS Inferentia

See a more [rich explanation](https://www.run.house/examples/llama-tgi-inference-on-aws-inferentia)
of this example on our site.

This example demonstrates how to deploy a [Llama 7B model](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) using
[TGI](https://github.com/huggingface/optimum-neuron/tree/main/text-generation-inference) on AWS Inferentia
using Runhouse, specifically with the [AWS Neuron SDK](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/).

## Setup credentials and dependencies
Install the required dependencies:
```shell
$ pip install -r requirements.txt
```

We'll be launching an AWS Inferentia instance via [SkyPilot](https://github.com/skypilot-org/skypilot), so we need to
make sure our AWS credentials are set up:
```shell
$ aws configure
$ sky check
```
