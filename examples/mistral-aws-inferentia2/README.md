# Deploy Mistral's 7B Model with TGI on AWS Inferentia

See a more [rich explanation](https://www.run.house/examples/mistral-tgi-inference-on-aws-inferentia)
of this example on our site.

This example demonstrates how to deploy [Mistral's 7B Model](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1)
on AWS Inferentia using Runhouse.

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
