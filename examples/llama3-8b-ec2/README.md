# Deploy Llama3 8B Chat Model Inference on AWS EC2

See a more [rich explanation](https://www.run.house/examples/llama3-8b-chat-model-inference-aws-ec2)
of this example on our site.

This example demonstrates how to deploy a
[LLama2 13B model from Hugging Face](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf)
on AWS EC2 using Runhouse.

Make sure to sign the waiver on the model page so that you can access it.

## Setup credentials and dependencies

Optionally, set up a virtual environment:
```shell
$ conda create -n llama3-rh python=3.9.15
$ conda activate llama3-rh
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
We'll be downloading the Llama2 model from Hugging Face, so we need to set up our Hugging Face token:
```shell
$ export HF_TOKEN=<your huggingface token>
```

After that, you can just run the example:
```shell
$ python llama3_ec2.py
```
