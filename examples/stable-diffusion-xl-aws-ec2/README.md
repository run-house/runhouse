# Deploy Stable Diffusion XL 1.0 on AWS EC2

See a more [rich explanation](https://www.run.house/examples/stable-diffusion-xl-on-aws-ec2)
of this example on our site.

This example demonstrates how to deploy a
[Stable Diffusion XL model from Hugging Face](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
on AWS EC2 using Runhouse.

## Setup credentials and dependencies

Optionally, set up a virtual environment:
```shell
$ conda create -n rh-sdxl python=3.9.15
$ conda activate rh-sdxl
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
$ python sdxl.py
```
