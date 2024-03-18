# Deploy Stable Diffusion XL 1.0 on AWS Inferentia

See a more [rich explanation](https://www.run.house/examples/stable-diffusion-xl-on-aws-inferentia)
of this example on our site.

This example demonstrates how to deploy a
[Stable Diffusion XL model from Hugging Face](https://huggingface.co/aws-neuron/stable-diffusion-xl-base-1-0-1024x1024)
on AWS Inferentia2 using
Runhouse. [AWS Inferentia2 instances](https://aws.amazon.com/ec2/instance-types/inf2/)
are powered by AWS Neuron, a custom hardware accelerator for machine learning
inference workloads. This example uses a model that was pre-compiled for AWS Neuron, and is available on the
Hugging Face Hub.

## Setup credentials and dependencies

Optionally, set up a virtual environment:
```shell
$ conda create -n rh-inf2 python=3.9.15
$ conda activate rh-inf2
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
$ python inf2_sdxl.py
```
