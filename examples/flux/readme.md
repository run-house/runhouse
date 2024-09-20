# Deploy Flux1 Schnell on AWS EC2

See a more [rich explanation](https://www.run.house/examples/host-and-run-flux1-image-genai-aws)
of this example on our site.

This example demonstrates how to deploy a
[Flux.1 model from Hugging Face](https://huggingface.co/black-forest-labs/FLUX.1-schnell)
on AWS EC2 using Runhouse. Schnell is smaller than their Dev version, but fits easily onto a single A10G.

## Setup credentials and dependencies

Optionally, set up a virtual environment:
```shell
$ conda create -n rh-flux python=3.11
$ conda activate rh-flux
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

After that, you can just run the example:
```shell
$ python flux.py
```
