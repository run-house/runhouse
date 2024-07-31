# Deploy and Train a Model with Torch
This example demonstrates how to use the `SimpleTrainer` class to train and test a machine learning model using PyTorch and the . The `SimpleTrainer` class handles model training, evaluation, and prediction tasks and shows you how you can send model classes to train and execute on remote compute. 

## Setup and Installation

Optionally, set up a virtual environment:
```shell
$ conda create -n simple-trainer python=3.10
$ conda activate simple-trainer
```
Install the necessary dependencies:
```shell
$ pip install -r requirements.txt
```

We'll be launching an AWS EC2 instance via [SkyPilot](https://github.com/skypilot-org/skypilot), so we need to
make sure our AWS credentials are set up (you can use any cloud with Runhouse):
```shell
$ aws configure
$ sky check
```

## Run the Python script

```shell
$ python TorchBasicExample-AWS.py
```
