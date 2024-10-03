FROM python:3.11-slim

ENV HOME /root

RUN apt-get update && apt-get install -y git

# Install required Python packages
RUN pip install --no-cache-dir kfp kfp-kubernetes skypilot[aws] awscli runhouse torch torchvision
RUN pip install git+https://github.com/run-house/runhouse.git@main
RUN apt-get update && apt-get install -y rsync openssh-client

# Copy your custom Python module
COPY ../TorchBasicExample.py /root/training/TorchBasicExample.py
COPY ../requirements.txt /root/training/requirements.txt

RUN mkdir -p ~/.ssh
