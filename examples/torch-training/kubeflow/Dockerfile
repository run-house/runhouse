FROM python:3.11-slim

RUN apt-get update && apt-get install -y git

# Install required Python packages
RUN pip install --no-cache-dir kfp kfp-kubernetes skypilot[aws] awscli runhouse torch torchvision
RUN apt-get update && apt-get install -y rsync openssh-client

# Copy your custom Python module
COPY ../TorchBasicExample.py ~/root/TorchBasicExample.py

RUN mkdir -p ~/.ssh
