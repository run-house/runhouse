FROM python:3.11-slim


# Install required Python packages
RUN pip install --no-cache-dir kfp kfp-kubernetes runhouse skypilot[aws] awscli
RUN apt-get update && apt-get install -y rsync openssh-client

# Copy your custom Python module
COPY ../TorchBasicExample.py /TorchBasicExample.py

RUN mkdir -p ~/.ssh
