FROM python:3.11-slim

ENV HOME /root

RUN apt-get update && apt-get install -y git

# Install required Python packages
RUN pip install --no-cache-dir skypilot[aws] awscli runhouse torch "ray[data, train]"
RUN apt-get update && apt-get install -y rsync openssh-client

# Copy your custom Python module
COPY /dlrm_data_prepoc.py /root/code/dlrm_data_preproc.py
COPY /dlrm_training.py /root/code/dlrm_training.py
COPY /dlrm_inference.py /root/code/dlrm_inference.py
COPY /requirements.txt /root/code/requirements.txt
COPY /__init__.py /root/code/__init__.py

RUN mkdir -p ~/.ssh
