FROM python:3.11-slim

ENV HOME /root

RUN apt-get update && apt-get install -y git

# Install required Python packages
RUN pip install --no-cache-dir skypilot[aws] awscli runhouse torch ray 
RUN apt-get update && apt-get install -y rsync openssh-client

# Copy your custom Python module
COPY /dlrm_data_prepoc.py /root/dlrm_data_prepoc.py
COPY /dlrm_training.py /root/dlrm_training.py
COPY /dlrm_inference.py /root/dlrm_inference.py
COPY /requirements.txt /requirements.txt

RUN mkdir -p ~/.ssh