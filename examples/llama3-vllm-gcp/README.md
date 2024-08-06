# Run Llama 3 8B Model Inference with vLLM on GCP

This example demonstrates how to run a Llama 3 8B model from Hugging Face with vLLM on GCP using Runhouse.

Make sure to sign the waiver on the [Hugging Face model page](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)
so that you can access it.

## Setup credentials and dependencies

Optionally, set up a virtual environment:
```shell
$ conda create -n llama3-rh python=3.9.15
$ conda activate llama3-rh
```

Install the required dependencies:

```shell
$ pip install -r requirements.txt
```

We'll be launching a GCP instance via [SkyPilot](https://github.com/skypilot-org/skypilot), so we need to make sure your credentials are set up. You may be prompted to pick a cloud project to use after running `gcloud init`. If you don't have one ready yet, you can connect one later by listing your projects with `gcloud projects list` and setting one with `gcloud config set project <PROJECT_ID>`.

```shell
$ gcloud init
$ gcloud auth application-default login
$ sky check
```

We'll be downloading the Llama 3 model from Hugging Face, so we need to set up our Hugging Face token:

```shell
$ export HF_TOKEN=<your huggingface token>
```

## Run the Python script

```shell
$ python llama3_vllm_gcp.py
```
