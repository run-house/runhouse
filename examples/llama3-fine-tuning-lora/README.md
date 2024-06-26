# Fine-Tune Llama 3 with LoRA on AWS EC2

This example demonstrates how to fine-tune a Llama 3 8B model using
[LoRA](https://huggingface.co/docs/peft/main/en/conceptual_guides/lora) on AWS EC2 using Runhouse.

Make sure to sign the waiver on the [Hugging Face model page](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)
so that you can access it.

## Setup credentials and dependencies

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
$ python llama3_fine_tuning.py
```
