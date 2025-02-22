# Fine Tune Llama3 with LoRA on AWS EC2

See a more [rich explanation](https://www.run.house/examples/llama3-fine-tuning-with-lora)
of this example on our site.

This example demonstrates how to fine tune a model using
[Llama3](https://huggingface.co/NousResearch/Meta-Llama-3-8B) and
[LoRA](https://huggingface.co/docs/peft/main/en/conceptual_guides/lora) on AWS EC2 using Runhouse.

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
