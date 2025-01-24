# # Fine-Tune Llama 3 with LoRA on AWS EC2

# This example demonstrates how to fine-tune a Meta Llama 3 model with
# [LoRA](https://huggingface.co/docs/peft/main/en/conceptual_guides/lora) on AWS EC2 using Runhouse. See also our
# related post for [Llama 2 fine-tuning](https://www.run.house/examples/llama2-fine-tuning-with-lora).
#
# Make sure to sign the waiver on the [Hugging Face model](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)
# page so that you can access it.
#
# ## Set up credentials and dependencies
#
# Install the required dependencies:
# ```shell
# $ pip install "runhouse[aws]"
# ```
#
# If you are launching with open source (i.e. locally using your own credentials), you we need to
# make sure our AWS credentials are set up:
# ```shell
# $ aws configure
# $ sky check
# ```
#
# If you are launching with Runhouse, make sure you are logged in to runhouse:
# ```shell
# $ runhouse login
# ```
#
# To download the Llama 3 model on our EC2 instance, we need to set up a
# Hugging Face [token](https://huggingface.co/docs/hub/en/security-tokens):
# ```shell
# $ export HF_TOKEN=<your huggingface token>
# ```
#
# ## Create a model class
import gc
from pathlib import Path

import runhouse as rh

import torch
from datasets import load_dataset
from peft import AutoPeftModelForCausalLM, LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)
from trl import SFTConfig, SFTTrainer

# Next, we define a class that will hold the various methods needed to fine-tune the model.
# We'll later wrap this with `rh.module`. This is a Runhouse class that allows you to
# run code in your class on a remote machine.
#
# Learn more in the [Runhouse docs on functions and modules](/docs/tutorials/api-modules).

DEFAULT_MAX_LENGTH = 200


class FineTuner:
    def __init__(
        self,
        dataset_name="Shekswess/medical_llama3_instruct_dataset_short",
        base_model_name="meta-llama/Llama-3.2-3B-Instruct",
        fine_tuned_model_name="llama-3-3b-medical",
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.base_model_name = base_model_name
        self.fine_tuned_model_name = fine_tuned_model_name

        self.tokenizer = None
        self.base_model = None
        self.fine_tuned_model = None
        self.pipeline = None

    def load_base_model(self):
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=False,
        )  # configure the model for efficient training

        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name, quantization_config=quant_config, device_map="auto"
        )  # load the base model with the quantization configuration

        self.base_model.config.use_cache = False
        self.base_model.config.pretraining_tp = 1

    def load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_name, trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

    def load_pipeline(self, max_length: int):
        self.pipeline = pipeline(
            task="text-generation",
            model=self.fine_tuned_model,
            tokenizer=self.tokenizer,
            max_length=max_length,
        )  # Use the new fine-tuned model for generating text

    def load_dataset(self):
        return load_dataset(self.dataset_name, split="train")

    def load_fine_tuned_model(self):
        if not self.new_model_exists():
            raise FileNotFoundError(
                "No fine tuned model found on the cluster. "
                "Call the `tune` method to run the fine tuning."
            )

        self.fine_tuned_model = AutoPeftModelForCausalLM.from_pretrained(
            self.fine_tuned_model_name,
            device_map={"": "cuda:0"},  # Loads model into GPU memory
            torch_dtype=torch.bfloat16,
        )

        self.fine_tuned_model = self.fine_tuned_model.merge_and_unload()

    def new_model_exists(self):
        return Path(f"~/{self.fine_tuned_model_name}").expanduser().exists()

    def training_params(self):
        return SFTConfig(
            output_dir="./results_modified",
            num_train_epochs=1,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=1,
            optim="paged_adamw_32bit",
            save_steps=25,
            logging_steps=25,
            learning_rate=2e-4,
            weight_decay=0.001,
            fp16=False,
            bf16=False,
            max_grad_norm=0.3,
            max_steps=-1,
            warmup_ratio=0.03,
            group_by_length=True,
            lr_scheduler_type="constant",
            dataset_text_field="prompt",  # Dependent on your dataset
            report_to="tensorboard",
        )

    def sft_trainer(self, training_data, peft_parameters, train_params):
        # Set up the SFTTrainer with the model, training data, and parameters to learn from the new dataset
        return SFTTrainer(
            model=self.base_model,
            train_dataset=training_data,
            peft_config=peft_parameters,
            tokenizer=self.tokenizer,
            args=train_params,
        )

    def tune(self):
        if self.new_model_exists():
            return

        # Load the training data, tokenizer and model to be used by the trainer
        training_data = self.load_dataset()
        print("dataset loaded")
        if self.tokenizer is None:
            self.load_tokenizer()
        print("tokenizer loaded")
        if self.base_model is None:
            self.load_base_model()
        print("base model loaded")
        # Use LoRA to update a small subset of the model's parameters
        peft_parameters = LoraConfig(
            lora_alpha=16, lora_dropout=0.1, r=8, bias="none", task_type="CAUSAL_LM"
        )

        train_params = self.training_params()
        print("training to start")
        trainer = self.sft_trainer(training_data, peft_parameters, train_params)
        print("training")
        # Force clean the pytorch cache
        gc.collect()
        torch.cuda.empty_cache()

        trainer.train()

        # Save the fine-tuned model's weights and tokenizer files on the cluster
        trainer.model.save_pretrained(self.fine_tuned_model_name)
        trainer.tokenizer.save_pretrained(self.fine_tuned_model_name)

        # Clear VRAM from training
        del trainer
        del train_params
        del training_data
        self.base_model = None
        gc.collect()
        torch.cuda.empty_cache()

        print("Saved model weights and tokenizer on the cluster.")

    def generate(self, query: str, max_length: int = DEFAULT_MAX_LENGTH):
        if self.fine_tuned_model is None:
            # Load the fine-tuned model saved on the cluster
            self.load_fine_tuned_model()

        if self.tokenizer is None:
            self.load_tokenizer()

        if self.pipeline is None or max_length != DEFAULT_MAX_LENGTH:
            self.load_pipeline(max_length)

        # Format should reflect the format in the dataset_text_field in SFTTrainer
        output = self.pipeline(
            f"<|start_header_id|>system<|end_header_id|> Answer the question truthfully, you are a medical professional.<|eot_id|><|start_header_id|>user<|end_header_id|> This is the question: {query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        )
        return output[0]["generated_text"]


# ## Define Runhouse primitives
#
# Now, we define code that will run locally when we run this script and set up
# our Runhouse module on a remote cluster. First, we create a cluster with the desired instance type and provider.
# Our `instance_type` here is defined as `A10G:1`, which is the accelerator type and count that we need. We could
# alternatively specify a specific AWS instance type, such as `p3.2xlarge` or `g4dn.xlarge`.
#
# Learn more in the [Runhouse docs on clusters](/docs/tutorials/api-clusters).
#
# :::note{.info title="Note"}
# Make sure that all the following code runs within a `if __name__ == "__main__":` block, as shown below. Otherwise,
# the script code will run when Runhouse attempts to run code remotely. We'll break up the block in this example to
# improve readability.
# :::
if __name__ == "__main__":

    # First, we define the image for our module. This includes the required dependencies that need
    # to be installed on the remote machine, as well as any secrets that need to be synced up from local to remote.
    # Then, we launch a cluster with a GPU.
    # Finally, passing `huggingface` to the `sync_secrets` method will load the Hugging Face token we set up earlier.
    img = (
        rh.Image(name="llama2finetuning")
        .install_packages(
            [
                "torch",
                "tensorboard",
                "transformers",
                "bitsandbytes",
                "peft",
                "trl>0.12.0",
                "accelerate",
                "scipy",
            ]
        )
        .sync_secrets(["huggingface"])
    )

    cluster = rh.cluster(
        name="rh-a10x",
        gpus="A10G:1",
        memory="32+",
        image=img,
        provider="aws",
    ).up_if_not()
    cluster.restart_server()
    # Finally, we define our module and run it on the remote cluster. We construct it normally and then call
    # `to` to run it on the remote cluster. Alternatively, we could first check for an existing instance on the cluster
    # by calling `cluster.get(name="llama3-medical-model", remote = True)`. This would return the remote model after an initial run.
    # If we want to update the module each time we run this script, we prefer to use `to`.
    RemoteFineTuner = rh.module(FineTuner).to(cluster, name="FineTuner")
    fine_tuner_remote = RemoteFineTuner(name="llama3-medical-model")

    # ## Fine-tune the model on the cluster
    #
    # We can call the `tune` method on the model class instance as if it were running locally.
    # This will run the function on the remote cluster and return the response to our local machine automatically.
    # Further calls will also run on the remote machine, and maintain state that was updated between calls, like
    # `self.fine_tuned_model`.
    # Once the base model is fine-tuned, we save this new model on the cluster and use it to generate our text predictions.
    #
    # :::note{.info title="Note"}
    # For this example we are using a [small subset](https://huggingface.co/datasets/Shekswess/medical_llama3_instruct_dataset_short)
    # of 1000 samples that are already compatible with the model's prompt format.
    # :::
    fine_tuner_remote.tune()

    # ## Generate Text
    # Now that we have fine-tuned our model, we can generate text by calling the `generate` method with our query:
    query = "What's the best treatment for sunburn?"
    generated_text = fine_tuner_remote.generate(query)
    print(generated_text)
