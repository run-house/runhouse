# ## Fine-Tune Llama 3 with LoRA

# This example demonstrates how to fine-tune a Meta Llama 3 model with
# [LoRA](https://huggingface.co/docs/peft/main/en/conceptual_guides/lora)
#
# Make sure to sign the waiver on the [Hugging Face model](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)
# page so that you can access it.
#
import gc
from pathlib import Path

import kubetorch as kt

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

# ## Create a Training Class
# Next, we define a class that will hold the various methods needed to fine-tune the model.
# We'll later wrap this with `kt.cls`. This is a regular class that allows you to
# run code in your class on a remote machine.
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
            per_device_train_batch_size=2,
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


# ## Define Compute and Execution
#
# Now, we define code that will run locally when we run this script and set up
# our module on a remote compute. First, we define compute with the desired instance type.
# Our `gpus` requirement here is defined as `L4:1`, which is the accelerator type and count that we need.

if __name__ == "__main__":

    # First, we define the image for our module. This includes the required dependencies that need
    # to be installed on the remote machine, as well as any secrets that need to be synced up from local to remote.
    # Then, we launch a cluster with a GPU.
    # Finally, passing `huggingface` to the `sync_secrets` method will load the Hugging Face token we set up earlier.
    img = (
        kt.images.pytorch()
        .pip_install(
            [
                "tensorboard",
                "transformers",
                "bitsandbytes",
                "peft",
                "trl",
                "accelerate",
                "scipy",
            ]
        )
        .sync_secrets(["huggingface"])
    )

    gpu = kt.Compute(
        gpus="L4:1",
        memory="32+",
        image=img,
    )

    # Finally, we define our module and run it on the remote gpu. We construct it normally and then call
    # `to` to run it on the remote cluster.
    fine_tuner_remote = kt.cls(FineTuner).to(gpu)

    # ## Fine-tune the model on the cluster
    #
    # We can call the `tune` method on the model class instance as if it were running locally.
    # This will run the function on the remote compute and return the response to our local machine automatically.
    # Further calls will also run on the remote compute, and maintain state that was updated between calls, like
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
