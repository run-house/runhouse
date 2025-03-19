# ## Fine-Tune Llama 3 with LoRA on AWS EC2

import gc
import os
from pathlib import Path

import kubetorch as kt
import torch
import torch.distributed as dist

from datasets import load_dataset
from peft import AutoPeftModelForCausalLM, LoraConfig

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)
from trl import SFTConfig, SFTTrainer


DEFAULT_MAX_LENGTH = 200

# ## Define the FineTuner class
# The `FineTuner` class will hold the methods needed to fine-tune the model and run inference text.
# This is regular Python using standard HF Transformers and TRL methods to fine-tune the model.
# We will write code and iterate locally, but dispatch this to distributed remote compute using Kubetorch, and
# run the actual training over arbitrary scale.
class FineTuner:
    def __init__(
        self,
        dataset_name="mlabonne/guanaco-llama2-1k",
        base_model_name="meta-llama/Llama-3.2-3B-Instruct",
        fine_tuned_model_name="llama-3-3b-enhanced",
    ):
        self.dataset_name = dataset_name
        self.base_model_name = base_model_name
        self.fine_tuned_model_name = fine_tuned_model_name

        self.tokenizer = None
        self.base_model = None
        self.fine_tuned_model = None
        self.pipeline = None
        self.rank = None

    def load_base_model(self):
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=False,
        )

        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name, quantization_config=quant_config, device_map="auto"
        )

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
        )

    def load_dataset(self):
        return load_dataset(self.dataset_name, split="train")

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
            dataset_text_field="text",
            group_by_length=True,
            lr_scheduler_type="constant",
            report_to="tensorboard",
        )

    def sft_trainer(self, training_data, peft_parameters, train_params):
        return SFTTrainer(
            model=self.base_model,
            train_dataset=training_data,
            peft_config=peft_parameters,
            tokenizer=self.tokenizer,
            args=train_params,
        )

    def tune(self):
        # Force clean the pytorch cache
        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()

        os.environ["OMP_NUM_THREADS"] = "1"

        torch.distributed.init_process_group(backend="nccl")
        self.rank = dist.get_rank()
        print(self.rank)

        training_data = self.load_dataset()

        if self.tokenizer is None:
            self.load_tokenizer()

        if self.base_model is None:
            self.load_base_model()

        peft_parameters = LoraConfig(
            lora_alpha=16, lora_dropout=0.1, r=8, bias="none", task_type="CAUSAL_LM"
        )

        train_params = self.training_params()
        trainer = self.sft_trainer(training_data, peft_parameters, train_params)

        trainer.train()
        self.save_model(trainer)

    def save_model(self, trainer):
        self.fine_tuned_model = trainer.model
        trainer.model.save_pretrained(self.fine_tuned_model_name)
        trainer.tokenizer.save_pretrained(self.fine_tuned_model_name)
        print("Saved model weights and tokenizer on the remote compute.")

    def load_fine_tuned_model(self):
        if not self.new_model_exists():
            raise FileNotFoundError(
                "No fine tuned model found on the remote compute. "
                "Call the `tune` method to run the fine tuning."
            )

        self.fine_tuned_model = AutoPeftModelForCausalLM.from_pretrained(
            self.fine_tuned_model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        self.fine_tuned_model = self.fine_tuned_model.merge_and_unload()

    def new_model_exists(self):
        return Path(f"~/{self.fine_tuned_model_name}").expanduser().exists()

    def generate(self, query: str, max_length: int = DEFAULT_MAX_LENGTH):
        if self.rank == 0:
            if self.fine_tuned_model is None:
                # Load the fine-tuned model saved on the remote compute
                self.load_fine_tuned_model()

            if self.tokenizer is None:
                self.load_tokenizer()

            if self.pipeline is None or max_length != DEFAULT_MAX_LENGTH:
                self.load_pipeline(max_length)

            output = self.pipeline(f"<s>[INST] {query} [/INST]")
            return output[0]["generated_text"]


# ## Define Compute and Execution
#
# Now, we define code that will run locally when we run this script and set up
# our fine tuner module on a remote compute. First, we define compute with the desired requirements for a node.
# Our `gpus` requirement here is defined as `L4:1`, which is the accelerator type and count that we need.
# We will set the number of nodes to 4 and our training will be distributed across these nodes.
if __name__ == "__main__":
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
                "datasets",
            ]
        )
        .sync_secrets(["huggingface"])
    )

    num_nodes = 4
    replicas_per_node = 1
    gpus_per_node = 1

    compute = kt.Compute(
        gpus=f"L4:{gpus_per_node}",
        image=img,
    ).up_if_not()

    fine_tuner = (
        kt.cls(FineTuner)
        .to(compute)
        .distribute(
            "pytorch", num_replicas=num_nodes, replicas_per_node=replicas_per_node
        )
    )

    fine_tuner.tune()

    query = "Who is Randy Jackson?"
    generated_text = fine_tuner.generate(query)
    print(generated_text)
