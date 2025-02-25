import gc
import os
from pathlib import Path

import runhouse as rh
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
        print("Saved model weights and tokenizer on the cluster.")

    def load_fine_tuned_model(self):
        if not self.new_model_exists():
            raise FileNotFoundError(
                "No fine tuned model found on the cluster. "
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
                # Load the fine-tuned model saved on the cluster
                self.load_fine_tuned_model()

            if self.tokenizer is None:
                self.load_tokenizer()

            if self.pipeline is None or max_length != DEFAULT_MAX_LENGTH:
                self.load_pipeline(max_length)

            output = self.pipeline(f"<s>[INST] {query} [/INST]")
            return output[0]["generated_text"]


if __name__ == "__main__":
    img = (
        rh.Image(name="llamafinetuning")
        .pip_install(
            [
                "torch",
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

    num_nodes = 3

    # Requires access to a cloud account with the necessary permissions to launch compute.
    cluster = rh.compute(
        name=f"rh-L4x{num_nodes}",
        num_nodes=num_nodes,
        instance_type="L4:1",
        provider="aws",
        image=img,
        use_spot=True,
        autostop_mins=360,
    ).up_if_not()
    cluster.restart_server()
    fine_tuner_remote = rh.cls(FineTuner).to(cluster, name="ft_model")
    fine_tuner = fine_tuner_remote(name="ft_model_instance").distribute(
        "pytorch", num_replicas=num_nodes, replicas_per_node=1
    )

    fine_tuner.tune()

    query = "Who is Randy Jackson?"
    generated_text = fine_tuner.generate(query)
    print(generated_text)
