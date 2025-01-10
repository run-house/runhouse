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
    TrainingArguments,
)

from trl import SFTTrainer


DEFAULT_MAX_LENGTH = 200


## This is the class that we will send to remote and do all the work
## It is just a normal Python class defined without any special RH code or DSL
## The only thing to remember is that the work is happening remotely, so saving and accessing state is necessary within the instance
class FineTuner:
    def __init__(
        self,
        dataset_name="Shekswess/medical_llama3_instruct_dataset_short",
        base_model_name="meta-llama/Meta-Llama-3-8B-Instruct",
        fine_tuned_model_name="llama-3-8b-medical",
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.base_model_name = base_model_name
        self.fine_tuned_model_name = fine_tuned_model_name

        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self.epochs_trained = 0

        self.train_data = None
        self.test_data = None
        self.eval_results = None

    def load_base_model(self):
        # configure the model for efficient training
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=False,
        )

        # load the base model with the quantization configuration
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name, quantization_config=quant_config, device_map={"": 0}
        )

        self.model.config.use_cache = False
        self.model.config.pretraining_tp = 1

    def load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_name, trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

    def load_pipeline(self, max_length: int):
        from transformers import pipeline

        # Use the new fine-tuned model for generating text
        self.pipeline = pipeline(
            task="text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_length=max_length,
        )

    def load_train_and_test(self, dataset_name_new=None):
        if dataset_name_new is not None:
            self.dataset_name = dataset_name_new

        self.train_data = load_dataset(self.dataset_name, split="train")

        # Take 15% of train dataset and make it our test dataset
        test_size = int(0.15 * len(self.train_data))
        self.train_data = self.train_data.shuffle(seed=42)  # Seed for reproducibility
        self.test_data = self.train_data.select(range(test_size))
        self.train_data = self.train_data.select(range(test_size, len(self.train_data)))

    def load_fine_tuned_model(self):
        if not self.new_model_exists():
            raise FileNotFoundError(
                "No fine tuned model found on the cluster. "
                "Call the `tune` method to run the fine tuning."
            )

        self.model = AutoPeftModelForCausalLM.from_pretrained(
            self.fine_tuned_model_name,
            device_map={"": "cuda:0"},  # Loads model into GPU memory
            torch_dtype=torch.bfloat16,
        )

        self.model = self.model.merge_and_unload()

    def new_model_exists(self):
        return Path(f"~/{self.fine_tuned_model_name}").expanduser().exists()

    def training_params(self, num_train_epochs):
        return TrainingArguments(
            output_dir="./results_modified",
            num_train_epochs=num_train_epochs,
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
            report_to="tensorboard",
        )

    def sft_trainer(
        self, training_data, evaluation_data, peft_parameters, train_params
    ):
        # Set up the SFTTrainer with the model, training data, and parameters to learn from the new dataset
        return SFTTrainer(
            model=self.model,
            train_dataset=training_data,
            eval_dataset=evaluation_data,
            peft_config=peft_parameters,
            dataset_text_field="prompt",  # Dependent on your dataset
            tokenizer=self.tokenizer,
            args=train_params,
        )

    def tune(self, num_train_epochs=1):
        # Load the training data, tokenizer and model to be used by the trainer

        if self.train_data is None:
            self.load_train_and_test()

        if self.tokenizer is None:
            self.load_tokenizer()

        if self.model is None:
            self.load_base_model()

        # Use LoRA to update a small subset of the model's parameters
        peft_parameters = LoraConfig(
            lora_alpha=16, lora_dropout=0.1, r=8, bias="none", task_type="CAUSAL_LM"
        )

        train_params = self.training_params(num_train_epochs)
        trainer = self.sft_trainer(
            self.train_data, self.test_data, peft_parameters, train_params
        )

        # Force clean the pytorch cache
        gc.collect()
        torch.cuda.empty_cache()

        trainer.train()

        # Save the fine-tuned model's weights and tokenizer files on the cluster
        trainer.model.save_pretrained(self.fine_tuned_model_name)
        trainer.tokenizer.save_pretrained(self.fine_tuned_model_name)

        # self.eval_results = trainer.evaluate() #not enough memory on cluster

        # Clear VRAM from training
        del trainer
        del train_params
        gc.collect()
        torch.cuda.empty_cache()

        self.epochs_trained = self.epochs_trained + num_train_epochs
        print("Saved model weights and tokenizer on the cluster.")

    def generate(self, query: str, max_length: int = DEFAULT_MAX_LENGTH):
        if self.model is None:
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

    def get_training_status(self):
        status = {
            "tokenizer_loaded": self.tokenizer is not None,
            "model_loaded": self.model is not None,
            "pipeline_loaded": self.pipeline is not None,
            "training_completed": self.new_model_exists(),
            "epochs_trained": self.epochs_trained,
            "eval results": self.eval_results,
        }

        return status

    def save_model_s3(self, s3_bucket, s3_directory):
        try:  ## Avoid failing if you're just trying the example and don't have S3 setup
            import os

            import boto3

            s3_client = boto3.client("s3")

            for root, dirs, files in os.walk(self.fine_tuned_model_name):
                for file in files:
                    local_path = os.path.join(root, file)
                    relative_path = os.path.relpath(
                        local_path, self.fine_tuned_model_name
                    )
                    s3_path = os.path.join(s3_directory, relative_path)
                    s3_client.upload_file(local_path, s3_bucket, s3_path)

            print("uploaded fine tuned model")
        except:
            print("did not upload fine tuned model - check s3 configs")


# ### This is the code that is run to launch the cluster, do the dispatch, and execute there
if __name__ == "__main__":

    # ## Launch a cluster
    # You will need a HF_TOKEN as an env variable
    # Reqs will be installed by Runhouse on remote
    # We can also show you how to launch with a Docker container / conda env
    img = (
        rh.Image(name="ft_image")
        .install_packages(
            [
                "torch",
                "tensorboard",
                "scipy",
                "peft==0.4.0",
                "bitsandbytes==0.40.2",
                "transformers==4.31.0",
                "trl==0.4.7",
                "accelerate",
            ]
        )
        .sync_secrets(["huggingface"])
    )

    cluster = rh.cluster(
        name="a10g-rh", instance_type="A10G:1", memory="32+", provider="aws", image=img
    ).up_if_not()

    ## Here, we will access a remote instance of our fine tuner class
    fine_tuner_remote_name = "rh_finetuner"

    # We check if we have already created a "rh_finetuner" on the remote which is an *instance* of the remote fine tuner class
    fine_tuner_remote = cluster.get(fine_tuner_remote_name, default=None, remote=True)

    # If we have not, then we will send the local class to remote, and create an instance of it named "rh_finetuner"
    # If you disconnect locally after calling tune, you can simply reconnect to the remote object using this block from another local session
    if fine_tuner_remote is None:
        fine_tuner = rh.module(FineTuner).to(cluster, name="llama3-medical-model")
        fine_tuner_remote = fine_tuner(name=fine_tuner_remote_name)

    ## Once we have accessed the remote class, we can call against it as if it were a local object
    fine_tuner_remote.tune()

    # Once the fine tuner is complete, we can query against it
    query = "What's the best treatment for sunburn?"
    generated_text = fine_tuner_remote.generate(query)
    print(generated_text)
