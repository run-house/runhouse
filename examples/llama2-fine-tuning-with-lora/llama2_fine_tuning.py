# # Fine Tune Llama 2 with LoRA on AWS EC2

# This example demonstrates fine tune a model using
# [Llama 2](https://huggingface.co/NousResearch/Llama-2-7b-chat-hf) and
# [LoRA](https://huggingface.co/docs/peft/main/en/conceptual_guides/lora) on AWS EC2 using Runhouse.
#
# ## Setup credentials and dependencies
#
# Install the required dependencies:
# ```shell
# $ pip install "runhouse[aws]"
# ```
#
# We'll be launching an AWS EC2 instance via [SkyPilot](https://github.com/skypilot-org/skypilot), so we need to
# make sure our AWS credentials are set up:
# ```shell
# $ aws configure
# $ sky check
# ```
#
# ## Setting up a model class
#
# We import runhouse, the only required library we need locally:
import runhouse as rh

# Next, we define a class that will hold the various methods needed to fine tune the model.
# You'll notice this class inherits from `rh.Module`.
# This is a Runhouse class that allows you to
# run code in your class on a remote machine.
#
# Learn more in the [Runhouse docs on functions and modules](/docs/tutorials/api-modules).

DEFAULT_MAX_LENGTH = 200


class FineTuner(rh.Module):
    def __init__(
        self,
        dataset_name="mlabonne/guanaco-llama2-1k",
        base_model_name="NousResearch/Llama-2-7b-chat-hf",
        fine_tuned_model_name="llama-2-7b-enhanced",
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
        import torch
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig

        # configure the model for efficient training
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=False,
        )

        # load the base model with the quantization configuration
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name, quantization_config=quant_config, device_map={"": 0}
        )

        self.base_model.config.use_cache = False
        self.base_model.config.pretraining_tp = 1

    def load_tokenizer(self):
        from transformers import AutoTokenizer

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
            model=self.fine_tuned_model,
            tokenizer=self.tokenizer,
            max_length=max_length,
        )

    def load_dataset(self):
        from datasets import load_dataset

        return load_dataset(self.dataset_name, split="train")

    def load_fine_tuned_model(self):
        import torch
        from peft import AutoPeftModelForCausalLM

        if not self.new_model_exists():
            raise FileNotFoundError(
                "No fine tuned model found on the cluster. "
                "Call the `tune` method to run the fine tuning."
            )

        self.fine_tuned_model = AutoPeftModelForCausalLM.from_pretrained(
            self.fine_tuned_model_name,
            device_map={"": "cuda:0"},
            torch_dtype=torch.bfloat16,
        )
        self.fine_tuned_model = self.fine_tuned_model.merge_and_unload()

    def new_model_exists(self):
        from pathlib import Path

        return Path(f"~/{self.fine_tuned_model_name}").expanduser().exists()

    def training_params(self):
        from transformers import TrainingArguments

        return TrainingArguments(
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
            report_to="tensorboard",
        )

    def sft_trainer(self, training_data, peft_parameters, train_params):
        from trl import SFTTrainer

        # Set up the SFTTrainer with the model, training data, and parameters to learn from the new dataset
        return SFTTrainer(
            model=self.base_model,
            train_dataset=training_data,
            peft_config=peft_parameters,
            dataset_text_field="text",
            tokenizer=self.tokenizer,
            args=train_params,
        )

    def tune(self):
        import gc

        import torch
        from peft import LoraConfig

        if self.new_model_exists():
            return

        # Load the training data, tokenizer and model to be used by the trainer
        training_data = self.load_dataset()

        if self.tokenizer is None:
            self.load_tokenizer()

        if self.base_model is None:
            self.load_base_model()

        # Use LoRA to update a small subset of the model's parameters
        peft_parameters = LoraConfig(
            lora_alpha=16, lora_dropout=0.1, r=8, bias="none", task_type="CAUSAL_LM"
        )

        train_params = self.training_params()
        trainer = self.sft_trainer(training_data, peft_parameters, train_params)

        # Force clean the pytorch cache
        gc.collect()
        torch.cuda.empty_cache()

        trainer.train()

        # Save the fine-tuned model's weights and tokenizer files on the cluster
        trainer.model.save_pretrained(self.fine_tuned_model_name)
        trainer.tokenizer.save_pretrained(self.fine_tuned_model_name)

        print("Saved model weights and tokenizer on the cluster.")

    def generate(self, query: str, max_length: int = DEFAULT_MAX_LENGTH):
        if self.fine_tuned_model is None:
            # Load the fine-tuned model saved on the cluster
            self.load_fine_tuned_model()

        if self.tokenizer is None:
            self.load_tokenizer()

        if self.pipeline is None or max_length != DEFAULT_MAX_LENGTH:
            self.load_pipeline(max_length)

        output = self.pipeline(f"<s>[INST] {query} [/INST]")
        return output[0]["generated_text"]


# ## Setting up Runhouse primitives
#
# Now, we define the main function that will run locally when we run this script, and set up
# our Runhouse module on a remote cluster. First, we create a cluster with the desired instance type and provider.
# Our `instance_type` here is defined as `A10G:1`, which is the accelerator type and count that we need. We could
# alternatively specify a specific AWS instance type, such as `p3.2xlarge` or `g4dn.xlarge`.
#
# Learn more in the [Runhouse docs on clusters](/docs/tutorials/api-clusters).
#
# :::note{.info title="Note"}
# Make sure that your code runs within a `if __name__ == "__main__":` block, as shown below. Otherwise,
# the script code will run when Runhouse attempts to run code remotely.
# :::
if __name__ == "__main__":
    cluster = rh.cluster(name="rh-a10x", instance_type="A10G:1", provider="aws")

    # Next, we define the environment for our module. This includes the required dependencies that need
    # to be installed on the remote machine, as well as any secrets that need to be synced up from local to remote.
    # Passing `huggingface` to the `secrets` parameter will load the Hugging Face token we set up earlier.
    #
    # Learn more in the [Runhouse docs on envs](/docs/tutorials/api-envs).
    env = rh.env(
        name="ft_env",
        reqs=[
            "torch",
            "tensorboard",
            "scipy",
            "peft==0.4.0",
            "bitsandbytes==0.40.2",
            "transformers==4.31.0",
            "trl==0.4.7",
            "accelerate",
        ],
    )

    # Finally, we define our module and run it on the remote cluster. We construct it normally and then call
    # `get_or_to` to run it on the remote cluster. Using `get_or_to` allows us to load the exiting Module
    # by the name `ft_env` if it was already put on the cluster. If we want to update the module each
    # time we run this script, we can use `to` instead of `get_or_to`.
    #
    # Note that we also pass the `env` object to the `get_or_to` method, which will ensure that the environment is
    # set up on the remote machine before the module is run.
    fine_tuner_remote = FineTuner().get_or_to(cluster, env=env, name="ft_model")

    # ## Fine-tuning the model on the cluster
    #
    # We can call the `tune` method on the model class instance if it were running locally.
    # This will run the function on the remote cluster and return the response to our local machine automatically.
    # Further calls will also run on the remote machine, and maintain state that was updated between calls, like
    # `self.model`.
    # Once the model is fine-tuned, we save this new model on the cluster and use it to generate our text predictions.
    #
    # :::note{.info title="Note"}
    # For this example we are using a [small subset](https://huggingface.co/datasets/mlabonne/guanaco-llama2-1k)
    # of 1,000 samples that are already compatible with the model's prompt format.
    # :::
    fine_tuner_remote.tune()

    # ## Generate Text
    # Now that we have fine-tuned our model, we can generate text by calling the `generate` method with our query:
    query = "Who is Randy Jackson?"
    generated_text = fine_tuner_remote.generate(query)
    print(generated_text)
