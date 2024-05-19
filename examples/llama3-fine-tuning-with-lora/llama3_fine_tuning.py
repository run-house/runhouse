# # Fine Tune Llama3 with LoRA on AWS EC2

# This example demonstrates fine tune a model using
# [Llama3](https://huggingface.co/meta-llama/Meta-Llama-3-8B) and
# [LoRA with Torchtune](https://pytorch.org/torchtune/stable/tutorials/llama3.html) on AWS EC2 using Runhouse.
#
# ## Setup credentials and dependencies
#
# ```
# Install the few required dependencies:
# ```shell
# $ pip install -r requirements.txt
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
# We import runhouse, the only required library we need to install locally:
from pathlib import Path

import runhouse as rh


# Next, we define a class that will hold the various methods needed to fine tune the model.
# You'll notice this class inherits from `rh.Module`.
# This is a Runhouse class that allows you to
# run code in your class on a remote machine.
#
# Learn more in the [Runhouse docs on functions and modules](/docs/tutorials/api-modules).


class FineTuner(rh.Module):
    def __init__(
        self,
        base_model_name="meta-llama/Meta-Llama-3-8B",
        output_dir="~/results",
        generate_config="~/custom_generation_config.yaml",
        **model_kwargs,
    ):
        super().__init__()
        self.base_model_name = base_model_name
        self.model_kwargs = model_kwargs
        self.output_dir = output_dir
        self.generate_config = generate_config

        self.model = None
        self.tokenizer = None

    @property
    def checkpoint_dir(self):
        return str(Path(f"{self.base_checkpoint_dir}/original").expanduser())

    @property
    def base_checkpoint_dir(self):
        return str(Path(self.output_dir).expanduser())

    @property
    def path_to_config(self):
        return Path(self.generate_config).expanduser()

    def load_base_model(self):
        import subprocess

        path_to_token = Path("~/.cache/huggingface/token").expanduser()
        if not path_to_token.exists():
            raise FileNotFoundError(
                f"Hugging Face token not found in path: {path_to_token}."
            )

        with open(path_to_token, "r") as f:
            hf_token = f.read().strip()

        download_cmd = [
            "tune",
            "download",
            self.base_model_name,
            "--output-dir",
            self.base_checkpoint_dir,
            "--hf-token",
            hf_token,
        ]

        try:
            subprocess.run(download_cmd, check=True)
        except subprocess.CalledProcessError as e:
            raise e

    def tune(self):
        if not Path(self.checkpoint_dir).exists():
            self.load_base_model()

        if not Path(self.path_to_config).expanduser().exists():
            self.save_eval_config()

        if not Path(self.base_checkpoint_dir).exists():
            import subprocess

            command = [
                "tune",
                "run",
                "lora_finetune_single_device",
                "--config",
                "llama3/8B_lora_single_device",
                f"checkpointer.checkpoint_dir={self.checkpoint_dir}",
                f"tokenizer.path={self.checkpoint_dir}/tokenizer.model",
                f"checkpointer.output_dir={self.checkpoint_dir}",
            ]

            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"Fine-tuning failed: {result.stderr}")

    def save_eval_config(self):
        import yaml

        config_data = {
            "model": {"_component_": "torchtune.models.llama3.llama3_8b"},
            "checkpointer": {
                "_component_": "torchtune.utils.FullModelMetaCheckpointer",
                "checkpoint_dir": self.checkpoint_dir,
                "checkpoint_files": ["consolidated.00.pth"],
                "output_dir": self.checkpoint_dir,
                "model_type": "LLAMA3",
            },
            "device": "cuda",
            "dtype": "bf16",
            "seed": 1234,
            "tokenizer": {
                "_component_": "torchtune.models.llama3.llama3_tokenizer",
                "path": f"{self.checkpoint_dir}/tokenizer.model",
            },
            "prompt": "Hello, my name is",
            "max_new_tokens": 300,
            "temperature": 0.6,
            "top_k": 300,
            "quantizer": None,
        }

        with open(self.path_to_config, "w") as file:
            yaml.dump(config_data, file, default_flow_style=False)

    def generate(self, prompt: str, max_generated_tokens: int = 300):
        import torch
        from omegaconf import OmegaConf
        from torchtune.models.llama3 import llama3_8b, llama3_tokenizer
        from torchtune.utils import generate

        if not self.path_to_config.exists():
            raise FileNotFoundError(
                f"Config file not found at {self.path_to_config}. Please run `tune` first."
            )

        config = OmegaConf.load(self.path_to_config)

        # Free up GPU memory
        torch.cuda.empty_cache()

        if self.model is None:
            # Load the tokenizer and model
            self.tokenizer = llama3_tokenizer(path=config.tokenizer.path)
            self.model = llama3_8b()

            # Move model to device using mixed precision if possible
            with torch.cuda.amp.autocast():
                self.model.to(config.device)

        encoded_prompt = (
            torch.tensor(self.tokenizer.encode(prompt, add_bos=True, add_eos=False))
            .unsqueeze(0)
            .to(config.device)
        )

        # Use mixed precision
        with torch.cuda.amp.autocast():
            # Generate the tokens
            generated_tokens = generate(
                model=self.model,
                prompt=encoded_prompt,
                max_generated_tokens=max_generated_tokens,
                **self.model_kwargs,
            )

        res = self.tokenizer.decode(generated_tokens[0].tolist())
        return res


# ## Setting up Runhouse primitives
#
# Now, we define the main function that will run locally when we run this script, and set up
# our Runhouse module on a remote cluster. First, we create a cluster with the desired instance type and provider.
# Our `instance_type` here is defined as `g5.4xlarge`, which is an AWS instance type with a GPU. We can alternatively
# specify an accelerator type and count, such as `A100:1`, and any instance type with those specifications will be used.
#
# Learn more in the [Runhouse docs on clusters](/docs/tutorials/api-clusters).
#
# :::note{.info title="Note"}
# Make sure that your code runs within a `if __name__ == "__main__":` block, as shown below. Otherwise,
# the script code will run when Runhouse attempts to run code remotely.
# :::
if __name__ == "__main__":
    # For GCP, Azure, or Lambda Labs
    # rh.cluster(name='rh-a10x', instance_type='A100:1').save()

    # For AWS (single A100s not available, base A10G may have insufficient CPU RAM)
    cluster = rh.cluster(
        name="rh-a10x-torchtune", instance_type="g5.4xlarge", provider="aws"
    ).up_if_not()

    # Next, we define the environment for our module. This includes the required dependencies that need
    # to be installed on the remote machine, as well as any secrets that need to be synced up from local to remote.
    # Passing `huggingface` to the `secrets` parameter will load the Hugging Face token onto the cluster, which
    # is needed for loading the model.
    #
    # Learn more in the [Runhouse docs on envs](/docs/tutorials/api-envs).
    env = rh.env(
        name="ft_env",
        reqs=["torchtune", "omegaconf"],
        working_dir="./",
        secrets=["huggingface"],
    )

    # Finally, we define our module and run it on the remote cluster. We construct it normally and then call
    # `get_or_to` to run it on the remote cluster. Using `get_or_to` allows us to load the exiting Module
    # by the name `ft_env` if it was already put on the cluster. If we want to update the module each
    # time we run this script, we can use `to` instead of `get_or_to`.
    #
    # Note that we also pass the `env` object to the `get_or_to` method, which will ensure that the environment is
    # set up on the remote machine before the module is run.
    fine_tuner_remote = FineTuner().to(cluster, env=env, name="ft_model")

    # ## Fine-tuning the model on the cluster
    #
    # We can call the `tune` method on the model class instance if it were running locally.
    # This will run the function on the remote cluster and return the response to our local machine automatically.
    # Further calls will also run on the remote machine, and maintain state that was updated between calls, like
    # `self.model`.
    # Once the model is fine-tuned, we save this new model on the cluster and use it to generate our text predictions.
    #
    fine_tuner_remote.tune()

    # ## Generate Text
    # Now that we have fine-tuned our model, we can generate text by calling the `generate` method with our prompt:
    prompt = "Hello, my name is"
    generated_text = fine_tuner_remote.generate(prompt)
    print(generated_text)
