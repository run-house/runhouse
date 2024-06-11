# # Deploy Llama 3 8B Chat Model Inference on AWS EC2

# This example demonstrates how to deploy a
# [LLama 3 8B model from Hugging Face](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)
# on AWS EC2 using Runhouse.
#
# Make sure to sign the waiver on the model page so that you can access it.
#
# ## Setup credentials and dependencies
#
# Optionally, set up a virtual environment:
# ```shell
# $ conda create -n llama3-rh python=3.9.15
# $ conda activate llama3-rh
# ```
# Install the required dependencies:
# ```shell
# $ pip install "runhouse[aws]" torch
# ```
#
# We'll be launching an AWS EC2 instance via [SkyPilot](https://github.com/skypilot-org/skypilot), so we need to
# make sure our AWS credentials are set up:
# ```shell
# $ aws configure
# $ sky check
# ```
# We'll be downloading the Llama 3 model from Hugging Face, so we need to set up our Hugging Face token:
# ```shell
# $ export HF_TOKEN=<your huggingface token>
# ```
#
# ## Setting up a model class
#
# We import `runhouse` and `torch`, because that's all that's needed to run the script locally.
# The actual transformers imports can happen within the functions
# that will be sent to the Runhouse cluster; we don't need those locally.

import runhouse as rh
import torch

# Next, we define a class that will hold the model and allow us to send prompts to it.
# You'll notice this class inherits from `rh.Module`.
# This is a Runhouse class that allows you to
# run code in your class on a remote machine.
#
# Learn more in the [Runhouse docs on functions and modules](/docs/tutorials/api-modules).
class HFChatModel(rh.Module):
    def __init__(self, model_id="meta-llama/Meta-Llama-3-8B-Instruct", **model_kwargs):
        super().__init__()
        self.model_id, self.model_kwargs = model_id, model_kwargs
        self.pipeline = None

    def load_model(self):
        import transformers

        self.pipeline = transformers.pipeline(
            "text-generation",
            model=self.model_id,
            model_kwargs=self.model_kwargs,
            device="cuda",
        )

    def predict(self, prompt_text, **inf_kwargs):
        if not self.pipeline:
            self.load_model()

        messages = [
            {
                "role": "system",
                "content": "You are a pirate chatbot who always responds in pirate speak!",
            },
            {"role": "user", "content": prompt_text},
        ]

        prompt = self.pipeline.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        terminators = [
            self.pipeline.tokenizer.eos_token_id,
            self.pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]

        outputs = self.pipeline(
            prompt,
            max_new_tokens=256,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
        return outputs[0]["generated_text"][len(prompt) :]


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
    gpu = rh.cluster(
        name="rh-a10x", instance_type="A10G:1", memory="32+", provider="aws"
    )

    # Next, we define the environment for our module. This includes the required dependencies that need
    # to be installed on the remote machine, as well as any secrets that need to be synced up from local to remote.
    # Passing `huggingface` to the `secrets` parameter will load the Hugging Face token we set up earlier.
    #
    # Learn more in the [Runhouse docs on envs](/docs/tutorials/api-envs).
    env = rh.env(
        reqs=[
            "torch",
            "transformers",
            "accelerate",
            "bitsandbytes",
            "safetensors",
            "scipy",
        ],
        secrets=["huggingface"],  # Needed to download Llama 3 from HuggingFace
        name="llama3inference",
        working_dir="./",
    )

    # Finally, we define our module and run it on the remote cluster. We construct it normally and then call
    # `get_or_to` to run it on the remote cluster. Using `get_or_to` allows us to load the exiting Module
    # by the name `llama3-8b-model` if it was already put on the cluster. If we want to update the module each
    # time we run this script, we can use `to` instead of `get_or_to`.
    #
    # Note that we also pass the `env` object to the `get_or_to` method, which will ensure that the environment is
    # set up on the remote machine before the module is run.
    remote_hf_chat_model = HFChatModel(
        torch_dtype=torch.bfloat16,
    ).get_or_to(gpu, env=env, name="llama3-8b-model")

    # ## Calling our remote function
    #
    # We can call the `predict` method on the model class instance if it were running locally.
    # This will run the function on the remote cluster and return the response to our local machine automatically.
    # Further calls will also run on the remote machine, and maintain state that was updated between calls, like
    # `self.pipeline`.
    while True:
        prompt = input(
            "\n\n... Enter a prompt to chat with the model, and 'exit' to exit ...\n"
        )
        if prompt.lower().strip() == "exit":
            break
        output = remote_hf_chat_model.predict(prompt)
        print("\n\n... Model Output ...\n")
        print(output)
