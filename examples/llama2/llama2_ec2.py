# # Deploy Llama2 13B Chat Model Inference on AWS EC2

# This example demonstrates how to deploy a
# [LLama2 13B model from Hugging Face](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf)
# on AWS EC2 using Runhouse.
#
# First, you should install the required dependencies.
#
# Optionally, set up a virtual environment:
# ```shell
# conda create -n llama-demo-apps python=3.8
# conda activate llama-demo-apps
# ```
# Install the few required dependencies:
# ```shell
# pip install -r requirements.txt
# ```
# Then, we import runhouse and other required libraries:

import runhouse as rh
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

# Next, we define a class that will hold the model and allow us to send prompts to it.
# You'll notice this class inherits from `rh.Module`.
# This is a Runhouse class that allows you to
# run code in your class on a remote machine.


class HFChatModel(rh.Module):
    def __init__(self, model_id="meta-llama/Llama-2-13b-chat-hf", **model_kwargs):
        super().__init__()
        self.model_id, self.model_kwargs = model_id, model_kwargs
        self.tokenizer, self.model = None, None

    def load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, clean_up_tokenization_spaces=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id, **self.model_kwargs
        )

    def predict(self, prompt_text, **inf_kwargs):
        default_inf_kwargs = {
            "temperature": 0.7,
            "max_new_tokens": 500,
            "repetition_penalty": 1.0,
        }
        default_inf_kwargs.update(inf_kwargs)
        if not self.model:
            self.load_model()
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to("cuda")
        generated_ids = self.model.generate(
            **inputs, **default_inf_kwargs, streamer=TextStreamer(self.tokenizer)
        )
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]


# Now, we define the main function that will run locally when we run this script, and set up
# our Runhouse module on a remote cluster. First, we create a cluster with the desired instance type and provider.
# Our `instance_type` here is defined as `A10G:1`, which is the accelerator type and count that we need. We could
# alternatively specify a specific AWS instance type, such as `p3.2xlarge` or `g4dn.xlarge`.
#
# We will also need AWS credentials set up locally in order for Runhouse (using SkyPilot under the hood) to launch
# the EC2 instance. Run:
# ```shell
# aws configure
# ```
# and enter keys for your role in your AWS account. You can run:
# ```shell
# sky check
# ```
# afterwards to verify that your SkyPilot can launch instances in your AWS account.
if __name__ == "__main__":
    gpu = rh.cluster(name="rh-a10x", instance_type="A10G:1", provider="aws")

    # Next, we define the environment for our module. This includes the required dependencies that need
    # to be installed on the remote machine, as well as any secrets that need to be synced up from local to remote.
    # In this case, we need to make sure our Hugging Face token is set up. Run:
    # ```shell
    # export HF_TOKEN=<your-huggingface-token>
    # ```
    # on your local machine. Passing `huggingface` to the required secrets will automatically load from here.
    env = rh.env(
        reqs=[
            "torch",
            "transformers==4.31.0",
            "accelerate==0.21.0",
            "bitsandbytes==0.40.2",
            "safetensors>=0.3.1",
            "scipy",
        ],
        secrets=["huggingface"],  # Needed to download Llama2
        name="llama2inference",
        working_dir="./",
    )

    # Finally, we define our module and run it on the remote cluster. We construct it normally and then call
    # `get_or_to` to run it on the remote cluster. This will return the module if it already exists on the
    # remote cluster, or create it.
    #
    # If you are iterating and changing the code, you can just use `.to`,
    # which will update the module on the remote cluster each time you run the script. Using `get_or_to` allows us
    # to load the exiting Module by the name `llama-13b-model` if it was already put on the cluster.
    # This allows us to quickly run further inference queries after the first time the model is pinned to memory.
    # Note that we also pass the `env` object to the `get_or_to` method, which will ensure that the environment is
    # set up on the remote machine before the module is run.
    remote_hf_chat_model = HFChatModel(
        model_id="meta-llama/Llama-2-13b-chat-hf",
        load_in_4bit=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    ).get_or_to(gpu, env=env, name="llama-13b-model")

    # We can call the `predict` method on the model class instance if it were running locally.
    # This will run the function on the remote cluster and return the response to our local machine automatically.
    # Further calls will also run on the remote machine, and maintain state that was updated between calls, like
    # `self.model` and `self.tokenizer`.
    while True:
        prompt = input(
            "\n\n... Enter a prompt to chat with the model, and 'exit' to exit ...\n"
        )
        if prompt.lower() == "exit":
            break
        output = remote_hf_chat_model.predict(prompt)
        print("\n\n... Model Output ...\n")
        print(output)
