# # Deploy Stable Diffusion XL 1.0 on AWS EC2

# This example demonstrates how to deploy a
# [Stable Diffusion XL model from Hugging Face](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
# on AWS EC2 using Runhouse.
# First, you should install the required dependencies.
#
# Optionally, set up a virtual environment:
# ```shell
# conda create -n rh-sdxl python=3.9.15
# conda activate rh-sdxl
# ```
# Install the few required dependencies:
# ```shell
# pip install -r requirements.txt
# ```
# Then, we import runhouse and other required libraries:

import base64
import os
from io import BytesIO

import runhouse as rh
from PIL import Image

# Next, we define a class that will hold the model and allow us to send prompts to it.
# You'll notice this class inherits from `rh.Module`.
# This is a Runhouse class that allows you to
# run code in your class on a remote machine.
class StableDiffusionXLPipeline(rh.Module):
    def __init__(
        self,
        model_id: str = "stabilityai/stable-diffusion-xl-base-1.0",
        model_dir: str = "sdxl",
    ):
        super().__init__()
        self.model_dir = model_dir
        self.model_id = model_id
        self.pipeline = None

    def _model_loaded_on_disk(self):
        return (
            self.model_dir
            and os.path.isdir(self.model_dir)
            and len(os.listdir(self.model_dir)) > 0
        )

    def _load_pipeline(self):
        from diffusers import DiffusionPipeline
        from huggingface_hub import snapshot_download

        if not self._model_loaded_on_disk():
            # save compiled model to local directory
            # Downloads our compiled model from the HuggingFace Hub
            # and makes sure we exlcude the symlink files and "hidden" files, like .DS_Store, .gitignore, etc.
            snapshot_download(
                self.model_id,
                local_dir=self.model_dir,
                local_dir_use_symlinks=False,
                allow_patterns=["[!.]*.*"],
            )

        # load local converted model into pipeline
        self.pipeline = DiffusionPipeline.from_pretrained(
            self.model_dir, device_ids=[0, 1]
        )
        self.pipeline.to("cuda")

    def generate(self, input_prompt: str, output_format: str = "JPEG", **parameters):
        # extract prompt from data
        if not self.pipeline:
            self._load_pipeline()

        generated_images = self.pipeline(input_prompt, **parameters)["images"]

        # postprocess convert image into base64 string
        encoded_images = []
        for image in generated_images:
            buffered = BytesIO()
            image.save(buffered, format=output_format)
            encoded_images.append(base64.b64encode(buffered.getvalue()).decode())

        # always return the first
        return encoded_images


def decode_base64_image(image_string):
    base64_image = base64.b64decode(image_string)
    buffer = BytesIO(base64_image)
    return Image.open(buffer)


# Now, we define the main function that will run locally when we run this script, and set up
# our Runhouse module on a remote cluster. First, we create a cluster with the desired instance type and provider.
# Our `instance_type` here is defined as `g5.8xlarge`, which is an AWS instance type. We can alternatively specify
# an accelerator type and count, such as `A10G:1`, and any instance type with those specifications will be used.
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

    cluster = rh.cluster(
        name="rh-g5",
        instance_type="g5.8xlarge",
        provider="aws",
    ).up_if_not()

    # Next, we define the environment for our module. This includes the required dependencies that need
    # to be installed on the remote machine, as well as any secrets that need to be synced up from local to remote.
    # In this case, we need to make sure our Hugging Face token is set up. Run:
    # ```shell
    # export HF_TOKEN=<your-huggingface-token>
    # ```
    # on your local machine. Passing `huggingface` to the required secrets will automatically load from here.
    env = rh.env(
        name="sdxl_inference",
        reqs=[
            "diffusers==0.21.4",
            "huggingface_hub",
            "torch",
            "transformers==4.31.0",
            "accelerate==0.21.0",
        ],
        secrets=["huggingface"],  # Needed to download Llama2
        env_vars={"NEURON_RT_NUM_CORES": "2"},
    ).to(cluster)

    # Finally, we define our module and run it on the remote cluster. We construct it normally and then call
    # `get_or_to` to run it on the remote cluster. This will return the module if it already exists on the
    # remote cluster, or create it.
    #
    # If you are iterating and changing the code, you can just use `.to`,
    # which will update the module on the remote cluster each time you run the script. Using `get_or_to` allows us
    # to load the exiting Module by the name `sdxl` if it was already put on the cluster.
    # This allows us to quickly run further inference queries after the first time the model is pinned to memory.
    # Note that we also pass the `env` object to the `get_or_to` method, which will ensure that the environment is
    # set up on the remote machine before the module is run.
    model = StableDiffusionXLPipeline().get_or_to(cluster, env=env, name="sdxl")

    # We can call the `generate` method on the model class instance if it were running locally.
    # This will run the function on the remote cluster and return the response to our local machine automatically.
    # Further calls will also run on the remote machine, and maintain state that was updated between calls, like
    # `self.model`.
    prompt = "A woman runs through a large, grassy field towards a house."
    response = model.generate(
        prompt,
        num_inference_steps=25,
        negative_prompt="disfigured, ugly, deformed",
    )

    for gen_img in response:
        img = decode_base64_image(gen_img)
        img.show()
