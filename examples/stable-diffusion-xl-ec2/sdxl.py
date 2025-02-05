# # Deploy Stable Diffusion XL 1.0 on AWS EC2

# This example demonstrates how to deploy a
# [Stable Diffusion XL model from Hugging Face](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
# on AWS EC2 using Runhouse.
#
# ## Setup credentials and dependencies
#
# Optionally, set up a virtual environment:
# ```shell
# $ conda create -n rh-sdxl python=3.9.15
# $ conda activate rh-sdxl
# ```
# Install the required dependencies:
# ```shell
# $ pip install "runhouse[aws]" Pillow
# ```
#
# We'll be launching an AWS EC2 instance via [SkyPilot](https://github.com/skypilot-org/skypilot), so we need to
# make sure our AWS credentials are set up:
# ```shell
# $ aws configure
# $ sky check
# ```
# We'll be downloading the Stable Diffusion model from Hugging Face, so we need to set up our Hugging Face token:
# ```shell
# $ export HF_TOKEN=<your huggingface token>
# ```
#
# ## Setting up a model class
#
# We import runhouse and other required libraries:

import base64
import os
from io import BytesIO

import runhouse as rh
from PIL import Image

# Next, we define a class that will hold the model and allow us to send prompts to it.
# We'll later wrap this with `rh.cls`. This is a Runhouse class that allows you to
# run code in your class on a remote machine.
#
# Learn more in the [Runhouse docs on functions and modules](/docs/tutorials/api-modules).
class StableDiffusionXLPipeline:
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
        import torch
        from diffusers import DiffusionPipeline

        # load local converted model into pipeline
        self.pipeline = DiffusionPipeline.from_pretrained(
            self.model_id, torch_dtype=torch.float16
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


# ## Setting up Runhouse primitives
#
# Now, we define the main function that will run locally when we run this script, and set up
# our Runhouse module on a remote cluster. First, we create a cluster with the desired instance type and provider.
# Our `instance_type` here is defined as `g5.8xlarge`, which is an AWS instance type. We can alternatively specify
# an accelerator type and count, such as `A10G:1`, and any instance type with those specifications will be used.
#
# Learn more in the [Runhouse docs on clusters](/docs/tutorials/api-clusters).
#
# :::note{.info title="Note"}
# Make sure that your code runs within a `if __name__ == "__main__":` block, as shown below. Otherwise,
# the script code will run when Runhouse attempts to run code remotely.
# :::
if __name__ == "__main__":
    # First, we define the image for our execution. This includes the required dependencies that need
    # to be installed on the remote machine, as well as any secrets that need to be synced up from local to remote.
    # Passing `huggingface` to the `sync_secrets` method will load the Hugging Face token we set up earlier.
    img = rh.Image(name="sdxl_inference").install_packages(
        [
            "diffusers==0.31.0",
            "torch",
            "transformers==4.31.0",
            "accelerate==0.21.0",
        ]
    )

    cluster = rh.compute(
        name="rh-g5",
        instance_type="g5.8xlarge",
        provider="aws",
        image=img,
    ).up_if_not()
    cluster.sync_secrets(["huggingface"])

    # Finally, we define our module and run it on the remote cluster. We construct it normally and then call
    # `to` to run it on the remote cluster. Alternatively, we could first check for an existing instance on the cluster
    # by calling `cluster.get(name="sdxl")`. This would return the remote model after an initial run.
    # If we want to update the module each time we run this script, we prefer to use `to`.
    RemoteStableDiffusion = rh.cls(StableDiffusionXLPipeline).to(
        cluster, name="StableDiffusionXLPipeline"
    )
    remote_sdxl = RemoteStableDiffusion(name="sdxl")

    # ## Calling our remote function
    #
    # We can call the `generate` method on the model class instance if it were running locally.
    # This will run the function on the remote cluster and return the response to our local machine automatically.
    # Further calls will also run on the remote machine, and maintain state that was updated between calls, like
    # `self.model`.
    prompt = "A woman runs through a large, grassy field towards a house."
    response = remote_sdxl.generate(
        prompt,
        num_inference_steps=25,
        negative_prompt="disfigured, ugly, deformed",
    )

    for gen_img in response:
        img = decode_base64_image(gen_img)
        img.show()
