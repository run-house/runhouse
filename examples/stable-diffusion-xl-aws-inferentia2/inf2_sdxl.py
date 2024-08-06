# # Deploy Stable Diffusion XL 1.0 on AWS Inferentia

# This example demonstrates how to deploy a
# [Stable Diffusion XL model from Hugging Face](https://huggingface.co/aws-neuron/stable-diffusion-xl-base-1-0-1024x1024)
# on AWS Inferentia2 using
# Runhouse. [AWS Inferentia2 instances](https://aws.amazon.com/ec2/instance-types/inf2/)
# are powered by AWS Neuron, a custom hardware accelerator for machine learning
# inference workloads. This example uses a model that was pre-compiled for AWS Neuron, and is available on the
# Hugging Face Hub.
#
# ## Setup credentials and dependencies
#
# Optionally, set up a virtual environment:
# ```shell
# $ conda create -n rh-inf2 python=3.9.15
# $ conda activate rh-inf2
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
# You'll notice this class inherits from `rh.Module`.
# This is a Runhouse class that allows you to
# run code in your class on a remote machine.
#
# Learn more in the [Runhouse docs on functions and modules](/docs/tutorials/api-modules).
class StableDiffusionXLPipeline(rh.Module):
    def __init__(
        self,
        model_id: str = "aws-neuron/stable-diffusion-xl-base-1-0-1024x1024",
        model_dir: str = "sdxl_neuron",
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
        from huggingface_hub import snapshot_download
        from optimum.neuron import NeuronStableDiffusionXLPipeline

        if not self._model_loaded_on_disk():
            # save compiled model to local directory
            # Downloads our compiled model from the HuggingFace Hub
            # using the revision as neuron version reference
            # and makes sure we exclude the symlink files and "hidden" files, like .DS_Store, .gitignore, etc.
            snapshot_download(
                self.model_id,
                revision="2.15.0",
                local_dir=self.model_dir,
                local_dir_use_symlinks=False,
                allow_patterns=["[!.]*.*"],
            )

        # load local converted model into pipeline
        self.pipeline = NeuronStableDiffusionXLPipeline.from_pretrained(
            self.model_dir, device_ids=[0, 1]
        )

    def generate(self, input_prompt: str, output_format: str = "JPEG", **parameters):
        if not self.pipeline:
            self._load_pipeline()

        generated_images = self.pipeline(input_prompt, **parameters)["images"]

        if output_format == "PIL":
            return generated_images

        # postprocess convert image into base64 string
        encoded_images = []
        for image in generated_images:
            buffered = BytesIO()
            image.save(buffered, format=output_format)
            encoded_images.append(base64.b64encode(buffered.getvalue()).decode())

        return encoded_images


def decode_base64_image(image_string):
    base64_image = base64.b64decode(image_string)
    buffer = BytesIO(base64_image)
    return Image.open(buffer)


# ## Setting up Runhouse primitives
#
# Now, we define the main function that will run locally when we run this script, and set up
# our Runhouse module on a remote cluster. First, we create a cluster with the desired instance type and provider.
# Our `instance_type` here is defined as `inf2.8xlarge`, which is one of the special
# [AWS Inferentia2 instance types](https://aws.amazon.com/ec2/instance-types/inf2/).
# We can alternatively specify an accelerator type and count, such as `A10G:1`,
# and any instance type with those specifications will be used.
#
# We use a specific `image_id`, which in this case is the
# [Deep Learning AMI Base Neuron](https://aws.amazon.com/releasenotes/aws-deep-learning-ami-base-neuron-ubuntu-20-04/)
# which comes with the AWS Neuron drivers preinstalled. The image_id is region-specific. To change the region,
# use the AWS CLI command on the page above under "Query AMI-ID with AWSCLI."
#
# The cluster we set up here also uses `tls` for the `server_connection_type`, which means that all communication
# will be over HTTPS and encrypted. We need to tell SkyPilot to open port 443 for this to work.
#
# We also set `den_auth` to `True`, which means that we will use [Runhouse Den](/dashboard) to
# authenticate public requests to this cluster. This means that we can open this cluster to the public internet, and
# only people who have ran `runhouse login` and set up Runhouse accounts will be able to access it.
#
# Learn more in the [Runhouse docs on clusters](/docs/tutorials/api-clusters).
#
# :::note{.info title="Note"}
# Make sure that your code runs within a `if __name__ == "__main__":` block, as shown below. Otherwise,
# the script code will run when Runhouse attempts to run code remotely.
# :::
if __name__ == "__main__":
    cluster = rh.cluster(
        name="rh-inf2",
        instance_type="inf2.8xlarge",
        provider="aws",
        image_id="ami-0e0f965ee5cfbf89b",
        region="us-east-1",
        server_connection_type="tls",
        open_ports=[443],
        den_auth=True,
    ).up_if_not()

    # Set up dependencies

    # We can run commands directly on the cluster via `cluster.run()`. Here, we set up the environment for our
    # upcoming environment that installed some AWS-neuron specific libraries. The `torch_neuronx` library needs to be
    # installed before the rest of the env is set up in order to avoid a
    # [common error](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/frameworks/torch/torch-neuronx/
    # training-troubleshooting.html#protobuf-error-typeerror-descriptors-cannot-not-be-created-directly),
    # so we run this first.
    cluster.run(
        [
            "python -m pip config set global.extra-index-url https://pip.repos.neuron.amazonaws.com",
            "python -m pip install neuronx-cc==2.* torch-neuronx==1.13.1.1.13.1",
        ],
    )

    # Next, we define the environment for our module. This includes the required dependencies that need
    # to be installed on the remote machine, as well as any secrets that need to be synced up from local to remote.
    # Passing `huggingface` to the `secrets` parameter will load the Hugging Face token we set up earlier.
    # We also can set environment variables, such as `NEURON_RT_NUM_CORES` which is required for AWS Neuron.
    #
    # Learn more in the [Runhouse docs on envs](/docs/tutorials/api-envs).
    env = rh.env(
        name="sdxl_inference",
        reqs=[
            "optimum-neuron==0.0.20",
            "diffusers==0.27.2",
        ],
        secrets=["huggingface"],  # Needed to download model
        env_vars={"NEURON_RT_NUM_CORES": "2"},
    )

    # Finally, we define our module and run it on the remote cluster. We construct it normally and then call
    # `get_or_to` to run it on the remote cluster. Using `get_or_to` allows us to load the exiting Module
    # by the name `sdxl_neuron` if it was already put on the cluster. If we want to update the module each
    # time we run this script, we can use `to` instead of `get_or_to`.
    #
    # Note that we also pass the `env` object to the `get_or_to` method, which will ensure that the environment is
    # set up on the remote machine before the module is run.
    model = StableDiffusionXLPipeline().get_or_to(cluster, env=env, name="sdxl_neuron")

    # ## Calling our remote function
    #
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

    img = decode_base64_image(response[0])
    img.show()
