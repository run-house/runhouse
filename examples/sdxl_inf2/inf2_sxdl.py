import base64
import os
from io import BytesIO

import runhouse as rh
from PIL import Image


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
            # and makes sure we exlcude the symlink files and "hidden" files, like .DS_Store, .gitignore, etc.
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

    def generate(self, **kwargs):
        # extract prompt from data
        prompt = kwargs.pop("inputs", kwargs)
        parameters = kwargs.pop("parameters", None)

        if not self.pipeline:
            self._load_pipeline()

        if parameters is not None:
            generated_images = self.pipeline(prompt, **parameters)["images"]
        else:
            generated_images = self.pipeline(prompt)["images"]

        # postprocess convert image into base64 string
        encoded_images = []
        for image in generated_images:
            buffered = BytesIO()
            image.save(buffered, format="JPEG")
            encoded_images.append(base64.b64encode(buffered.getvalue()).decode())

        # always return the first
        return {"generated_images": encoded_images}


# helper decoder
def decode_base64_image(image_string):
    base64_image = base64.b64decode(image_string)
    buffer = BytesIO(base64_image)
    return Image.open(buffer)


if __name__ == "__main__":
    # Source tutorial, originally for Sagemaker now on Runhouse:
    # https://www.philschmid.de/inferentia2-stable-diffusion-xl

    # setup RH cluster
    cluster = rh.cluster(
        name="rh-inf2",
        instance_type="inf2.8xlarge",
        provider="aws",
        # Hugging Face Neuron Deep Learning AMI
        # https://aws.amazon.com/marketplace/pp/prodview-gr3e6yiscria2
        image_id="ami-0f2c9159df4d244a2",
        region="us-east-1",
    ).up_if_not()

    # Set up dependencies

    # Need to install torch_neuronx first to avoid protobuf error
    # https://awsdocs-neuron.readthedocs-hosted.com/en/latest/frameworks/torch/torch-neuronx/training-troubleshooting.html#protobuf-error-typeerror-descriptors-cannot-not-be-created-directly
    cluster.run(
        [
            "python -m pip config set global.extra-index-url https://pip.repos.neuron.amazonaws.com",
            "pip install torch_neuronx",
        ]
    )

    env = rh.env(
        name="sdxl_inference",
        reqs=[
            "optimum-neuron==0.0.13",
            "diffusers==0.21.4",
        ],
        secrets=["huggingface"],  # Needed to download Llama2
        env_vars={"NEURON_RT_NUM_CORES": "2"},
    ).to(cluster)

    # Set up model on remote
    model = StableDiffusionXLPipeline().get_or_to(cluster, env=env, name="sdxl_neuron")

    # define prompt
    prompt = "A woman runs through a large, grassy field towards a house."

    # run prediction
    response = model.generate(
        inputs=prompt,
        parameters={
            "num_inference_steps": 25,
            "negative_prompt": "disfigured, ugly, deformed",
        },
    )

    img = decode_base64_image(response["generated_images"][0])
    img.show()
