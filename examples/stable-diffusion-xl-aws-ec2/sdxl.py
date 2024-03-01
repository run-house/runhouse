import base64
import os
from io import BytesIO

import runhouse as rh
from PIL import Image


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


# helper decoder
def decode_base64_image(image_string):
    base64_image = base64.b64decode(image_string)
    buffer = BytesIO(base64_image)
    return Image.open(buffer)


if __name__ == "__main__":

    # setup RH cluster
    cluster = rh.cluster(
        name="rh-g5",
        instance_type="g5.8xlarge",
        provider="aws",
    ).up_if_not()

    # Set up dependencies
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

    # Set up model on remote
    model = StableDiffusionXLPipeline().get_or_to(cluster, env=env, name="sdxl")

    # define prompt
    prompt = "A woman runs through a large, grassy field towards a house."

    # run prediction
    response = model.generate(
        prompt,
        num_inference_steps=25,
        negative_prompt="disfigured, ugly, deformed",
    )

    for gen_img in response:
        img = decode_base64_image(gen_img)
        img.show()
