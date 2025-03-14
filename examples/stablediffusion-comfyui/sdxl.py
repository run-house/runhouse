import base64
import os
from io import BytesIO

import runhouse as rh
from PIL import Image


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


if __name__ == "__main__":
    img = rh.Image(name="sdxl_inference").pip_install(
        [
            "diffusers",
            "torch",
            "transformers",
            "accelerate",
        ]
    )

    cluster = rh.compute(
        name="rh-g5",
        instance_type="g5.8xlarge",
        provider="aws",
        image=img,
    ).up_if_not()
    cluster.sync_secrets(["huggingface"])

    RemoteStableDiffusion = rh.cls(StableDiffusionXLPipeline).to(
        cluster, name="StableDiffusionXLPipeline"
    )
    remote_sdxl = RemoteStableDiffusion(name="sdxl")

    prompt = "A woman runs through a large, grassy field towards a house."
    response = remote_sdxl.generate(
        prompt,
        num_inference_steps=25,
        negative_prompt="ugly",
    )

    for gen_img in response:
        img = decode_base64_image(gen_img)
        img.show()
