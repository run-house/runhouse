import kubetorch as kt

# ## Create Flux Pipeline with Kubetorch
# First, we define a class that will hold the model and allow us to send prompts to it.
# To deploy it as a service, we simply decorate the class to send it to our cluster
# when we call `kubetorch deploy` in the CLI.
img = (
    kt.images.pytorch()
    .pip_install(
        [
            "diffusers",
            "transformers[sentencepiece]",
            "accelerate",
        ]
    )
    .sync_secrets(["huggingface"])
)


@kt.compute(
    gpus="A10G:1", memory="64", image=img
)  # Send to compute with an A10 GPU and 64GB of memory
@kt.distribute("auto", num_replicas=(1, 4))  # Autoscale between 1 and 4 replicas
class FluxPipeline:
    def __init__(
        self,
        model_id: str = "black-forest-labs/FLUX.1-schnell",  # Schenll is smaller and faster while dev is more powerful but slower
    ):
        super().__init__()
        self.model_id = model_id
        self.pipeline = None

    def _load_pipeline(self):
        import torch
        from diffusers import FluxPipeline

        if not self.pipeline:
            self.pipeline = FluxPipeline.from_pretrained(
                self.model_id, torch_dtype=torch.bfloat16, use_safetensors=True
            )
            self.pipeline.enable_sequential_cpu_offload()  # Optimizes memory usage to allow the model to fit and inference on an A10 which has 24GB of memory

    def generate(self, input_prompt: str, **parameters):
        import torch

        torch.cuda.empty_cache()

        if not self.pipeline:
            self._load_pipeline()

        image = self.pipeline(
            input_prompt,
            guidance_scale=0.0,
            num_inference_steps=4,
            max_sequence_length=256,
            generator=torch.Generator("cpu").manual_seed(0),
        ).images[0]

        return image


if __name__ == "__main__":
    # We can load the remote model from anywhere that has access to the cluster
    flux_pipeline = FluxPipeline.from_name("flux")

    # We can call the `generate` method on the model class instance if it were running locally.
    # This will run the function on the remote cluster and return the response to our local machine automatically.
    # We can also call this from a different machine or script and create composite ML systems.
    prompt = "A woman runs through a large, grassy field towards a house."
    response = flux_pipeline.generate(prompt)
    response.save("flux-schnell.png")
    response.show()
