import runhouse as rh

# First, we define a class that will hold the model and allow us to send prompts to it.
# We'll later wrap this with `rh.module`. This is a Runhouse class that allows you to
# run code in your class on a remote machine.
#
# Learn more in the [Runhouse docs on functions and modules](/docs/tutorials/api-modules).
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

    cluster = rh.cluster(
        name="rh-g5",
        instance_type="g5.8xlarge",
        provider="aws",
    ).up_if_not()

    # Next, we define the environment for our module. This includes the required dependencies that need
    # to be installed on the remote machine, as well as any secrets that need to be synced up from local to remote.
    # Passing `huggingface` to the `secrets` parameter is optional and not needed here, but useful if we need to authenticate to download a model
    #
    # Learn more in the [Runhouse docs on envs](/docs/tutorials/api-envs).
    env = rh.env(
        name="flux_inference",
        reqs=[
            "diffusers",
            "torch",
            "transformers[sentencepiece]",
            "accelerate",
        ],
    )

    # Finally, we define our module and run it on the remote cluster. We construct it normally and then call
    # `to` to run it on the remote cluster. Alternatively, we could first check for an existing instance on the cluster
    # by calling `cluster.get(name="flux")`. This would return the remote model after an initial run.
    # If we want to update the module each time we run this script, we prefer to use `to`.
    #
    # Note that we also pass the `env` object to the `to` method, which will ensure that the environment is
    # set up on the remote machine before the module is run.
    RemoteFlux = rh.module(FluxPipeline).to(cluster, env=env, name="FluxPipeline")
    remote_flux = RemoteFlux(
        name="flux"
    )  # This has now been set up as a service on the remote cluster and can be used for inference.

    # ## Calling our remote function
    #
    # We can call the `generate` method on the model class instance if it were running locally.
    # This will run the function on the remote cluster and return the response to our local machine automatically.
    # Further calls will also run on the remote machine, and maintain state that was updated between calls
    prompt = "A woman runs through a large, grassy field towards a house."
    response = remote_flux.generate(prompt)
    response.save("flux-schnell.png")
    response.show()
