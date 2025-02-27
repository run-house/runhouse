# # Run Llama 3 8B Model Inference with vLLM on AWS

# This example demonstrates how to run a Llama 3 8B model from Hugging Face
# with vLLM on AWS using Runhouse.
#
# Make sure to sign the waiver on the [Hugging Face model page](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)
# so that you can access it.
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
# $ pip install "runhouse[aws]" asyncio
# ```
#
# This also requires an AWS credential to be available in your local or CI enviornment.
#
# We'll be downloading the Llama 3 model from Hugging Face, so we need to set up our Hugging Face token:
# ```shell
# $ export HF_TOKEN=<your huggingface token>
# ```
#
# ## Define a Llama 3 model class
# We import `runhouse` and `asyncio` because that's all that's needed to run the script locally.
# The actual vLLM imports are defined in the environment on the cluster in which the function itself is served.

import asyncio

import runhouse as rh

# Next, we define a class that will hold the model and allow us to send prompts to it.
# We'll later wrap this with `rh.cls`.
# This is a Runhouse class that allows you to run code in your class on a remote machine.
class LlamaModel:
    def __init__(self, model_id="meta-llama/Meta-Llama-3-8B-Instruct", **model_kwargs):
        super().__init__()
        self.model_id, self.model_kwargs = model_id, model_kwargs
        self.engine = None

    def load_engine(self):
        import gc

        import torch
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.engine.async_llm_engine import AsyncLLMEngine
        from vllm.model_executor.parallel_utils.parallel_state import (
            destroy_model_parallel,
        )

        # Cleanup methods to free memory for cases where you reload the model
        destroy_model_parallel()
        gc.collect()
        torch.cuda.empty_cache()

        args = AsyncEngineArgs(
            model=self.model_id,  # Hugging Face Model ID
            tensor_parallel_size=1,  # Increase if using additional GPUs
            trust_remote_code=True,  # Trust remote code from Hugging Face
            enforce_eager=True,  # Set to False for production use cases
        )
        self.engine = AsyncLLMEngine.from_engine_args(args)

    async def generate(self, prompt: str, **sampling_params):
        from vllm.sampling_params import SamplingParams
        from vllm.utils import random_uuid

        if not self.engine:
            self.load_engine()

        sampling_params = SamplingParams(**sampling_params)
        request_id = random_uuid()
        results_generator = self.engine.generate(prompt, sampling_params, request_id)

        async for output in results_generator:
            final_output = output
        responses = []
        for output in final_output.outputs:
            responses.append(output.text)
        return responses


# ## Set up Runhouse primitives
#
# Now, we define the main function that will run locally when we run this script and set up
# our Runhouse module on a remote cluster. First, we create a cluster with the desired instance type and provider.
# Our `gpus` here is defined as `L4:1`, which is the GPU type and count that we need.


async def main():
    # First, we define the image for our module. This includes the required dependencies that need
    # to be installed on the remote machine, as well as any secrets that need to be synced up from local to remote.
    # Passing `"huggingface"` to the `sync_secrets` method will load the Hugging Face token we set up earlier.
    img = (
        rh.Image(name="llama3inference")
        .pip_install(
            [
                "torch",
                "vllm==0.2.7",
            ]  # pydantic version error to be aware of based on version
        )
        .sync_secrets(["huggingface"])
    )

    gpu_cluster = rh.compute(
        name="rh-l4x",
        gpus="L4:1",
        memory="32+",
        provider="aws",
        image=img,
        autostop_mins=30,  # Number of minutes to keep the cluster up after inactivity, relevant for clouds
    ).up_if_not()

    # Finally, we define our module and run it on the remote cluster. We construct it normally and then call
    # `to` to run it on the remote cluster. Alternatively, we could first check for an existing instance on the cluster
    # by calling `cluster.get(name="llama3-8b-model")`. This would return the remote model after an initial run.
    # If we want to update the module each time we run this script, we prefer to use `to`.
    RemoteLlamaModel = rh.cls(LlamaModel).get_or_to(gpu_cluster, name="Llama3Model")
    remote_llama_model = RemoteLlamaModel(name="llama3-8b-model")

    # ## Calling our remote function
    #
    # We can call the `generate` method on the model class instance as if it were running locally.
    # This will run the function on the remote cluster and return the response to our local machine automatically.
    # Further calls will also run on the remote machine, and maintain state that was updated between calls, like
    # `self.engine`.
    prompt = "The best chocolate chip cookie is"

    ans = await remote_llama_model.generate(
        prompt=prompt, temperature=0, top_p=0.95, max_tokens=100
    )
    for text_output in ans:
        print(f"... Generated Text:\n{prompt}{text_output}\n")


# ## Run the script
# Make sure that your code runs within a `if __name__ == "__main__":` block.
# Otherwise, the script code will run when Runhouse attempts to run code remotely.
if __name__ == "__main__":

    asyncio.run(main())
