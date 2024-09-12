# # Run Llama 3 8B Model Inference with vLLM on GCP

# This example demonstrates how to run a Llama 3 8B model from Hugging Face
# with vLLM on GCP using Runhouse.
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
# $ pip install "runhouse[gcp]" asyncio
# ```
#
# We'll be launching a GCP instance via [SkyPilot](https://github.com/skypilot-org/skypilot), so we need to
# make sure your credentials are set up. You may be prompted to pick a cloud project to use after running `gcloud init`.
# If you don't have one ready yet, you can connect one later by listing your projects
# with `gcloud projects list` and setting one with `gcloud config set project <PROJECT_ID>`.
# ```shell
# $ gcloud init
# $ gcloud auth application-default login
# $ sky check
# ```
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
# We'll later wrap this with `rh.module`.
# This is a Runhouse class that allows you to run code in your class on a remote machine.
#
# Learn more in the [Runhouse docs on functions and modules](/docs/tutorials/api-modules).
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
# Our `instance_type` here is defined as `L4:1`, which is the accelerator type and count that we need. We could
# alternatively specify a specific [GCP instance](https://cloud.google.com/compute/docs/gpus) type, such as `g2-standard-8`.
#
# Learn more in the [Runhouse docs on clusters](/docs/tutorials/api-clusters).
#
# :::note{.info title="Note"}
# The Python code we'll run is contained in an asynchronous function, `main`. To make this guide more readable, it's
# contents are rendered as top-level code snippets, but they should be included in the `main` method for running.
# :::
async def main():
    gpu_cluster = rh.cluster(
        name="rh-l4x",
        instance_type="L4:1",
        memory="32+",
        provider="gcp",
        autostop_mins=30,  # Number of minutes to keep the cluster up after inactivity
        # (Optional) Include the following to create exposed TLS endpoints:
        # open_ports=[443], # Expose HTTPS port to public
        # server_connection_type="tls", # Specify how runhouse communicates with this cluster
        # den_auth=False, # No authentication required to hit this cluster (NOT recommended)
    ).up_if_not()

    # We'll set an `autostop_mins` of 30 for this example. If you'd like your cluster to run indefinitely, set `autostop_mins=-1`.
    # You can use SkyPilot in the terminal to manage your active clusters with `sky status` and `sky down <cluster_id>`.
    #
    # Next, we define the environment for our module. This includes the required dependencies that need
    # to be installed on the remote machine, as well as any secrets that need to be synced up from local to remote.
    # Passing `huggingface` to the `secrets` parameter will load the Hugging Face token we set up earlier.
    #
    # Learn more in the [Runhouse docs on envs](/docs/tutorials/api-envs).
    env = rh.env(
        reqs=["torch", "vllm==0.2.7"],  # >=0.3.0 causes pydantic version error
        secrets=["huggingface"],  # Needed to download Llama 3 from HuggingFace
        name="llama3inference",
    )

    # Finally, we define our module and run it on the remote cluster. We construct it normally and then call
    # `to` to run it on the remote cluster. Alternatively, we could first check for an existing instance on the cluster
    # by calling `cluster.get(name="llama3-8b-model")`. This would return the remote model after an initial run.
    # If we want to update the module each time we run this script, we prefer to use `to`.
    #
    # Note that we also pass the `env` object to the `to` method, which will ensure that the environment is
    # set up on the remote machine before the module is run.
    RemoteLlamaModel = rh.module(LlamaModel).to(
        gpu_cluster, env=env, name="Llama3Model"
    )
    remote_llama_model = RemoteLlamaModel(name="llama3-8b-model")

    # ## Calling our remote function
    #
    # We can call the `generate` method on the model class instance as if it were running locally.
    # This will run the function on the remote cluster and return the response to our local machine automatically.
    # Further calls will also run on the remote machine, and maintain state that was updated between calls, like
    # `self.engine`.
    prompt = "The best chocolate chip cookie is"
    ans = await remote_llama_model.generate(
        prompt=prompt, temperature=0.8, top_p=0.95, max_tokens=100
    )
    for text_output in ans:
        print(f"... Generated Text:\n{prompt}{text_output}\n")

    # :::note{.info title="Note"}
    # Your initial run of this script may take a few minutes to deploy an instance on GCP, set up the environment,
    # and load the Llama 3 model. Subsequent runs will reuse the cluster and generally take seconds.
    # :::

    # ### Advanced: Sharing and TLS endpoints
    # Runhouse makes it easy to share your module or create a public endpoint you can curl or use in your apps.
    # Use the optional settings in your cluster definition above to expose an endpoint. You can additionally
    # enable [Runhouse Den](/dashboard) auth to require an auth token and provide access to your teammates.
    #
    # Fist, create or log in to your Runhouse account.
    # ```shell
    # $ runhouse login
    # ```
    #
    # Once you've logged in to an account, use the following lines to enable Den Auth on the cluster, save
    # your resources to the Den UI, and grant access to your collaborators.
    # ```python
    # gpu_cluster.enable_den_auth()  # Enable Den Auth
    # gpu_cluster.save()
    # remote_llama_model.save()  # Save the module to Den for easy reloading
    # remote_llama_model.share(users=["friend@yourcompany.com"], access_level="read")
    # ```
    #
    # Learn more: [Sharing](/docs/tutorials/quick-start-den#sharing)
    #
    # ### OpenAI Compatible Server
    # By default, vLLM implements OpenAI's Completions and Chat API.
    # This means that you can call your self-hosted Llama 3 model on GCP with OpenAI's Python library.
    # Read more about this and implementing chat templates in [vLLM's documentation](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html).


# ## Run the script
# Finally, we'll run the script to deploy the model and run inference.
# :::note{.info title="Note"}
# Make sure that your code runs within a `if __name__ == "__main__":` block.
# Otherwise, the script code will run when Runhouse attempts to run code remotely.
# :::
if __name__ == "__main__":
    asyncio.run(main())

# Please reference the Github link at the top of this page (if viewing via run.house/examples)
# for the full Python file you can compare to or run yourself.
