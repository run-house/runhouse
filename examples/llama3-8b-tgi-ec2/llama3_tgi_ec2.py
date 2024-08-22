# # Deploy Llama 3 8B with TGI on AWS EC2

# This example demonstrates how to deploy a Meta Llama 3 8B model from Hugging Face
# with [TGI](https://huggingface.co/docs/text-generation-inference/messages_api) on AWS EC2 using Runhouse.
#
# Make sure to sign the waiver on the [Hugging Face model page](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)
# so that you can access it.
#
# ## Setup credentials and dependencies
# Install the required dependencies:
# ```shell
# $ pip install "runhouse[aws]"
# ```
#
# We'll be launching an AWS EC2 instance via [SkyPilot](https://github.com/skypilot-org/skypilot), so we need to make
# sure our AWS credentials are set up:
# ```shell
# $ aws configure
# $ sky check
# ```
#
# ## Setting up a model class
# We import built-in packages and runhouse, the only required library to have installed locally:

import time
from pathlib import Path

import requests

import runhouse as rh

# Next, we define a class that will hold the model and allow us to send prompts to it.
# We'll later wrap this with `rh.module`. This is a Runhouse class that allows you to
# run code in your class on a remote machine.
#
# Learn more in the [Runhouse docs on functions and modules](/docs/tutorials/api-modules).
class TGIInference:
    def __init__(
        self,
        model_id="meta-llama/Meta-Llama-3-8B-Instruct",
        image_uri="ghcr.io/huggingface/text-generation-inference:latest",
        max_input_length=2048,
        max_total_tokens=4096,
        container_port=8080,
        **model_kwargs,
    ):
        super().__init__()
        self.docker_client = None

        self.model_id = model_id
        self.image_uri = image_uri
        self.max_input_length = max_input_length
        self.max_total_tokens = max_total_tokens
        self.model_kwargs = model_kwargs

        self.container_port = container_port
        self.container_name = "text-generation-service"

    def _load_docker_client(self):
        import docker

        self.docker_client = docker.from_env()

    def _model_is_deployed(self):
        if self.docker_client is None:
            self._load_docker_client()

        containers = self.docker_client.containers.list(
            filters={"name": self.container_name}
        )

        return bool(containers)

    def deploy(self):
        # Adapted from: https://huggingface.co/docs/text-generation-inference/quicktour
        import docker

        if self._model_is_deployed():
            return

        print("Model has not yet been deployed, loading image and running container.")

        home_dir = str(Path.home())
        data_volume_path = f"{home_dir}/data"

        device_request = docker.types.DeviceRequest(
            count=-1,
            capabilities=[["gpu"]],
        )

        start_time = time.time()
        timeout = 600

        # Load the HF token which was synced onto the cluster as part of the env setup
        hf_secret = rh.secret(provider="huggingface")
        hf_token = hf_secret.values.get("token")

        model_cmd = (
            f"--model-id {self.model_id} "
            f"--max-input-length {self.max_input_length} "
            f"--max-total-tokens {self.max_total_tokens}"
        )

        # Add any other model kwargs to the command
        # https://github.com/huggingface/optimum-neuron/tree/main/text-generation-inference#choosing-service-parameters
        for key, value in self.model_kwargs.items():
            model_cmd += f" --{key} {value}"

        container = self.docker_client.containers.run(
            self.image_uri,
            name=self.container_name,
            detach=True,
            ports={"80/tcp": self.container_port},
            volumes={data_volume_path: {"bind": "/data", "mode": "rw"}},
            command=model_cmd,
            environment={"HF_TOKEN": hf_token},
            device_requests=[device_request],
            shm_size="1g",
        )

        print("Container started, waiting for model to load.")

        # Wait for model to load inside the container
        for line in container.logs(stream=True):
            current_time = time.time()
            elapsed_time = current_time - start_time

            log_line = line.strip().decode("utf-8")
            if "Connected" in log_line:
                print("Finished loading model, endpoint is ready.")
                break

            if elapsed_time > timeout:
                print(f"Failed to load model within {timeout} seconds. Exiting.")
                break


# ## Setting up Runhouse primitives
#
# Now, we define the main function that will run locally when we run this script, and set up
# our Runhouse module on a remote cluster. First, we create a cluster with the desired instance type and provider.
# Our `instance_type` here is defined as `A10G:1`, which will launch an appropriate
# an [G5 instance on EC2](https://aws.amazon.com/ec2/instance-types/g5/) with the NVIDIA A10G GPU.
#
# We also open port 8080, which is the port that the TGI model will be running on.
#
# Learn more about clusters in the [Runhouse docs](/docs/tutorials/api-clusters).
#
# :::note{.info title="Note"}
# Make sure that all the following code runs within a `if __name__ == "__main__":` block, as shown below. Otherwise,
# the script code will run when Runhouse attempts to run code remotely. We'll break up the block in this example to
# improve readability.
# :::
if __name__ == "__main__":
    port = 8080  # Set to 8080 for http
    cluster = rh.cluster(
        name="rh-a10",
        instance_type="A10G:1",
        memory="32+",
        provider="aws",
        autostop_mins=30,  # Number of minutes to keep the cluster up after inactivity, -1 for indefinite
        open_ports=[port],  # Expose HTTP port to public
    ).up_if_not()

    # Next, we define the environment for our module. This includes the required dependencies that need
    # to be installed on the remote machine, as well as any secrets that need to be synced up from local to remote.
    #
    # Passing `huggingface` to the `secrets` parameter will load the Hugging Face token we set up earlier. This is
    # needed to download the model from the Hugging Face model hub. Runhouse will handle saving the token down
    # on the cluster in the default Hugging Face token location (`~/.cache/huggingface/token`).
    #
    # Learn more in the [Runhouse docs on envs](/docs/tutorials/api-envs).
    env = rh.env(
        name="tgi_env",
        reqs=["docker", "torch", "transformers"],
        secrets=["huggingface"],
    )

    # Finally, we define our module and run it on the remote cluster. We construct it normally and then call
    # `to` to run it on the remote cluster. Alternatively, we could first check for an existing instance on the cluster
    # by calling `cluster.get(name="tgi-inference")`. This would return the remote model after an initial run.
    # If we want to update the module each time we run this script, we prefer to use `to`.
    #
    # Note that we also pass the `env` object to the `to` method, which will ensure that the environment is
    # set up on the remote machine before the module is run.
    RemoteTGIInference = rh.module(TGIInference).to(
        cluster, env=env, name="TGIInference"
    )

    remote_tgi_model = RemoteTGIInference(container_port=port, name="tgi-inference")

    # Note: For more info on access controls, see the
    # [Runhouse docs on sharing](https://www.run.house/docs/tutorials/quick-start-den#sharing).

    # ## Deploying the model
    # We can call the `deploy` method on the model class instance if it were running locally.
    # This will load and run the model on the remote cluster.
    # We only need to do this setup step once, as further calls will use the existing docker container deployed
    # on the cluster and maintain state between calls:
    remote_tgi_model.deploy()

    # ## Sending a prompt to the model
    prompt_message = "The brightest known star is"

    # We'll use the Messages API to send the prompt to the model.
    # See [here](https://huggingface.co/docs/text-generation-inference/messages_api#streaming) for more info
    # on the Messages API

    # Call the model with the prompt messages:
    # Note: We can also update some of the [default parameters](https://huggingface.github.io/text-generation-inference/#/Text%20Generation%20Inference/generate)
    data = {
        "inputs": prompt_message,
        "parameters": {
            "max_new_tokens": 50,
            "temperature": 0.9,
            "top_p": 0.92,
            "do_sample": True,
            "repetition_penalty": 1.2,
            "frequency_penalty": 1.0,
        },
    }
    headers = {
        "Content-Type": "application/json",
    }

    response = requests.post(
        f"http://{cluster.address}:{port}/generate",
        headers=headers,
        json=data,
        verify=False,
    )
    print(response.json())

    # For streaming results, use the `/generate_stream` endpoint and iterate over the results:
    # ```python
    # for message in resp:
    #    print(message)
    # ```

    # Alternatively, we can also call the model via HTTP
    # Note: We can also use a streaming route by replacing `generate` with `generate_stream`:
    print(
        f"curl http://{cluster.address}:{port}/generate -X POST -d '"
        f'{{"inputs":"{prompt_message}","parameters":{{"max_new_tokens":20}}}}'
        "' -H 'Content-Type: application/json'"
    )
