# # Deploy Llama 7B Model with TGI on AWS EC2

# This example demonstrates how to deploy a [Llama 7B model](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)
# using [TGI](https://huggingface.co/docs/text-generation-inference/messages_api) on AWS EC2 using Runhouse.
#
# ## Setup credentials and dependencies
# Install the required dependencies:
# ```shell
# $ pip install -r requirements.txt
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
# We import runhouse, the only required library to have installed locally:

import time
from pathlib import Path

import requests

import runhouse as rh

# Next, we define a class that will hold the model and allow us to send prompts to it.
# You'll notice this class inherits from `rh.Module`.
# This is a Runhouse class that allows you to
# run code in your class on a remote machine.
#
# Learn more in the [Runhouse docs on functions and modules](/docs/tutorials/api-modules).
class TGIInference(rh.Module):
    def __init__(
        self,
        model_id="meta-llama/Llama-2-7b-chat-hf",
        image_uri="ghcr.io/huggingface/text-generation-inference:latest",
        max_input_length=2048,
        max_total_tokens=4096,
        **model_kwargs,
    ):
        super().__init__()
        self.docker_client = None

        self.model_id = model_id
        self.image_uri = image_uri
        self.max_input_length = max_input_length
        self.max_total_tokens = max_total_tokens
        self.model_kwargs = model_kwargs

        self.container_port = 8080
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

    def restart_container(self):
        if self.docker_client is None:
            self._load_docker_client()

        try:
            container = self.docker_client.containers.get(self.container_name)
            container.stop()
            container.remove()
        except Exception as e:
            raise RuntimeError(f"Failed to stop or remove container: {e}")

        # Deploy a new container
        self.deploy()


# ## Setting up Runhouse primitives
#
# Now, we define the main function that will run locally when we run this script, and set up
# our Runhouse module on a remote cluster. First, we create a cluster with the desired instance type and provider.
# Our `instance_type` here is defined as `g5.4xlarge`, which is
# an [AWS instance type on EC2](https://aws.amazon.com/ec2/instance-types/g5/) with a GPU.
#
# For this model we'll need a GPU and at least 16GB of RAM
# We also open port 8080, which is the port that the TGI model will be running on.
#
# Learn more about clusters in the [Runhouse docs](/docs/tutorials/api-clusters).
#
# NOTE: Make sure that your code runs within a `if __name__ == "__main__":` block, as shown below. Otherwise,
# the script code will run when Runhouse attempts to run code remotely.
if __name__ == "__main__":
    port = 8080
    cluster = rh.cluster(
        name="rh-g5-4xlarge",
        instance_type="g5.4xlarge",
        provider="aws",
        open_ports=[port],
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
        working_dir="./",
    )

    # Finally, we define our module and run it on the remote cluster. We construct it normally and then call
    # `get_or_to` to run it on the remote cluster. Using `get_or_to` allows us to load the exiting Module
    # by the name `tgi_inference` if it was already put on the cluster. If we want to update the module each
    # time we run this script, we can use `to` instead of `get_or_to`.
    #
    # Note that we also pass the `env` object to the `get_or_to` method, which will ensure that the environment is
    # set up on the remote machine before the module is run.
    remote_tgi_model = TGIInference().get_or_to(cluster, env=env, name="tgi-inference")

    # ## Sharing an inference endpoint
    # We can publish this module for others to use:
    # ```python
    # remote_tgi_model.share(visibility="public")
    # ```

    # Alternatively we can share with specific users:
    # ```python
    # remote_tgi_model.share(["user1@gmail.com", "user2@gmail.com"], access_level="read")
    # ```

    # Note: For more info on fine-grained access controls, see the
    # [Runhouse docs on sharing](https://www.run.house/docs/tutorials/quick-start-den#sharing).

    # ## Deploying the model
    # We can call the `deploy` method on the model class instance if it were running locally.
    # This will load and run the model on the remote cluster.
    # We only need to do this setup step once, as further calls will use the existing docker container deployed
    # on the cluster and maintain state between calls:
    remote_tgi_model.deploy()

    # ## Sending a prompt to the model
    prompt_message = "What is Deep Learning?"

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
        f"http://{cluster.address}:{port}/generate", headers=headers, json=data
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
