# # Deploy Mistral's 7B Model with TGI on AWS EC2

# This example demonstrates how to deploy a
# [TGI model](https://huggingface.co/docs/text-generation-inference/messages_api) on AWS EC2 using Runhouse.
# This example draws inspiration from
# [Huggingface's tutorial on AWS SageMaker](https://huggingface.co/blog/text-generation-inference-on-inferentia2).
# Zephyr is a 7B fine-tuned version of [Mistral's 7B-v0.1 model](https://huggingface.co/mistralai/Mistral-7B-v0.1).
#
# ## Setup credentials and dependencies
# Install the required dependencies:
# ```shell
# $ pip install openai "runhouse[aws]"
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
# We import runhouse and openai, the required libraries to have installed locally:

import time
from pathlib import Path

import runhouse as rh
from openai import OpenAI


# Next, we define a class that will hold the model and allow us to send prompts to it.
# You'll notice this class inherits from `rh.Module`.
# This is a Runhouse class that allows you to
# run code in your class on a remote machine.
#
# Learn more in the [Runhouse docs on functions and modules](/docs/tutorials/api-modules).
class TGIInference(rh.Module):
    def __init__(
        self,
        model_id="teknium/OpenHermes-2.5-Mistral-7B",
        image_uri="ghcr.io/huggingface/text-generation-inference:1.4",
    ):
        super().__init__()
        self.docker_client = None

        self.model_id = model_id
        self.image_uri = image_uri

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
        timeout = 300

        container = self.docker_client.containers.run(
            self.image_uri,
            name=self.container_name,
            detach=True,
            ports={"80/tcp": self.container_port},
            volumes={data_volume_path: {"bind": "/data", "mode": "rw"}},
            command="--model-id " + self.model_id,
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
# :::note{.info title="Note"}
# Make sure that your code runs within a `if __name__ == "__main__":` block, as shown below. Otherwise,
# the script code will run when Runhouse attempts to run code remotely.
# :::
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
    # Learn more in the [Runhouse docs on envs](/docs/tutorials/api-envs).
    env = rh.env(
        name="tgi_env",
        reqs=["docker", "openai", "torch", "transformers"],
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
    prompt_messages = [
        {"role": "user", "content": "What is your favourite condiment?"},
        {
            "role": "assistant",
            "content": "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount "
            "of zesty flavour to whatever I'm cooking up in the kitchen!",
        },
        {"role": "user", "content": "Do you have mayonnaise recipes?"},
    ]

    # We'll use the Messages API to send the prompt to the model.
    # See [here](https://huggingface.co/docs/text-generation-inference/messages_api#streaming) for more info
    # on the Messages API, and using the OpenAI python client

    # Initialize the OpenAI client, with the URL set to the cluster's address:
    base_url = f"http://{cluster.head_ip}:{port}/v1"
    client = OpenAI(base_url=base_url, api_key="-")

    # Call the model with the prompt messages:
    chat_completion = client.chat.completions.create(
        model="tgi", messages=prompt_messages, stream=False
    )
    print(chat_completion)

    # For streaming results, set `stream=True` and iterate over the results:
    # ```python
    # for message in chat_completion:
    #    print(message)
    # ```

    # Alternatively, we can also call the model via HTTP:
    print(
        f"curl http://{cluster.head_ip}:{port}/v1/chat/completions -X POST -d '"
        '{"model": "tgi", "stream": false, "messages": [{"role": "system", "content": "You are a helpful assistant."},'
        '{"role": "user", "content": "What is deep learning?"}]}'
        "' -H 'Content-Type: application/json'"
    )
