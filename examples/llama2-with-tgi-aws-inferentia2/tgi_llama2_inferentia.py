# # Deploy Llama 2 7B Model with TGI on AWS Inferentia2
# This example demonstrates how to deploy Llama 2 7B with
# [TGI](https://github.com/huggingface/optimum-neuron/tree/main/text-generation-inference) on AWS Inferentia2
# using Runhouse, specifically with the [AWS Neuron SDK](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/).
# It will launch and set up the hardware, deploy the TGI container, and show multiple ways to run inference with
# the model, including the Messages API and OpenAI client.
#
# ## Setup credentials and dependencies
# Install the required dependencies:
# ```shell
# $ pip install "runhouse[aws]"
# ```
#
# We'll be using [Llama 2](https://huggingface.co/aws-neuron/Llama-2-7b-hf-neuron-budget), which is a gated
# model and requires a Hugging Face token in order to access it.
#
# To set up your Hugging Face token, run the following command in your local terminal:
# ```shell
# $ export HF_TOKEN=<your huggingface token>
# ```
#
# We'll be launching an AWS EC2 instance via [SkyPilot](https://github.com/skypilot-org/skypilot), so we need to make
# sure our AWS credentials are set up with SkyPilot:
# ```shell
# $ aws configure
# $ sky check
# ```
#
# ## Setting up a model class
import time
from pathlib import Path

import requests

# We import runhouse, the only required library to have installed locally:
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
        model_id="aws-neuron/Llama-2-7b-hf-neuron-budget",
        image_uri="ghcr.io/huggingface/neuronx-tgi:latest",
        max_batch_size=2,
        max_input_length=1024,
        max_batch_prefill_tokens=1024,
        max_total_tokens=2048,
        **model_kwargs,
    ):
        super().__init__()
        self.docker_client = None

        self.model_id = model_id
        self.image_uri = image_uri
        self.max_batch_size = max_batch_size
        self.max_input_length = max_input_length
        self.max_total_tokens = max_total_tokens
        self.max_batch_prefill_tokens = max_batch_prefill_tokens
        self.model_kwargs = model_kwargs

        self.container_port = 8080
        self.container_name = "text-generation-service"

    def _load_docker_client(self):
        import docker

        self.docker_client = docker.from_env()

    def _run_container(self):
        print("Running container")
        # https://github.com/huggingface/optimum-neuron/tree/main/text-generation-inference#using-a-neuron-model-from-the--huggingface-hub-recommended
        model_cmd = (
            f"--model-id {self.model_id} "
            f"--max-batch-size {self.max_batch_size} "
            f"--max-input-length {self.max_input_length} "
            f"--max-total-tokens {self.max_total_tokens} "
            f"--max-batch-prefill-tokens {self.max_batch_prefill_tokens}"
        )

        # Add any other model kwargs to the command
        # https://github.com/huggingface/optimum-neuron/tree/main/text-generation-inference#choosing-service-parameters
        for key, value in self.model_kwargs.items():
            model_cmd += f" --{key} {value}"

        # shared volume mounted at /data to cache the models
        home_dir = str(Path.home())
        data_volume_path = f"{home_dir}/data"

        start_time = time.time()
        timeout = 600

        # Load the HF token which was synced onto the cluster as part of the Image setup
        hf_secret = rh.secret(provider="huggingface")
        hf_token = hf_secret.values.get("token")

        # Note: device we've chosen has one neuron device, but we can specify multiple devices if relevant
        container = self.docker_client.containers.run(
            image=self.image_uri,
            name=self.container_name,
            detach=True,
            ports={"80/tcp": self.container_port},
            volumes={data_volume_path: {"bind": "/data", "mode": "rw"}},
            environment={"HF_TOKEN": hf_token},
            devices=["/dev/neuron0"],
            command=model_cmd,
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

    def _model_is_deployed(self):
        if self.docker_client is None:
            self._load_docker_client()

        containers = self.docker_client.containers.list(
            filters={"name": self.container_name}
        )
        return bool(containers)

    def deploy(self):
        if self._model_is_deployed():
            return

        print("Model has not yet been deployed, loading image and running container.")

        self._run_container()

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
# Our `instance_type` here is defined as `inf2.8xlarge`, which is
# an [AWS inferentia instance type on EC2](https://aws.amazon.com/ec2/instance-types/inf2/).
#
# We also open port 8080, which is the port that the TGI model will be running on.
#
# We use a specific `image_id`, which in this case is the
# [Deep Learning AMI Base Neuron](https://aws.amazon.com/releasenotes/aws-deep-learning-ami-base-neuron-ubuntu-20-04/)
# which comes with the AWS Neuron drivers preinstalled. The image_id is region-specific. To change the region,
# use the AWS CLI command on the page above under "Query AMI-ID with AWSCLI."
# Learn more about clusters in the [Runhouse docs](/docs/tutorials/api-clusters).
#
# :::note{.info title="Note"}
# Make sure that your code runs within a `if __name__ == "__main__":` block, as shown below. Otherwise,
# the script code will run when Runhouse attempts to run code remotely.
# :::
if __name__ == "__main__":
    port = 8080

    # We define the image for our module. This includes the required dependencies that need
    # to be installed on the remote machine, as well as any secrets that need to be synced up from local to remote.
    #
    # Passing `huggingface` to the `sync_secrets` method will load the Hugging Face token we set up earlier. This is
    # needed to download the model from the Hugging Face model hub. Runhouse will handle saving the token down
    # on the cluster in the default Hugging Face token location (`~/.cache/huggingface/token`).
    img = (
        rh.Image(name="tgi", image_id="ami-0e0f965ee5cfbf89b")
        .install_packages(["docker"])
        .sync_secrets(["huggingface"])
    )

    # Launch an 8xlarge AWS Inferentia2 instance
    cluster = rh.cluster(
        name="rh-inf2-8xlarge",
        instance_type="inf2.8xlarge",
        region="us-east-1",
        disk_size=512,
        provider="aws",
        image=img,
        open_ports=[port],
    ).up_if_not()

    # We can run commands directly on the cluster via `cluster.run()`. Here, we set up the environment for our
    # upcoming environment (more on that below) that installed some AWS-neuron specific libraries.
    # We install the `transformers-neuronx` library before restarting the Runhouse cluster (not affecting the underlying infra) to avoid
    # [common errors](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/frameworks/torch/torch-neuronx/training-troubleshooting.html):
    cluster.run(
        [
            "python -m pip config set global.extra-index-url https://pip.repos.neuron.amazonaws.com",
            "python -m pip install neuronx-cc==2.* torch-neuronx==1.13.1.1.13.1 transformers-neuronx==0.9.474",
        ],
    )

    cluster.restart_server()

    # Finally, we define our module and run it on the remote cluster. We construct it normally and then call
    # `get_or_to` to run it on the remote cluster. Using `get_or_to` allows us to load the exiting Module
    # by the name `tgi_inference` if it was already put on the cluster. If we want to update the module each
    # time we run this script, we can use `to` instead of `get_or_to`.
    remote_tgi_model = TGIInference().get_or_to(cluster, name="tgi-inference")

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
    # on using the Messages API

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
        f"http://{cluster.head_ip}:{port}/generate", headers=headers, json=data
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
        f"curl http://{cluster.head_ip}:{port}/generate -X POST -d '"
        f'{{"inputs":"{prompt_message}","parameters":{{"max_new_tokens":20}}}}'
        "' -H 'Content-Type: application/json'"
    )
