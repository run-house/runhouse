# # Deploy Mistral's 7B Model on AWS Inferentia

# This example demonstrates how to deploy a
# [Mitral 7B Model](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1) on AWS Inferentia using Runhouse.
#
# ## Setup credentials and dependencies
# Install the required dependencies:
# ```shell
# $ pip install runhouse[aws]
# ```
#
# We'll be launching an AWS Inferentia instance via [SkyPilot](https://github.com/skypilot-org/skypilot), so we
# need to make sure our AWS credentials are set up with SkyPilot:
# ```shell
# $ aws configure
# $ sky check
# ```
#
# ## Setting up a model class

# We import runhouse, the only required library to have installed locally:
import runhouse as rh

# Next, we define a class that will hold the model and allow us to send prompts to it.
# You'll notice this class inherits from `rh.Module`.
# This is a Runhouse class that allows you to
# run code in your class on a remote machine.
#
# Learn more in the [Runhouse docs on functions and modules](/docs/tutorials/api-modules).
class MistralInstruct(rh.Module):
    def __init__(
        self,
        model_id="mistralai/Mistral-7B-Instruct-v0.1",
        model_path="mistralai/Mistral-7B-Instruct-v0.1-split",
        batch_size=1,
        tp_degree=2,
        n_positions=256,
        amp="bf16",
        **model_kwargs
    ):
        super().__init__()

        self.model_id = model_id
        self.model_path = model_path

        self.batch_size = batch_size
        self.tp_degree = tp_degree
        self.n_positions = n_positions
        self.amp = amp
        self.model_kwargs = model_kwargs

        self.model_cpu = None
        self.model_neuron = None
        self.tokenizer = None

    def _load_pretrained_model(self):
        from transformers import AutoModelForCausalLM
        from transformers_neuronx.module import save_pretrained_split

        self.model_cpu = AutoModelForCausalLM.from_pretrained(self.model_id)
        save_pretrained_split(self.model_cpu, self.model_path)

    def _load_neuron_model(self):
        from transformers_neuronx import constants
        from transformers_neuronx.config import NeuronConfig
        from transformers_neuronx.mistral.model import MistralForSampling

        if self.model_cpu is None:
            # Load and save the CPU model
            self._load_pretrained_model()

        # Set sharding strategy for GQA to be shard over heads
        neuron_config = NeuronConfig(
            grouped_query_attention=constants.GQA.SHARD_OVER_HEADS
        )

        # Create and compile the Neuron model
        self.model_neuron = MistralForSampling.from_pretrained(
            self.model_path,
            batch_size=self.batch_size,
            tp_degree=self.tp_degree,
            n_positions=self.n_positions,
            amp=self.amp,
            neuron_config=neuron_config,
            **self.model_kwargs
        )
        self.model_neuron.to_neuron()

    def _load_tokenizer(self):
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

    def generate(self, messages: list, return_tensors="pt", sequence_length=256):
        import torch

        if self.tokenizer is None:
            self._load_tokenizer()

        if self.model_neuron is None:
            self._load_neuron_model()

        encodeds = self.tokenizer.apply_chat_template(
            messages, return_tensors=return_tensors
        )

        # Run inference
        with torch.inference_mode():
            generated_sequence = self.model_neuron.sample(
                encodeds, sequence_length=sequence_length, start_ids=None
            )

        return [self.tokenizer.decode(tok) for tok in generated_sequence]


# ## Setting up Runhouse primitives
#
# Now, we define the main function that will run locally when we run this script, and set up
# our Runhouse module on a remote cluster. First, we create a cluster with the desired instance type and provider.
# Our `instance_type` here is defined as `inf2.8xlarge`, which is
# an [AWS instance type on Inferentia](https://aws.amazon.com/ec2/instance-types/inf2/).
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
    cluster = rh.cluster(
        name="rh-inf2-8xlarge",
        instance_type="inf2.8xlarge",
        image_id="ami-0e0f965ee5cfbf89b",
        region="us-east-1",
        disk_size=512,
        provider="aws",
    ).up_if_not()

    # We can run commands directly on the cluster via `cluster.run()`. Here, we set up the environment for our
    # upcoming environment (more on that below) that installed some AWS-neuron specific libraries.
    # We install the `transformers-neuronx` library before the env is set up in order to avoid
    # [common errors](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/frameworks/torch/torch-neuronx/training-troubleshooting.html):
    cluster.run(
        [
            "python -m pip config set global.extra-index-url https://pip.repos.neuron.amazonaws.com",
            "python -m pip install neuronx-cc==2.* torch-neuronx==1.13.1.1.13.1 transformers-neuronx==0.9.474",
        ],
    )

    # Next, we define the environment for our module. This includes the required dependencies that need
    # to be installed on the remote machine.
    #
    # Learn more in the [Runhouse docs on envs](/docs/tutorials/api-envs).
    env = rh.env(
        name="instruct_env",
        working_dir="./",
    )

    # Finally, we define our module and run it on the remote cluster. We construct it normally and then call
    # `get_or_to` to run it on the remote cluster. Using `get_or_to` allows us to load the exiting Module
    # by the name `mistral-instruct` if it was already put on the cluster. If we want to update the module each
    # time we run this script, we can use `to` instead of `get_or_to`.
    #
    # Note that we also pass the `env` object to the `get_or_to` method, which will ensure that the environment is
    # set up on the remote machine before the module is run.
    remote_instruct_model = MistralInstruct().get_or_to(
        cluster, env=env, name="mistral-instruct"
    )

    # ## Loading and prompting the model
    # We can call the `generate` method on the model class instance if it were running locally.
    # This will load the tokenizer and model on the remote cluster.
    # We only need to do this setup step once, as further calls will use the existing model on the cluster and
    # maintain state between calls:
    prompt_messages = [
        {"role": "user", "content": "What is your favourite condiment?"},
        {
            "role": "assistant",
            "content": "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount "
            "of zesty flavour to whatever I'm cooking up in the kitchen!",
        },
        {"role": "user", "content": "Do you have mayonnaise recipes?"},
    ]

    chat_completion = remote_instruct_model.generate(prompt_messages)
    print(chat_completion)
