# # Launch DeepSeek-R1 on Your Own Cloud Account
# DeepSeek-R1 32B distill is a smaller yet still powerful distillation of the DeepSeek R1 model.
# The performance of the Qwen-32B is roughly comparable across many tasks to o1-mini, with
# the obvious advantage of being able to run within your own cloud.
#
# The best part is that we can run inference over an un-quantized version with vLLM using
# 4 x L4 GPUs. These are very available on AWS and GCP (A10 is a good sub on Azure), as well as
# most vertical GPU cloud providers, and the spot price is just $1/hour.
#
# We can easily add additional nodes and additional GPUs with Runhouse, and calling LLM to increase
# tensor and pipeline parallelism.
#
# To get started, you simply need to install Runhouse with additional install for the cloud you want to use
# ```shell
# $ pip install "runhouse[aws]" torch vllm
# ```
#
# If you do not have a Runhouse account, you will launch from your local machine. Set up your
# cloud provider CLI:
# ```shell
# $ aws configure
# $ gcloud init
# ```
# We'll be downloading the model from Hugging Face, so we may need to set up our Hugging Face token:
# ```shell
# $ export HF_TOKEN=<your huggingface token>
# ```


# ## Defining the vLLM Inference Class
# We define a class that will hold the model and allow us to send prompts to it.
# This is regular, undecorated Python code, that implements methods to
# load the model (automatically downloading from HuggingFace), and to generate text from a prompt.
import runhouse as rh
from vllm import LLM, SamplingParams


class DeepSeek_Distill_Qwen_vLLM:
    def __init__(self, num_gpus, model_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"):
        self.model_id = model_id
        self.model = None
        self.num_gpus = num_gpus

    def load_model(self, pipeline_parallel_size=1):
        print("loading model")
        self.model = LLM(
            self.model_id,
            tensor_parallel_size=self.num_gpus,
            pipeline_parallel_size=pipeline_parallel_size,
            dtype="bfloat16",
            trust_remote_code=True,
            max_model_len=8192,  # Reduces size of KV store at the cost of length
        )
        print("model loaded")

    def generate(
        self, queries, temperature=1, top_p=0.95, max_tokens=2560, min_tokens=32
    ):
        if self.model is None:
            self.load_model()

        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            min_tokens=min_tokens,
        )  # non-exhaustive of possible options here

        outputs = self.model.generate(queries, sampling_params)
        return outputs


# ## Launch Compute and Run Inference
# Now we will define compute using Runhouse and send our inference class to the remote machine.
# First, we define an image with torch and vllm and a cluster with 4 x L4 with 1 node.
# Then, we send our inference class to the remote cluster and instantiate a the remote inference class
# with the name `deepseek` which we can access by name later. Finally, we call the remote inference class
# as if it were local to generate text from a list of prompts and print the results. If you launch with multiple nodes
# you can take advantage of vllm's parallelism.

if __name__ == "__main__":
    img = (
        rh.Image(name="vllm_inference")
        .install_packages(
            [
                "torch",
                "vllm",
            ]
        )
        .sync_secrets(["huggingface"])
    )

    num_gpus = 4
    num_nodes = 1
    gpu_type = "L4"
    use_spot = True
    launcher = "local"
    autostop_mins = 120
    provider = "aws"  # gcp, azure, lambda, etc.

    cluster = rh.compute(
        name=f"rh-{num_gpus}x{num_nodes}",
        num_nodes=num_nodes,
        instance_type=f"{gpu_type}:{num_gpus}",
        provider=provider,
        image=img,
        use_spot=use_spot,
        launcher=launcher,
        autostop_mins=autostop_mins,
    ).up_if_not()  # use cluster.restart_server() if you need to reset the remote cluster without tearing it down

    inference_remote = rh.module(DeepSeek_Distill_Qwen_vLLM).to(
        cluster, name="deepseek_vllm"
    )
    llama = inference_remote(
        name="deepseek", num_gpus=num_gpus
    )  # Instantiate class. Can later use cluster.get("deepseek", remote = True) to grab remote inference if already running

    cluster.ssh_tunnel(
        8265, 8265
    )  # View cluster resource utilization dashboard on localhost:8265

    queries = [
        "What is the relationship between bees and a beehive compared to programmers and...?",
        "How many R's are in Strawberry?",
        "Roman numerals are formed by appending the conversions of decimal place values from highest to lowest. Converting a decimal place value into a Roman numeral has the following rules: If the value does not start with 4 or 9, select the symbol of the maximal value that can be subtracted from the input, append that symbol to the result, subtract its value, and convert the remainder to a Roman numeral. If the value starts with 4 or 9 use the subtractive form representing one symbol subtracted from the following symbol, for example, 4 is 1 (I) less than 5 (V): IV and 9 is 1 (I) less than 10 (X): IX. Only the following subtractive forms are used: 4 (IV), 9 (IX), 40 (XL), 90 (XC), 400 (CD) and 900 (CM). Only powers of 10 (I, X, C, M) can be appended consecutively at most 3 times to represent multiples of 10. You cannot append 5 (V), 50 (L), or 500 (D) multiple times. If you need to append a symbol 4 times use the subtractive form. Given an integer, write Python code to convert it to a Roman numeral.",
    ]

    outputs = llama.generate(queries)

    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt}, Generated text: {generated_text}")
