# # Launch DeepSeek-R1 on Your Own Cloud Account
# DeepSeek-R1 32B distill is a smaller yet still powerful distillation of the DeepSeek R1 model.
# The performance of the Qwen-32B is roughly comparable across many tasks to o1-mini, with
# the obvious advantage of being able to run within your own cloud.
#
# The best part is that we can run inference over an un-quantized version with vLLM using
# 4 x L4 GPUs. These are very available on AWS and GCP (A10 is a good sub on Azure), as well as
# most vertical GPU cloud providers, and the spot price is just $1/hour.
#
# We can easily add additional nodes and additional GPUs, and using vLLM to increase
# tensor and pipeline parallelism.
#
# Note, there must be a remote machine with Kubetorch running, and you need a Kubeconfig locally.
# Assuming that's the case, you simply need to install Kubetorch and the vllm package:
# ```shell
# $ pip install "kubetorch" torch vllm
# ```
#
# We'll be downloading the model from Hugging Face, so we may need to set up our Hugging Face token:
# ```shell
# $ export HF_TOKEN=<your huggingface token>
# ```


# ## Defining the vLLM Inference Class
# We define a class that will hold the model and allow us to send prompts to it.
# This is regular, undecorated Python code, that implements methods to
# load the model (automatically downloading from HuggingFace), and to generate text from a prompt.
import kubetorch as kt
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
# Now we will define compute using Kubetorch and send our inference class to the remote compute.
# First, we define an image with torch and vllm and 8 x L4 with 1 node.
# Then, we send our inference class to the remote compute and instantiate a the remote inference class
# with the name `deepseek` which we can access by name later. Finally, we call the remote inference class
# as if it were local to generate text from a list of prompts and print the results. If you launch with multiple nodes
# you can take advantage of vllm's parallelism.

if __name__ == "__main__":
    num_gpus = 8
    gpu_type = "L4"

    # Define the image and compute
    img = kt.images.pytorch().pip_install(["vllm"]).sync_secrets(["huggingface"])
    gpus = kt.compute(gpus=f"{gpu_type}:{num_gpus}")

    # Send the inference class to the remote compute
    init_args = dict(
        num_gpus=num_gpus,
    )
    inference_remote = kt.cls(DeepSeek_Distill_Qwen_vLLM).to(gpus, kwargs=init_args)

    # Run inference remotely and print the results
    queries = [
        "What is the relationship between bees and a beehive compared to programmers and...?",
        "How many R's are in Strawberry?",
        "Roman numerals are formed by appending the conversions of decimal place values from highest to lowest. Converting a decimal place value into a Roman numeral has the following rules: If the value does not start with 4 or 9, select the symbol of the maximal value that can be subtracted from the input, append that symbol to the result, subtract its value, and convert the remainder to a Roman numeral. If the value starts with 4 or 9 use the subtractive form representing one symbol subtracted from the following symbol, for example, 4 is 1 (I) less than 5 (V): IV and 9 is 1 (I) less than 10 (X): IX. Only the following subtractive forms are used: 4 (IV), 9 (IX), 40 (XL), 90 (XC), 400 (CD) and 900 (CM). Only powers of 10 (I, X, C, M) can be appended consecutively at most 3 times to represent multiples of 10. You cannot append 5 (V), 50 (L), or 500 (D) multiple times. If you need to append a symbol 4 times use the subtractive form. Given an integer, write Python code to convert it to a Roman numeral.",
    ]

    outputs = inference_remote.generate(queries)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt}, Generated text: {generated_text}")
