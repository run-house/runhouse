# # Launch DeepSeek-R1 on Your Own Cloud Account
# The Llama-70B distill of DeepSeek R1 is the most powerful and largest of the
# DeepSeek distills, but still easily fits on a single node of GPUs - even 8 x L4s with minor optimizations.
# Of course, inference speed will improve if you change to A100s or H100s on your cloud provider,
# but the L4s are cost-effective for experimentation or low throughput, latency-agnostic workloads, and also
# fairly available on spot meaning you can serve this model for as low as ~$4/hour. This is the full model,
# not a quantization of it.
#
# On benchmarks, this Llama70B distillation meets or exceeds the performance of GPT-4o-0513,
# Claude-3.5-Sonnet-1022, and o1-mini across most quantitative and coding tasks. Real world
# quality of output depends on your use case. This will run the model at ~20 tokens a second,
# which means it takes 1-5 minutes per question asked. It will take some time to download the model
# to the remote machine on the first run.
#
# We can easily add additional nodes, which will automatically form the compute. We will
# rely fully on vllm to make use of them and increasing tensor and pipeline parallelism.
#

# ## Defining the vLLM Inference Class
# We define a class that will hold the model and allow us to send prompts to it.
# This is regular, undecorated Python code, that implements methods to
# load the model (automatically downloading from HuggingFace), and to generate text from a prompt.

import kubetorch as kt
from vllm import LLM, SamplingParams


class DeepSeek_Distill_Llama70B_vLLM:
    def __init__(self, num_gpus, model_id="deepseek-ai/DeepSeek-R1-Distill-Llama-70B"):
        self.model_id = model_id
        self.model = None
        self.num_gpus = num_gpus

    def load_model(self):
        print("loading model")
        self.model = LLM(
            self.model_id,
            tensor_parallel_size=self.num_gpus,
            dtype="bfloat16",
            trust_remote_code=True,
            max_model_len=8192,  # Reduces size of KV store
        )
        print("model loaded")

    def generate(
        self, queries, temperature=0.65, top_p=0.95, max_tokens=5120, min_tokens=32
    ):
        if self.model is None:
            self.load_model()

        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            min_tokens=min_tokens,
        )

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
    inference_remote = kt.cls(DeepSeek_Distill_Llama70B_vLLM).to(gpus, kwargs=init_args)

    # Run inference remotely and print the results
    queries = [
        "What is the relationship between bees and a beehive compared to programmers and...?",
        "How many R's are in Strawberry?",
        """Roman numerals are formed by appending the conversions of decimal place values from highest to lowest.
        Converting a decimal place value into a Roman numeral has the following rules: If the value does not start with 4 or 9, select the symbol of the maximal value that can be subtracted from the input, append that symbol to the result, subtract its value, and convert the remainder to a Roman numeral.
        If the value starts with 4 or 9 use the subtractive form representing one symbol subtracted from the following symbol, for example, 4 is 1 (I) less than 5 (V): IV and 9 is 1 (I) less than 10 (X): IX. Only the following subtractive forms are used: 4 (IV), 9 (IX), 40 (XL), 90 (XC), 400 (CD) and 900 (CM). Only powers of 10 (I, X, C, M) can be appended consecutively at most 3 times to represent multiples of 10. You cannot append 5 (V), 50 (L), or 500 (D) multiple times. If you need to append a symbol 4 times use the subtractive form.
        Given an integer, write and return Python code to convert it to a Roman numeral.""",
    ]

    outputs = inference_remote.generate(queries, temperature=0.7)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt}, Generated text: {generated_text}")
