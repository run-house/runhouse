# # Launch DeepSeek-R1 on Your Own Cloud Account
# DeepSeek-R1 32B distill is a smaller yet still powerful distillation of the DeepSeek R1 model.
# The performance of the Qwen-32B is slightly worse
# The best part is that we can run inference over an un-quantized version with vLLM using
# 4 x L4 GPUs. These are very available on AWS and GCP (A10 is a good sub on Azure), as well as
# most vertical GPU cloud providers.

import runhouse as rh
from vllm import LLM, SamplingParams


class DeepSeek_Distill_Qwen_vLLM:
    def __init__(self, num_gpus, model_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"):
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
        self, queries, temperature=1, top_p=0.95, max_tokens=2560, min_tokens=32
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

    # Requires access to a cloud account with the necessary permissions to launch compute.
    num_gpus = 4
    num_nodes = 1
    gpu_type = "L4"
    use_spot = True
    launcher = "local"
    autostop_mins = 120
    provider = "aws"  # gcp, azure, lambda, etc.

    cluster = rh.cluster(
        name=f"rh-{num_gpus}x{num_nodes}",
        num_nodes=num_nodes,
        instance_type=f"{gpu_type}:{num_gpus}",
        provider=provider,
        image=img,
        use_spot=use_spot,
        launcher=launcher,
        autostop_mins=autostop_mins,
    ).up_if_not()

    inference_remote = rh.module(DeepSeek_Distill_Qwen_vLLM).to(
        cluster, name="deepseek_vllm"
    )
    llama = inference_remote(name="deepseek", num_gpus=num_gpus)
    # llama = cluster.get("deepseek", remote = True) # Grab the remote inference module if already running from remote

    cluster.ssh_tunnel(8265, 8265)  # View Ray Dashboard on localhost:8265

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
