import runhouse as rh
from vllm import LLM, SamplingParams


class Llama70B_vLLM:
    def __init__(self, num_gpus, model_id="meta-llama/Llama-3.3-70B-Instruct"):
        self.model_id = model_id
        self.model = None
        self.sampling_params = None
        self.num_gpus = num_gpus

    def load_model(self, temperature=1, top_p=0.9, max_tokens=256, min_tokens=32):
        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            min_tokens=min_tokens,
        )
        print("loading model")
        self.model = LLM(
            self.model_id,
            tensor_parallel_size=self.num_gpus,
            dtype="bfloat16",
            trust_remote_code=True,
            max_model_len=8192,
        )
        print("model loaded")

    def generate(self, queries, temperature=1, top_p=0.95):
        if self.model is None:
            self.load_model(temperature, top_p)

        outputs = self.model.generate(queries, self.sampling_params)
        return outputs


if __name__ == "__main__":
    img = (
        rh.Image(name="vllm_llama_inference")
        .pip_install(
            [
                "torch",
                "vllm",
            ]
        )
        .sync_secrets(["huggingface"])
    )

    # Reuse the bring up cluster function to reuse the cluster if already up.
    from llama70b_hf_accelerate import bring_up_cluster

    # Requires access to a cloud account with the necessary permissions to launch compute.
    cluster = bring_up_cluster(img, restart_server=True)
    # cluster = bring_up_cluster(img, num_gpus= 8, num_nodes = 1, gpu_type = 'L4', use_spot = True, launcher = 'local', autostop_mins = 120, restart_server = True)

    inference_remote = rh.cls(Llama70B_vLLM).to(cluster, name="llama_model_vllm")
    llama = inference_remote(name="vllm_llama70b", num_gpus=8)
    # llama = cluster.get("vllm_llama70b", remote = True)
    queries = [
        "What is the best type of bread in the world?",
        "What are some cheeses that go with bread?",
        "What is the best way to make a sandwich?",
    ]
    outputs = llama.generate(queries)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt}, Generated text: {generated_text}")
