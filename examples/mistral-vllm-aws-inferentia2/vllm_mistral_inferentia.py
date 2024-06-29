import runhouse as rh

MAX_TOKENS = 100


class VLLM:
    def __init__(
        self, model_id="aws-neuron/Mistral-7B-Instruct-v0.2-seqlen-2048-bs-1-cores-2"
    ):
        from vllm import AsyncEngineArgs, AsyncLLMEngine

        self.engine = AsyncLLMEngine.from_engine_args(
            AsyncEngineArgs(model=model_id, max_model_len=MAX_TOKENS)
        )

    async def generate(
        self,
        prompt: str = "Explain superconductors like I'm five years old",
        max_tokens: int = MAX_TOKENS,
    ):
        import uuid

        from vllm import SamplingParams

        prompt = f"<s>[INST] You are a helpful and honest assistant. {prompt} [/INST] "
        stream = await self.engine.add_request(
            uuid.uuid4().hex, prompt, SamplingParams(max_tokens=max_tokens)
        )

        cursor = 0
        text = None
        async for request_output in stream:
            text = request_output.outputs[0].text
            print(text[cursor:])
            cursor = len(text)

        return text


if __name__ == "__main__":
    port = 8080
    cluster = rh.cluster(
        name="rh-inf2-8xlarge",
        instance_type="inf2.8xlarge",
        image_id="ami-0e0f965ee5cfbf89b",
        region="us-east-1",
        disk_size=512,
        provider="aws",
        open_ports=[port],
    ).up_if_not()

    cluster.run(
        [
            "python -m pip config set global.extra-index-url https://pip.repos.neuron.amazonaws.com",
            "python -m pip install neuronx-cc==2.* torch-neuronx==2.1.2.2.1.0 transformers-neuronx==0.10.0.360",
        ],
    )

    env = rh.env(
        name="vllm_env",
        secrets=["huggingface"],
        working_dir="local:./",
        setup_cmds=[
            "git clone https://github.com/vllm-project/vllm.git",
            "pip install -r vllm/requirements-neuron.txt",
            "pip install -e vllm",
        ],
        env_vars={"NEURON_RT_NUM_CORES": "2"},
    )

    RemoteVLLM = rh.module(VLLM).to(cluster, env=env)
    remote_tgi_app = RemoteVLLM(name="mistral-vllm")
    prompt_message = "What is Deep Learning?"
    print(remote_tgi_app.generate(prompt_message))
