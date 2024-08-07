class LlamaModel:
    def __init__(self, model_id="meta-llama/Meta-Llama-3-8B-Instruct", **model_kwargs):
        super().__init__()
        self.model_id, self.model_kwargs = model_id, model_kwargs
        self.engine = None

    def load_engine(self):
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.engine.async_llm_engine import AsyncLLMEngine
        from vllm.model_executor.parallel_utils.parallel_state import (
            destroy_model_parallel,
        )

        # This vLLM function resets the global variables, which enables initializing models
        destroy_model_parallel()

        args = AsyncEngineArgs(
            model=self.model_id,  # Hugging Face Model ID
            tensor_parallel_size=1,  # Increase if using additional GPUs
            trust_remote_code=True,  # Trust remote code from Hugging Face
            enforce_eager=True,  # Set to False in production to improve performance
        )
        self.engine = AsyncLLMEngine.from_engine_args(args)

    async def generate(self, prompt: str, **sampling_params):
        from vllm.sampling_params import SamplingParams
        from vllm.utils import random_uuid

        if not self.engine:
            self.load_engine()

        sampling_params = SamplingParams(**sampling_params)
        request_id = random_uuid()
        results_generator = self.engine.generate(prompt, sampling_params, request_id)

        async for output in results_generator:
            final_output = output
        responses = []
        for output in final_output.outputs:
            responses.append(output.text)
        return responses
