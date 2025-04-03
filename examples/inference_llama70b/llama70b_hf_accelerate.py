# # Launch Llama70B Inference on Your Own Kubernetes Cluster
# The Llama-70B is one of the more powerful open source models, but still easily fits on a single node of GPUs - even 8 x L4s.
# Of course, inference speed will improve if you change to A100s or H100s on your cloud provider,
# but the L4s are cost-effective for experimentation or low throughput, latency-agnostic workloads, and also
# fairly available on spot meaning you can serve this model for as low as ~$4/hour. This is the full model,
# not a quantization of it. In this example, we use HuggingFace Accelerate.

import kubetorch as kt
import torch
from transformers import pipeline

# ## Define Inference Class
# We define a regular class that uses HuggingFace Pipelines to run inference on the Llama-70B model.
# We will send this class to the remote compute by decorating the class to specify the compute
# and autoscaling.
img = (
    kt.images.pytorch()
    .pip_install(["transformers", "accelerate"])
    .sync_secrets(["huggingface"])
)


@kt.compute(gpus="L4:8", image=img, name="llama70b")
@kt.distribute("auto", num_replicas=(0, 4))
class Llama70B:
    def __init__(self, model_id="meta-llama/Llama-3.3-70B-Instruct"):
        self.model_id = model_id
        self.pipeline = None

    def load_pipeline(self):
        self.pipeline = pipeline(
            "text-generation",
            model=self.model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )

    def generate(self, query, temperature=1, max_new_tokens=100, top_p=0.9):
        if self.pipeline is None:
            self.load_pipeline()

        messages = [
            {
                "role": "system",
                "content": "You are a pirate chatbot with a love of Shakespeare who always in Shakespearian pirate speak!",
            },
            {"role": "user", "content": query},
        ]

        prompt = self.pipeline.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        terminators = [
            self.pipeline.tokenizer.eos_token_id,
            self.pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]

        outputs = self.pipeline(
            prompt,
            eos_token_id=terminators,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
        )
        return outputs[0]["generated_text"][len(prompt) :]


# ## Call Remote Service
# Once we have deployed the service with `kubetorch deploy`, we can call the service
# in Python directly by name as welll.
if __name__ == "__main__":

    llama = Llama70B.from_name("llama70b")

    query = "What is the best type of bread in the world?"
    generated_text = llama.generate(query)  # Running on your remote GPUs
    print(generated_text)
