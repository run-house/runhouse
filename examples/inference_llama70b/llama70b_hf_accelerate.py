# ## Inference Llama 3 with Accelerate

# Make sure to sign the waiver on the [Hugging Face model](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)
# page so that you can access it.
#
import kubetorch as kt
import torch
from transformers import pipeline


class Llama70B:
    def __init__(self, model_id="meta-llama/Llama-3.3-70B-Instruct"):
        self.model_id = model_id
        self.pipeline = None

    # HF can take a long time (3 hours+) to download the model. May be worth saving it elsewhere for repeated runs.
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

# ## Define Compute and Execution
#
# Now, we define code that will run locally when we run this script and set up
# our module on a remote compute. First, we define compute with the desired instance type.
# Our `gpus` requirement here is defined as `8`, which is the accelerator count that we need.

if __name__ == "__main__":
    # First, we define the image for our module. This includes the required dependencies that need
    # to be installed on the remote machine, as well as any secrets that need to be synced up from local to remote.
    # Then, we launch compute with attached GPUs.
    # Finally, passing `huggingface` to the `env vars` method will load the Hugging Face token.
    img = (
        kt.Image(image_id="nvcr.io/nvidia/pytorch:23.10-py3")
        .pip_install(
            [
                "transformers",
                "accelerate",
            ]
        )
        .sync_secrets(["huggingface"])
    )

    gpu = kt.Compute(
        gpus="8",
        memory="50Gi",
        image=img,
        launch_timeout="600",
    ).autoscale(min_replicas=1, scale_to_zero_grace_period=40)

    # Finally, we define our module and run it on the remote gpu. We construct it normally and then call
    # `to` to run it on the remote compute.
    inference_remote = kt.cls(Llama70B).to(cluster, name="llama_model")
    llama = inference_remote(name="inference_llama70b")


    query = "What's the best treatment for sunburn?"
    generated_text = llama.generate(query)
    print(generated_text)
