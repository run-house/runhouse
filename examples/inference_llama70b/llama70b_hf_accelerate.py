import runhouse as rh
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


def bring_up_cluster(
    img,
    num_gpus=8,
    num_nodes=1,
    gpu_type="L4",
    use_spot=True,
    launcher="local",
    autostop_mins=120,
    restart_server=False,
):
    # Requires access to a cloud account with the necessary permissions to launch compute.
    cluster = rh.compute(
        name=f"rh-{num_gpus}x{num_nodes}",
        num_nodes=num_nodes,
        instance_type=f"{gpu_type}:{num_gpus}",
        provider="aws",
        image=img,
        use_spot=use_spot,
        launcher=launcher,
        autostop_mins=autostop_mins,
    ).up_if_not()

    if restart_server:
        cluster.restart_server()

    return cluster


if __name__ == "__main__":
    img = (
        rh.Image(name="llama_inference")
        .install_packages(
            [
                "torch",
                "transformers",
                "accelerate",
            ]
        )
        .sync_secrets(["huggingface"])
    )

    # Requires access to a cloud account with the necessary permissions to launch compute.
    cluster = bring_up_cluster(img, restart_server=True)
    # cluster = bring_up_cluster(img, num_gpus= 8, num_nodes = 1, gpu_type = 'L4', use_spot = True, launcher = 'local', autostop_mins = 120, restart = True)

    # Tunnel to Ray dashboard on port 8265
    cluster.ssh_tunnel(8265, 8265)

    # Send our inference class to remote and instantiate it remotely
    inference_remote = rh.module(Llama70B).to(cluster, name="llama_model")
    llama = inference_remote(name="inference_llama70b")

    # Make a query
    query = "What is the best type of bread in the world?"
    generated_text = llama.generate(query)
    print(generated_text)
