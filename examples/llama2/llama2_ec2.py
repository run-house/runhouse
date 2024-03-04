import runhouse as rh
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer


class HFChatModel(rh.Module):
    def __init__(self, model_id="meta-llama/Llama-2-13b-chat-hf", **model_kwargs):
        super().__init__()
        self.model_id, self.model_kwargs = model_id, model_kwargs
        self.tokenizer, self.model = None, None

    def load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, clean_up_tokenization_spaces=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id, **self.model_kwargs
        )

    def predict(self, prompt_text, **inf_kwargs):
        default_inf_kwargs = {
            "temperature": 0.7,
            "max_new_tokens": 500,
            "repetition_penalty": 1.0,
        }
        default_inf_kwargs.update(inf_kwargs)
        if not self.model:
            self.load_model()
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to("cuda")
        generated_ids = self.model.generate(
            **inputs, **default_inf_kwargs, streamer=TextStreamer(self.tokenizer)
        )
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]


if __name__ == "__main__":
    gpu = rh.cluster(name="rh-a10x", instance_type="A10G:1", provider="aws")
    env = rh.env(
        reqs=[
            "torch",
            "transformers==4.31.0",
            "accelerate==0.21.0",
            "bitsandbytes==0.40.2",
            "safetensors>=0.3.1",
            "scipy",
        ],
        secrets=["huggingface"],  # Needed to download Llama2
        name="llama2inference",
        working_dir="./",
    )

    remote_hf_chat_model = HFChatModel(
        model_id="meta-llama/Llama-2-13b-chat-hf",
        load_in_4bit=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    ).get_or_to(gpu, env=env, name="llama-13b-model")

    while True:
        prompt = input(
            "\n\n... Enter a prompt to chat with the model, and 'exit' to exit ...\n"
        )
        if prompt.lower() == "exit":
            break
        output = remote_hf_chat_model.predict(prompt)
        print("\n\n... Model Output ...\n")
        print(output)
