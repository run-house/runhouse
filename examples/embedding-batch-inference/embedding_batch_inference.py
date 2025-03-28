# ## Offline Batch Inference
# We will use BGE to embed a large amount of text.

import kubetorch as kt


class BGEEmbedder:
    def __init__(self, model_id="BAAI/bge-large-en-v1.5"):
        self.model_id = model_id
        self.model = None

    def load_model(self):
        from vllm import LLM

        self.model = LLM(model=self.model_id, task="embed", enforce_eager=True)

    def load_data(self, dataset_name, text_column_name, split=None, data_files=None):
        from datasets import load_dataset
        from transformers import AutoTokenizer

        dataset = load_dataset(
            dataset_name,
            data_files=data_files,
            split=split,
        )

        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        max_length = tokenizer.model_max_length

        def tokenize_function(examples):
            return tokenizer(
                examples[text_column_name],
                padding=True,
                truncation=True,
                max_length=max_length,
            )

        tokenized_dataset = dataset.map(
            tokenize_function,
            num_proc=4,
            batched=True,
            remove_columns=dataset.column_names,
            batch_size=2000,
        )
        inputs = tokenized_dataset["input_ids"]
        return [{"prompt_token_ids": inp} for inp in inputs]

    def embed_dataset(
        self, dataset_name, text_column_name, split=None, data_files=None
    ):
        if self.model is None:
            self.load_model()

        data = self.load_data(dataset_name, text_column_name, data_files, split)
        return self.model.embed(data)


if __name__ == "__main__":

    # Define an image and compute, send the embedder to the compute and embed a dataset
    img = kt.images.ubuntu().pip_install(["vllm", "datasets", "transformers"])
    compute = kt.compute(gpus="T4:1", image=img)
    embedder = kt.cls(BGEEmbedder).to(compute)
    embedder.embed_dataset(
        "wikimedia/wikipedia",
        "text",
        split="train",
        data_files="20231101.en/train-00000-of-00041.parquet",
    )

    # If we want to do this in parallel, we can distribute the embedding and call against it in parallel
    replicas = 4
    compute = kt.compute(gpus="T4:1", image=img).distribute(replicas=replicas)
    embedder = kt.cls(BGEEmbedder).to(compute)
    data_files_list = [
        "20231101.en/train-00000-of-00041.parquet",
        "20231101.en/train-00001-of-00041.parquet",
        "20231101.en/train-00002-of-00041.parquet",
        "20231101.en/train-00003-of-00041.parquet",
    ]  # ETC

    from concurrent.futures import ThreadPoolExecutor
    from functools import partial

    with ThreadPoolExecutor(max_workers=replicas) as executor:
        embed_file = partial(
            embedder.embed_dataset, "wikimedia/wikipedia", "text", split="train"
        )
        results = list(executor.map(embed_file, data_files_list))
