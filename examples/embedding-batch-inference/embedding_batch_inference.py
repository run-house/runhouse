# ## Offline Batch Inference
# We will use BGE to embed a large amount of text. We start by defining a class
# that defines each replica of our inference service, using `vllm` to load the model.
import kubetorch as kt


class BGEEmbedder:
    def __init__(self, model_id="BAAI/bge-large-en-v1.5", batch_size=None):
        self.model_id = model_id
        self.model = None
        self.batch_size = batch_size

    def load_model(self):
        from vllm import LLM

        self.model = LLM(model=self.model_id, task="embed")

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
            num_proc=7,
            batched=True,
            remove_columns=dataset.column_names,
            batch_size=2000,
        )
        inputs = tokenized_dataset["input_ids"]
        return [{"prompt_token_ids": inp} for inp in inputs]

    def embed_dataset(self, dataset_name, text_column_name, split, data_files):
        if self.model is None:
            self.load_model()

        data = self.load_data(
            dataset_name,
            text_column_name=text_column_name,
            split=split,
            data_files=data_files,
        )
        if not self.batch_size:
            results = self.model.encode(data)
            return results[0].outputs.embedding

        # Process in batches
        all_embeddings = []
        for i in range(0, len(data), self.batch_size):
            batch = data[i : i + self.batch_size]
            batch_results = self.model.encode(batch)
            batch_embeddings = [result.outputs.embedding for result in batch_results]
            all_embeddings.extend(batch_embeddings)

        return all_embeddings[0]


# ## Single replica inference
# We can use the `BGEEmbedder` class to embed a dataset. We will use the `wikimedia/wikipedia` dataset.
# The base example just does this on a singl enode.
def batch_inference_single():
    img = kt.Image(image_id="vllm/vllm-openai:latest").pip_install(["datasets"])
    compute = kt.Compute(gpus="1", cpus="7", memory="24", image=img, launch_timeout=600)

    embedder = kt.cls(BGEEmbedder).to(compute)

    embeddings = embedder.embed_dataset(
        "wikimedia/wikipedia",
        text_column_name="text",
        split="train",
        data_files="20231101.en/train-00000-of-00041.parquet",
    )

    print(f"Generated {len(embeddings)} embeddings")
    print(f"First embedding dimension: {len(embeddings[0])}")
    print(embeddings)


# ## Distributed inference, multi-node
# We can simply call `distribute` on the compute object to distribute the inference to multiple nodes for
# embrassingly parallel tasks. This is especially useful for large datasets or large scale batch jobs.
# We handle all of the distribution and parallelism for you, as well as managing an efficient queue of
# work to replicas.
def batch_inference_distributed():
    replicas = 4
    img = kt.Image(image_id="vllm/vllm-openai:latest").pip_install(["datasets"])

    compute = kt.Compute(
        gpus="1", cpus="7", memory="24", image=img, concurrency=1, launch_timeout=600
    ).distribute(workers=replicas)

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
            embedder.embed_dataset,
            "wikimedia/wikipedia",
            "text",
            "train",
        )
        results = list(executor.map(embed_file, data_files_list))

    return results


if __name__ == "__main__":
    batch_inference_single()
