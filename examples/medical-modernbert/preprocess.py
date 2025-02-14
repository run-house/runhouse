import runhouse as rh
import s3fs
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer


def preprocess_data_to_s3(
    model_name,
    cpus,
    dataset_name="MedRAG/pubmed",
    max_length=8192,
    test_size=0.2,
    sample_rows=None,
    load_from_s3_path=None,
):

    # Load from S3
    if load_from_s3_path:
        fs = s3fs.S3FileSystem()
        s3_path = load_from_s3_path  # "s3://rh-demo-external/pubmed.parquet"
        fs.get(s3_path, "pubmed.parquet")
        dataset = Dataset.from_parquet("pubmed.parquet")
    else:
        retry = 0
        while retry < 5:
            try:
                dataset = load_dataset(dataset_name)
                dataset = dataset["train"]

            except Exception as e:
                retry += 1
                print(f"Error loading dataset. {e} Retrying... {retry}")
                continue
            break

    # Take just sample rows of dataset
    if sample_rows:
        dataset = dataset.select(range(sample_rows))

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def preprocess(examples):
        tokenized = tokenizer(
            examples["content"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_special_tokens_mask=True,
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    tokenized_dataset = dataset.map(
        preprocess,
        batched=True,
        num_proc=int(float(cpus)) - 2,
        remove_columns=dataset.column_names,
    )
    processed_dataset = tokenized_dataset.train_test_split(test_size=test_size, seed=42)
    test_eval_dataset = processed_dataset["test"].train_test_split(
        test_size=0.5, seed=42
    )
    processed_dataset["eval"] = test_eval_dataset["train"]
    processed_dataset["test"] = test_eval_dataset["test"]

    fs = s3fs.S3FileSystem()

    for split in ["train", "eval", "test"]:
        if sample_rows:
            s3_path = (
                f"s3://rh-demo-external/pubmed_processed/sample_{sample_rows}/{split}"
            )
        else:
            s3_path = f"s3://rh-demo-external/pubmed_processed/{split}"
        with fs.open(s3_path, "wb") as f:
            processed_dataset[split].to_parquet(f)


if __name__ == "__main__":
    data_img = (
        rh.Image()
        .install_packages(["datasets", "s3fs", "transformers"])
        .sync_secrets(["huggingface", "aws"])
    )
    preproc_compute = rh.compute(
        name="rh-data-preprocessing",
        num_cpus="24+",
        memory="60+",
        disk_size=2000,
        provider="aws",
        image=data_img,
        use_spot=False,
    ).up_if_not()
    # preproc_compute.restart_server()
    preproc_remote = rh.function(preprocess_data_to_s3).to(
        preproc_compute, name="preproc"
    )
    preproc_remote(
        model_name="answerdotai/ModernBERT-base",
        cpus=preproc_compute.compute_properties["num_cpus"],
        sample_rows=100000,
    )
    preproc_compute.teardown()
    # preproc_remote(model_name = "answerdotai/ModernBERT-base", cpus = preproc_compute.compute_properties["num_cpus"])
