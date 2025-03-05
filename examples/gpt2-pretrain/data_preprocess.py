def download_and_preprocess(block_size: int = 1024, sample: int = None):
    from datasets import load_dataset
    from transformers import AutoTokenizer

    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        data_files="data/CC-MAIN-2024-51/000_00000.parquet",
        ignore_verifications=True,
    )  # Take just one crawl for simplicity / size
    dataset = dataset["train"]
    if sample:
        dataset = dataset.select(range(sample))

    # Perform train-test split
    dataset = dataset.train_test_split(test_size=0.2)

    # Initialize tiktoken tokenizer (e.g., for OpenAI's cl100k_base)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_and_chunk(examples):
        tokenized_batch = tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=block_size,
            return_tensors="pt",
        )
        labels = tokenized_batch["input_ids"].clone()
        labels[:, :-1] = tokenized_batch["input_ids"][:, 1:]
        labels[:, -1] = tokenizer.pad_token_id  # Last token becomes padding token
        return {
            "input_ids": tokenized_batch["input_ids"],
            "attention_mask": tokenized_batch["attention_mask"],
            "labels": labels,
        }

    processed_dataset = dataset.map(
        tokenize_and_chunk, batched=True, remove_columns=dataset["train"].column_names
    )

    # Save locally
    processed_dataset.save_to_disk("processed_fineweb_edu")

    print("Tokenized dataset saved to 'processed_fineweb_edu'")
