import numpy as np
import torch
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer


def collate_fn(batch):
    batch = {key: torch.tensor([item[key] for item in batch]) for key in batch[0]}
    return batch


def create_dataloader(dataset_path, batch_size=1):
    dataset = Dataset.from_parquet(dataset_path)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        collate_fn=collate_fn,  # Convert to tensors
    )


def print_gpu_memory():
    gpu_id = torch.cuda.current_device()
    allocated = torch.cuda.memory_allocated(gpu_id) / 1024**2  # Convert to MB
    reserved = torch.cuda.memory_reserved(gpu_id) / 1024**2  # Convert to MB
    print(
        f"GPU {gpu_id} - Allocated Memory: {allocated:.2f} MB, Reserved Memory: {reserved:.2f} MB"
    )


def get_embeddings(model, dataloader, device="cuda:0"):
    torch.cuda.empty_cache()

    model.eval()
    model.to(device)

    all_embeddings = []
    all_masks = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            print_gpu_memory()
            print("batch")

            # Get model outputs
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            # Get the last hidden state
            last_hidden_state = outputs.last_hidden_state.to(torch.float32)

            # Move tensors to CPU and convert to numpy
            embeddings = last_hidden_state.cpu().numpy()
            masks = attention_mask.cpu().numpy()

            all_embeddings.append(embeddings)
            all_masks.append(masks)

    # Concatenate all batches
    embeddings = np.concatenate(all_embeddings, axis=0)
    masks = np.concatenate(all_masks, axis=0)

    return embeddings, masks


def get_pooled_embeddings(embeddings, attention_masks, pool_type="mean"):
    if pool_type == "cls":
        return embeddings[:, 0, :]

    # Mean pooling
    input_mask_expanded = attention_masks[..., np.newaxis]
    sum_embeddings = np.sum(embeddings * input_mask_expanded, axis=1)
    sum_mask = np.sum(input_mask_expanded, axis=1)
    return sum_embeddings / sum_mask


def run_inference(
    base_model,
    trained_model_path,
    dataset_path,
    batch_size=32,
    device="cuda:0",
    pool_type="mean",
):
    dataloader = create_dataloader(dataset_path, batch_size=batch_size)
    trained_model = AutoModel.from_pretrained(
        trained_model_path, torch_dtype=torch.bfloat16
    )
    trained_embeddings, trained_masks = get_embeddings(
        trained_model, dataloader, device
    )
    del trained_model
    torch.cuda.empty_cache()

    base_model = AutoModel.from_pretrained(base_model, torch_dtype=torch.bfloat16)
    base_embeddings, base_masks = get_embeddings(base_model, dataloader, device)
    return trained_embeddings[:, 0, :], base_embeddings[:, 0, :]


class EmbeddingsPredictor:
    def __init__(self):
        from transformers import pipeline

        # Set "cuda:0" as the device so the Huggingface pipeline uses GPU.
        self.model = pipeline("text-generation", model="gpt2", device="cuda:0")

    def __call__(self, batch: Dict[str, np.ndarray]) -> Dict[str, list]:
        predictions = self.model(
            list(batch["data"]), max_length=20, num_return_sequences=1
        )
        batch["output"] = [sequences[0]["generated_text"] for sequences in predictions]
        return batch


# Use 2 actors, each actor using 1 GPU. 2 GPUs total.
predictions = ds.map_batches(
    HuggingFacePredictor,
    num_gpus=1,
    # Specify the batch size for inference.
    # Increase this for larger datasets.
    batch_size=1,
    # Set the concurrency to the number of GPUs in your cluster.
    concurrency=2,
)
predictions.show(limit=1)


# Usage example
if __name__ == "__main__":

    print("Run from notebook")

    # Run inference
