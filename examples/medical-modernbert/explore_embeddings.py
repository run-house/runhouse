import numpy as np
import torch
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer


def collate_fn(batch):
    batch = {key: torch.tensor([item[key] for item in batch]) for key in batch[0]}
    return batch


def create_dataloader(dataset_path, batch_size=1, sample_rows=None):
    dataset = Dataset.from_parquet(dataset_path)
    if sample_rows:
        dataset = dataset.select(range(sample_rows))

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
    pool_type="cls",
    sample_rows=None,
):
    dataloader = create_dataloader(
        dataset_path, batch_size=batch_size, sample_rows=sample_rows
    )
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


# Usage example
if __name__ == "__main__":

    print("Run from notebook")

    # Run inference
