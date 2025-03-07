import logging

import os

import runhouse as rh

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim

from datasets import load_from_disk

from model import default_hparams, GPT2Model
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import StepLR
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm


class Trainer:
    def __init__(self):
        self.model = None
        self.optimizer = None
        self.train_dataset = None
        self.test_dataset = None
        self.train_loader = None
        self.eval_loader = None
        self.scheduler = None
        self.loss_fn = nn.CrossEntropyLoss()
        self.scaler = None

        self.device = None
        self.device_id = None
        self.rank = None
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

    def load_dataset(self, path, num_workers=4):
        self.logger.info(f"Loading dataset from {path}")
        dataset = load_from_disk(path)
        self.train_dataset = dataset["train"]
        self.eval_dataset = dataset["test"]

        def collate_fn(batch):
            token_lists = [
                torch.tensor(example["input_ids"])
                for example in batch
                if len(example["input_ids"]) > 0
            ]

            label_lists = [
                torch.tensor(example["labels"])
                for example in batch
                if len(example["labels"]) > 0
            ]

            if len(token_lists) == 0 or len(label_lists) == 0:
                return {"input_ids": torch.zeros(0), "labels": torch.zeros(0)}

            input_ids = torch.stack(token_lists, dim=0)
            labels = torch.stack(label_lists, dim=0)

            return {"input_ids": input_ids, "labels": labels}

        if not dist.is_initialized():
            self.init_comms()

        world_size = dist.get_world_size()

        train_sampler = DistributedSampler(
            self.train_dataset, num_replicas=world_size, rank=self.rank, shuffle=True
        )
        eval_sampler = DistributedSampler(
            self.eval_dataset, num_replicas=world_size, rank=self.rank, shuffle=False
        )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=8,
            collate_fn=collate_fn,
            sampler=train_sampler,
            num_workers=4,
        )

        self.eval_loader = DataLoader(
            self.eval_dataset,
            batch_size=8,
            collate_fn=collate_fn,
            sampler=eval_sampler,
            num_workers=4,
        )

    def init_comms(self):
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")

        if dist.is_initialized():
            self.rank = dist.get_rank()
        else:
            self.rank = 0

        self.device_id = self.rank % torch.cuda.device_count()
        self.device = torch.device(f"cuda:{self.device_id}")

    def load_model(self, hparams=default_hparams(), model_path=None):
        if model_path:
            self.model = GPT2Model(hparams)
            self.model.load_state_dict(torch.load(model_path))
            self.model.to(self.device)
        else:
            self.model = GPT2Model(hparams)
            self.model.to(self.device)

    def train_epoch(self):
        total_loss = 0
        self.model.train()

        for batch in tqdm(self.train_loader, desc="Training", disable=self.rank != 0):
            X, y = batch["input_ids"], batch["labels"]
            X = X.to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad()
            with autocast(device_type="cuda", dtype=torch.float16):
                y_pred = self.model(X)["logits"]
                loss = self.loss_fn(y_pred.reshape(-1, y_pred.size(-1)), y.reshape(-1))

            self.scaler.scale(loss).backward()

            # Gradient clipping (to prevent FP16 instability)
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            # Step optimizer and update scaler
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    def eval_epoch(self):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in tqdm(
                self.eval_loader, desc="Evaluation", disable=self.rank != 0
            ):
                X, y = batch["input_ids"], batch["labels"]
                X = X.to(self.device)
                y = y.to(self.device)

                with autocast(device_type="cuda", dtype=torch.float16):
                    y_pred = self.model(X)["logits"]
                    loss = self.loss_fn(
                        y_pred.reshape(-1, y_pred.size(-1)), y.reshape(-1)
                    )

                total_loss += loss.item()

        # Calculate average loss across all ranks
        avg_loss = torch.tensor(total_loss / len(self.eval_loader), device=self.device)
        dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
        avg_loss /= dist.get_world_size()
        return avg_loss.item()

    def train(
        self,
        dataset_path,
        epochs,
        lr=1e-4,
        weight_decay=1e-4,
        step_size=7,
        gamma=0.1,
        debug=False,
    ):

        if not dist.is_initialized():
            self.init_comms()

        if debug:
            os.environ["TORCH_USE_CUDA_DSA"] = "1"
            os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

        if not self.model:
            self.load_model()

        if not self.train_dataset:
            self.load_dataset(dataset_path)

        self.optimizer = optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = StepLR(self.optimizer, step_size=step_size, gamma=gamma)
        self.scaler = GradScaler()

        for i in range(epochs):
            loss = self.train_epoch()
            self.logger.info(f"Epoch {i} Loss: {loss}")
            self.save_model(f"model_{i}.pt")

            eval_loss = self.eval_epoch()
            self.logger.info(f"Epoch {i} Eval Loss: {eval_loss}")

        self.logger.info(f"Trained for {epochs} epochs")

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(X)
        return y_pred


if __name__ == "__main__":
    num_nodes = 3
    num_gpus_per_node = 4

    img = (
        rh.Image()
        .install_packages(
            [
                "tensorboard",
                "transformers",
                "accelerate",
                "scipy",
                "datasets",
                "s3fs",
                "torch",
                "tiktoken",
                "scikit-learn",
                "tqdm",
            ]
        )
        .sync_secrets(["huggingface", "aws"])
    )

    cluster = rh.compute(
        name=f"rh-L4x{num_gpus_per_node}x{num_nodes}",
        num_nodes=num_nodes,
        instance_type=f"L4:{num_gpus_per_node}",
        provider="aws",
        image=img,
        use_spot=False,
        autostop_mins=1200,
    ).up_if_not()  # Requires access to a cloud account with the necessary permissions to launch compute.

    # cluster.restart_server()

    from data_preprocess import download_and_preprocess

    hf_dataset_name = "HuggingFaceFW/fineweb-edu"
    hf_data_files = "data/CC-MAIN-2024-51/000_00000.parquet"  # Take this single file for ease/speed.

    remote_preprocess = (
        rh.function(download_and_preprocess)
        .to(cluster, name="preprocess")
        .distribute("pool", num_replicas=num_nodes, replicas_per_node=1)
    )

    # Run the data execution in parallel on all nodes. Alternatively, preprocess it and save to blob storage and reload.
    import concurrent.futures
    from functools import partial

    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(
            partial(
                remote_preprocess,
                hf_dataset_name=hf_dataset_name,
                hf_data_files=hf_data_files,
            ),
            range(num_nodes),
        )

    # Send the Trainer to the cluster
    RemoteTrainer = rh.cls(Trainer).to(cluster, name="gpt2")
    trainer = RemoteTrainer(name="gpt2_trainer").distribute(
        "pytorch",
        num_replicas=num_nodes * num_gpus_per_node,
        replicas_per_node=num_gpus_per_node,
    )
    trainer.load_dataset("processed_fineweb_edu")
    trainer.train(dataset_path="processed_fineweb_edu", epochs=10)
