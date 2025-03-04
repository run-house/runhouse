import os

import runhouse as rh

import torch
import torch.nn as nn
import torch.optim as optim

from model import default_hparams, GPT2Model
from torch.optim.lr_scheduler import StepLR
from torch.utils.data.dataloader import DataLoader, DistributedSampler
from torch.utils.data.dataset import Dataset


class Trainer:
    def __init__(self):
        self.model = None
        self.optimizer = None
        self.loss_fn = None
        self.train_dataset = None
        self.test_dataset = None
        self.train_loader = None
        self.eval_loader = None
        self.scheduler = None

        self.device = None
        self.device_id = None
        self.rank = None

    def load_dataset(self, path, num_workers=4):
        if not torch.distributed.is_initialized():
            self.init_comms()

        dataset = Dataset.from_parquet(path)
        self.train_dataset = dataset["train"]
        self.eval_dataset = dataset["test"]

        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()

        train_sampler = DistributedSampler(
            self.train_dataset, num_replicas=world_size, rank=rank, shuffle=True
        )
        eval_sampler = DistributedSampler(
            self.eval_dataset, num_replicas=world_size, rank=rank, shuffle=False
        )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=16,
            sampler=train_sampler,
            num_workers=4,
            pin_memory=True,
        )
        self.eval_loader = DataLoader(
            self.eval_dataset,
            batch_size=16,
            sampler=eval_sampler,
            num_workers=4,
            pin_memory=True,
        )

    def init_comms(self):
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")

        if torch.distributed.is_initialized():
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = 0

        self.device_id = self.rank % torch.cuda.device_count()
        self.device = torch.device(f"cuda:{self.device_id}")

    def load_model(self, hparams=default_hparams, model_path=None):
        if model_path:
            self.model = GPT2Model(hparams)
            self.model.load_state_dict(torch.load(model_path))
        else:
            self.model = GPT2Model(hparams)

    def train_epoch(self, X, y):
        self.model.train()
        self.optimizer.zero_grad()
        y_pred = self.model(X)
        loss = self.loss_fn(y_pred, y)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train(
        self,
        dataset_path,
        epochs,
        lr=1e-4,
        weight_decay=1e-4,
        step_size=7,
        gamma=0.1,
    ):

        if not torch.distributed.is_initialized():
            self.init_comms()

        self.load_dataset(dataset_path)
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = StepLR(self.optimizer, step_size=step_size, gamma=gamma)
        self.loss_fn = nn.CrossEntropyLoss()

        if not self.model:
            self.load_model()

        self.model.to(self.device)

        for i in range(epochs):
            loss = self.train_epoch(X, y)
            print(f"Epoch {i} Loss: {loss}")

            # If global rank is 0, save the model
            if self.rank == 0:
                self.save(f"model_{i}.pt")

    def eval(self, X, y):
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(X)
            loss = self.loss_fn(y_pred, y)
        return loss.item()

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(X)
        return y_pred


if __name__ == "__main__":
    num_nodes = 2
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
            ]
        )
        .sync_secrets(["huggingface", "aws"])
    )

    cluster = rh.compute(
        name=f"rh-L4x{num_gpus_per_node}x{num_nodes}-2",
        num_nodes=num_nodes,
        instance_type=f"L4:{num_gpus_per_node}",
        provider="aws",
        image=img,
        use_spot=False,
        autostop_mins=1000,
    ).up_if_not()  # Requires access to a cloud account with the necessary permissions to launch compute.

    RemoteTrainer = rh.cls(Trainer).to(cluster, name="gpt2")
    trainer = RemoteTrainer(name="gpt2_trainer").distribute(
        "pytorch",
        num_replicas=num_nodes * num_gpus_per_node,
        replicas_per_node=num_gpus_per_node,
    )

    trainer.train("data.parquet", 10)
