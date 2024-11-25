# ## 
# This script demonstrates how to set up a distributed training pipeline using PyTorch, DLRM, MovieLens, and AWS S3.
# The training pipeline involves initializing a distributed model, loading data from S3, and saving model checkpoints back to S3.
# Key components include:
# - 
# - 
# - 

import subprocess

import boto3

import runhouse as rh

import os
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
import torch.optim as optim


# ### DLRM Model Class
# Define the DLRM model class, with support for loading pretrained weights from S3.
# This is used by the trainer class to initialize the model.
class DLRM(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim):
        super(DLRM, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, users, items):
        user_embed = self.user_embedding(users)
        item_embed = self.item_embedding(items)
        x = torch.cat([user_embed, item_embed], dim=-1)
        return self.fc(x).squeeze(1)


# ### Trainer Class
# The Trainer class orchestrates the distributed training process, including:
# - Initializing the distributed communication backend
# - Setting up the model, data loaders, and optimizer
# - Implementing training and validation loops
# - Saving model checkpoints to S3
# - A predict method for inference using the trainer object
class DLRMTrainer:
    def __init__(self, s3_bucket, s3_path):
        self.rank = None
        self.device_id = None
        self.device = None
        self.model = None

        self.criterion = None
        self.optimizer = None
        self.scheduler = None

        self.train_loader = None
        self.val_loader = None

        self.s3_bucket = s3_bucket
        self.s3_path = s3_path

        print("Remote class initialized")

    def init_comms(self):
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")

        if torch.distributed.is_initialized():
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = 0

        self.device_id = self.rank % torch.cuda.device_count()
        self.device = torch.device(f"cuda:{self.device_id}")

    def init_model(
        self, num_users, num_items, embedding_dim, lr, weight_decay, step_size, gamma
    ):
        
        self.model = DDP(
            DLRM(
                num_users=num_users,
                num_items=num_items,
                embedding_dim=embedding_dim,
            ).to(self.device),
            device_ids=[self.device_id],
        )
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = StepLR(self.optimizer, step_size=step_size, gamma=gamma)

    def load_train(self, path, batch_size=32):
        print("Loading training data")
        subprocess.run(f"aws s3 sync {path} ~/train_dataset", shell=True)
        dataset = load_from_disk("~/train_dataset").with_format("torch")

        sampler = DistributedSampler(dataset)
        self.train_loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, sampler=sampler
        )

    def load_validation(self, path, batch_size=32):
        print("Loading validation data")
        subprocess.run(f"aws s3 sync {path} ~/val_dataset", shell=True)
        dataset = load_from_disk("~/val_dataset").with_format("torch")

        sampler = DistributedSampler(dataset)
        self.val_loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, sampler=sampler
        )

    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        num_batches = len(self.train_loader)
        print_interval = max(
            1, num_batches // 10
        )  # Adjust this as needed, here set to every 10% of batches

        batch_idx = 0

        for users, items, ratings in self.train_loader:
            users, items, ratings = users.cuda(self.rank), items.cuda(self.rank), ratings.cuda(self.rank)
            self.optimizer.zero_grad()
            outputs = self.model(users, items)
            loss = self.criterion(outputs, ratings)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

            if (batch_idx + 1) % print_interval == 0:
                print(f"Batch {batch_idx + 1}/{num_batches}, Loss: {loss.item():.4f}")

            batch_idx += 1

        avg_loss = running_loss / num_batches
        return avg_loss

    def validate_epoch(self):
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for users, items, ratings in self.val_loader:
                users, items, ratings = users.cuda(self.rank), items.cuda(self.rank), ratings.cuda(self.rank)
                outputs = self.model(users, items)
                loss = self.criterion(outputs, ratings)
                val_loss += loss.item()

        val_loss /= len(self.val_loader)
        
        return val_loss

    def train(
        self,
        num_classes,
        num_epochs,
        train_data_path,
        val_data_path,
        lr=1e-4,
        weight_decay=1e-4,
        step_size=7,
        gamma=0.1,
        weights_path=None,
    ):
        self.init_comms()
        print("Remote comms initialized")
        self.init_model(
            num_classes, weights_path, lr, weight_decay, step_size, gamma
        )
        print("Model initialized")

        # Load training and validation data
        self.load_train(train_data_path)
        self.load_validation(val_data_path)
        print("Data loaded")

        # Train the model
        for epoch in range(num_epochs):
            print(f"entering epoch {epoch}")
            train_loss = self.train_epoch()
            print(f"validating {epoch}")
            if self.rank == 0:
                val_accuracy = self.validate_epoch()

            print(f"scheduler stepping {epoch}")

            self.scheduler.step()
            print(
                f"Epoch {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}, Val Accuracy: {val_accuracy:.4f}"
            )

            # Save checkpoint every few epochs or based on validation performance
            if ((epoch + 1) % 5 == 0) and self.rank == 0:
                print("Saving checkpoint")
                self.save_checkpoint(f"dlrm_epoch_{epoch+1}.pth")

    def save_checkpoint(self, name):
        print("Saving model state")
        torch.save(self.model.state_dict(), name)
        print("Trying to put onto s3")
        s3 = boto3.client("s3")
        s3.upload_file(name, self.s3_bucket, self.s3_path + "checkpoints/" + name)
        print(f"Model saved to s3://{self.s3_bucket}/{self.s3_path}checkpoints/{name}")

    def cleanup(self):
        torch.distributed.destroy_process_group()

    def predict(self, user, item):
        
        self.model.eval()
        with torch.no_grad():
            output = self.model(user,item)
            return output

# ### Run distributed training with Runhouse
# The following code snippet demonstrates how to create a Runhouse cluster and run the distributed training pipeline on the cluster.
# - We define a 3 node cluster with GPUs where we will do the training. 
# - Then we dispatch the trainer class to the remote cluster 
# - We create an instance of the trainer class on remote, and call .distribute('pytorch') to properly setup the distributed training. It's that easy. 
# - This remote trainer instance is accessible by name - if we construct the cluster by name, and run cluster.get('trainer') we will get the remote trainer instance. This means you can make multithreaded calls against the trainer class. 
# - The main training loop trains the model for 15 epochs and the model checkpoints are saved to S3
if __name__ == "__main__":
    train_data_path = (
        "s3://rh-demo-external/dlrm-training-example/preprocessed_imagenet/train/"
    )
    val_data_path = (
        "s3://rh-demo-external/dlrm-training-example/preprocessed_imagenet/test/"
    )

    working_s3_bucket = "rh-demo-external"
    working_s3_path = "dlrm-training-example/"

    # Create a cluster of 3 GPUs
    gpus_per_node = 1
    num_nodes = 3

    gpu_cluster = (
        rh.cluster(
            name=f"rh-{num_nodes}x{gpus_per_node}GPU",
            instance_type=f"A10G:{gpus_per_node}",
            num_nodes=num_nodes,
            provider="aws",
            default_env=rh.env(
                name="pytorch_env",
                reqs=[
                    "torch==2.5.1",
                    "datasets",
                    "boto3",
                    "awscli",
                ],
            ),
        )
        .up_if_not()
        .save()
    )
    #gpu_cluster.restart_server(resync_rh=True)
    gpu_cluster.sync_secrets(["aws"])

    epochs = 15
    remote_trainer_class = rh.module(DLRMTrainer).to(gpu_cluster)

    remote_trainer = remote_trainer_class(
        name="trainer", s3_bucket=working_s3_bucket, s3_path=working_s3_path
    ).distribute(
        distribution="pytorch",
        replicas_per_node=gpus_per_node,
        num_replicas=gpus_per_node * num_nodes,
    )

    remote_trainer.train(
        num_epochs=epochs,
        num_classes=1000,
        train_data_path=train_data_path,
        val_data_path=val_data_path,
    )
