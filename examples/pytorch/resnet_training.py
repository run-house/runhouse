# ## ResNet-152 Training with PyTorch Distributed
# This script demonstrates how to set up a distributed training pipeline using PyTorch, ResNet-152, and AWS S3.
# The training pipeline involves initializing a distributed model, loading data from S3, and saving model checkpoints back to S3.
# Key components include:
# - A custom ResNet-152 model class with optional pretrained weights from S3.
# - A trainer class for managing the training loop, data loading, and distributed communication.
# - Integration with Runhouse for cluster management and remote execution.

import subprocess

import boto3

import runhouse as rh

import torch
from datasets import load_from_disk
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from torchvision import models

# ### ResNet152 Model Class
# Define the ResNet-152 model class, with support for loading pretrained weights from S3.
# This is used by the trainer class to initialize the model.
class ResNet152Model(nn.Module):
    def __init__(self, num_classes=1000, pretrained=False, s3_bucket=None, s3_key=None):
        super(ResNet152Model, self).__init__()

        # Initialize the ResNet-152 model
        self.model = models.resnet152(pretrained=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

        # Load weights from S3 if specified
        if pretrained and s3_bucket and s3_key:
            self.load_weights_from_s3(s3_bucket, s3_key)

    def load_weights_from_s3(self, s3_bucket, s3_key, weights_path):
        s3 = boto3.client("s3")
        # Download the weights to a local file
        s3.download_file(s3_bucket, s3_key, weights_path)

        # Load the weights
        self.model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        print("Pretrained weights loaded from S3.")

    def forward(self, x):
        return self.model(x)

# ### Trainer Class
# The Trainer class orchestrates the distributed training process, including:
# - Initializing the distributed communication backend
# - Setting up the model, data loaders, and optimizer
# - Implementing training and validation loops
# - Saving model checkpoints to S3
# - A predict method for inference using the trainer object
class ResNet152Trainer:
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
        self, num_classes, weight_path, lr, weight_decay, step_size, gamma
    ):
        if weight_path:
            self.model = DDP(
                ResNet152Model(
                    num_classes=num_classes,
                    pretrained=True,
                    s3_bucket=self.s3_bucket,
                    s3_key=weight_path,
                ).to(self.device),
                device_ids=[self.device_id],
            )
        else:
            self.model = DDP(
                ResNet152Model(num_classes=num_classes).to(self.device),
                device_ids=[self.device_id],
            )

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = StepLR(self.optimizer, step_size=step_size, gamma=gamma)

    def load_train(self, path):
        print("Loading training data")
        subprocess.run(f"aws s3 sync {path} ~/train_dataset", shell=True)
        dataset = load_from_disk("~/train_dataset").with_format("torch")

        sampler = DistributedSampler(dataset)
        self.train_loader = DataLoader(
            dataset, batch_size=32, shuffle=False, sampler=sampler
        )

    def load_validation(self, path):
        print("Loading validation data")
        subprocess.run(f"aws s3 sync {path} ~/val_dataset", shell=True)
        dataset = load_from_disk("~/val_dataset").with_format("torch")

        sampler = DistributedSampler(dataset)
        self.val_loader = DataLoader(
            dataset, batch_size=32, shuffle=False, sampler=sampler
        )

    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        num_batches = len(self.train_loader)
        print_interval = max(
            1, num_batches // 10
        )  # Adjust this as needed, here set to every 10% of batches

        for batch_idx, batch in enumerate(self.train_loader):
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

            if (batch_idx + 1) % print_interval == 0:
                print(f"Batch {batch_idx + 1}/{num_batches}, Loss: {loss.item():.4f}")

        avg_loss = running_loss / num_batches
        return avg_loss

    def validate_epoch(self):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.train_loader):
                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = correct / total
        return accuracy

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
            val_accuracy = self.validate_epoch()
            print(f"scheduler stepping {epoch}")
            self.scheduler.step()
            print(
                f"Epoch {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}, Val Accuracy: {val_accuracy:.4f}"
            )
            # Save checkpoint every few epochs or based on validation performance
            if ((epoch + 1) % 5 == 0) and self.rank == 0:
                print("Saving checkpoint")
                self.save_checkpoint(f"resnet152_epoch_{epoch+1}.pth")

    def save_checkpoint(self, name):
        print("Saving model state")
        torch.save(self.model.state_dict(), name)
        print("Trying to put onto s3")
        s3 = boto3.client("s3")
        s3.upload_file(name, self.s3_bucket, self.s3_path + "checkpoints/" + name)
        print(f"Model saved to s3://{self.s3_bucket}/{self.s3_path}checkpoints/{name}")

    def cleanup(self):
        torch.distributed.destroy_process_group()

    def predict(self, image):
        from torchvision import transforms

        preprocess = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        image = preprocess(image).unsqueeze(0).to(self.device)

        self.model.eval()
        with torch.no_grad():
            output = self.model(image)
            _, predicted = torch.max(output, 1)
            return predicted.item()

# ### Run distributed training with Runhouse
# The following code snippet demonstrates how to create a Runhouse cluster and run the distributed training pipeline on the cluster.
# - We define a 3 node cluster with GPUs where we will do the training. 
# - Then we dispatch the trainer class to the remote cluster 
# - We create an instance of the trainer class on remote, and call .distribute('pytorch') to properly setup the distributed training. It's that easy. 
# - This remote trainer instance is accessible by name - if we construct the cluster by name, and run cluster.get('trainer') we will get the remote trainer instance. This means you can make multithreaded calls against the trainer class. 
# - The main training loop trains the model for 15 epochs and the model checkpoints are saved to S3
if __name__ == "__main__":
    train_data_path = (
        "s3://rh-demo-external/resnet-training-example/preprocessed_imagenet/train/"
    )
    val_data_path = (
        "s3://rh-demo-external/resnet-training-example/preprocessed_imagenet/test/"
    )

    working_s3_bucket = "rh-demo-external"
    working_s3_path = "resnet-training-example/"

    # Create a cluster of 3 GPUs
    gpus_per_node = 1
    num_nodes = 2

    img = rh.Image(name="pytorch").install_packages(
        [
            "torch==2.5.1 torchvision==0.20.1",
            "Pillow==11.0.0",
            "datasets",
            "boto3 awscli",
        ],
    )
    gpu_cluster = rh.cluster(
        name=f"rh-{num_nodes}x{gpus_per_node}-gpu",
        instance_type=f"A10G:{gpus_per_node}",
        num_nodes=num_nodes,
        provider="aws",
        image=img,
    ).up_if_not()
    gpu_cluster.sync_secrets(["aws"])

    epochs = 15
    remote_trainer_class = rh.module(ResNet152Trainer).to(gpu_cluster)

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
