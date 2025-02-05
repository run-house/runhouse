# # Distributed ResNet Training with PyTorch Lightning
#
# In this example, we begin with regular Lightning code, defining the Lightning and data modules, and a Trainer class that encapsulates the training routine.
# Runhouse does not need changes or alterations to standard Lightning training code in order to distribute and run it, but rather is designed
# to work with your existing codebase and standard training routines.
#
# Then we have the script's main, which will launch the Runhouse cluster, dispatch the Lightning class to the remote compute,
# wire up distribution, and run the training.
import subprocess

import boto3
import lightning as L
import runhouse as rh
import torch
import torch.nn as nn
import torch.optim as optim

from datasets import load_from_disk
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import models

# ## ResNet152 Lightning Module and Data Module
# The ResNet152 Lightning Module is a standard PyTorch Lightning module that defines the ResNet152 model, the training and validation steps,
# and the optimizer and scheduler. The Data Module is a standard PyTorch Lightning data module that defines the training and validation dataloaders.
class ResNet152LitModule(L.LightningModule):
    def __init__(
        self,
        num_classes=1000,
        pretrained=False,
        s3_bucket=None,
        s3_key=None,
        weights_path=None,
        lr=1e-3,
        weight_decay=1e-4,
        step_size=10,
        gamma=0.1,
    ):
        super(ResNet152LitModule, self).__init__()
        self.save_hyperparameters(ignore=["s3_bucket", "s3_key"])

        self.model = models.resnet152(pretrained=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

        if pretrained and s3_bucket and s3_key:
            self.load_weights_from_s3(s3_bucket, s3_key, weights_path)

        self.criterion = nn.CrossEntropyLoss()

    def load_weights_from_s3(self, s3_bucket, s3_key, weights_path):
        s3 = boto3.client("s3")
        s3.download_file(s3_bucket, s3_key, weights_path)
        self.model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        print("Pretrained weights loaded from S3.")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch):
        images, labels = batch["image"].to(self.device), batch["label"].to(self.device)
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch):
        images, labels = batch["image"].to(self.device), batch["label"].to(self.device)
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
        acc = (preds == labels).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=self.hparams.step_size, gamma=self.hparams.gamma
        )
        return [optimizer], [scheduler]

    def save_weights_to_s3(self, s3_bucket, s3_key):
        torch.save(self.model.state_dict(), "resnet152.pth")
        s3 = boto3.client("s3")
        s3.upload_file("resnet152.pth", s3_bucket, s3_key)
        print("Weights saved to S3.")

    def teardown(self, stage=None):
        print(f"Teardown stage: {stage}")
        super().teardown(stage)


class ImageNetDataModule(L.LightningDataModule):
    def __init__(self, train_data_path, val_data_path, batch_size):
        super().__init__()
        print("init for imagenet")
        self.train_data_path = train_data_path
        self.val_data_path = val_data_path
        self.batch_size = batch_size

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            subprocess.run(
                f"aws s3 sync {self.train_data_path} ~/train_dataset", shell=True
            )
            self.train_dataset = load_from_disk("~/train_dataset").with_format("torch")
            subprocess.run(
                f"aws s3 sync {self.val_data_path} ~/val_dataset", shell=True
            )
            self.val_dataset = load_from_disk("~/val_dataset").with_format("torch")

    def train_dataloader(self):
        sampler = DistributedSampler(self.train_dataset)
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, sampler=sampler
        )

    def val_dataloader(self):
        sampler = DistributedSampler(self.train_dataset)
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, sampler=sampler
        )


# ## Encapsulation of Training
# We briefly encapsulate the training routine in a class that loads the data, model, and trainer, and fits the model. We also provide a method to save the model weights to S3.
# We will send this class to the remote compute in the main and create a remote instance of this class to run the training.
class ResNetTrainer:
    def __init__(self, num_nodes, gpus_per_node, working_s3_bucket, working_s3_path):
        self.num_nodes = num_nodes
        self.gpus_per_node = gpus_per_node

        self.working_s3_bucket = working_s3_bucket
        self.working_s3_path = working_s3_path

        self.data_module = None
        self.train_loader = None
        self.val_loader = None

        self.lit_module = None
        self.trainer = None

    def load_data(self, train_data_path, val_data_path, batch_size=32):
        self.data_module = ImageNetDataModule(
            train_data_path=train_data_path,
            val_data_path=val_data_path,
            batch_size=batch_size,
        )

    def load_model(
        self,
        num_classes,
        pretrained=False,
        s3_bucket=None,
        s3_key=None,
        weights_path=None,
    ):
        self.lit_module = ResNet152LitModule(
            num_classes=num_classes,
            pretrained=pretrained,
            s3_bucket=s3_bucket,
            s3_key=s3_key,
            weights_path=weights_path,
        )

    def load_trainer(self, epochs, strategy):
        self.trainer = L.Trainer(
            max_epochs=epochs,
            devices=self.gpus_per_node,
            num_nodes=self.num_nodes,
            strategy=strategy,
            logger=True,
            log_every_n_steps=1,
            accelerator="gpu",
        )

    def fit(self):
        if self.train_loader is None and self.data_module is None:
            raise ValueError(
                "Data module not loaded. Please call load_data() before calling fit()."
            )
        if self.lit_module is None:
            raise ValueError(
                "Lightning module not loaded. PLease call load_model() before calling fit()."
            )
        if self.trainer is None:
            raise ValueError(
                "Trainer not loaded. Please call load_trainer() before calling fit()."
            )

        import torch.distributed as dist

        dist.init_process_group(backend="nccl", init_method="env://")
        self.trainer.fit(self.lit_module, self.data_module)

    def save(self):
        if self.lit_module is None:
            raise ValueError(
                "Lightning module not loaded. Please call load_model() before calling save()."
            )

        self.lit_module.save_weights_to_s3(self.working_s3_bucket, self.working_s3_path)


# ## Launch Compute and Run the Training
# We will now launch Runhouse compute with multiple nodes and use it to train the ResNet with Lightning.
# The data we use here is a sampled, preprocessed set of images from the ImageNet dataset. You can
# see the preprocessing script at https://github.com/run-house/runhouse/blob/main/examples/pytorch-resnet/imagenet_preproc.py
# :::note{.info title="Note"}
# Make sure that your code runs within a `if __name__ == "__main__":` block, as shown below. Otherwise,
# the script code will run when Runhouse attempts to run code remotely.
# :::
if __name__ == "__main__":
    train_data_path = (
        "s3://rh-demo-external/resnet-training-example/preprocessed_imagenet/train/"
    )
    val_data_path = (
        "s3://rh-demo-external/resnet-training-example/preprocessed_imagenet/test/"
    )

    working_s3_bucket = "rh-demo-external"
    working_s3_path = "resnet-training-example/"

    # We will launch a 3 node cluster with 1 GPU per node, and define the image to use for the cluster
    # which here is a set of packages to install, but optionally could depend on a custom AMI, Docker image,
    # include env vars, further bash commands, etc.
    gpus_per_node = 1
    num_nodes = 3

    img = rh.Image().install_packages(
        [
            "torch==2.5.1",
            "torchvision==0.20.1",
            "Pillow==11.0.0",
            "datasets",
            "boto3",
            "awscli",
            "lightning",
            "runhouse==0.0.36",
        ]
    )

    gpus = (
        rh.compute(
            name=f"rh-{num_nodes}x{gpus_per_node}GPU",
            gpus=f"A10G:{gpus_per_node}",
            num_nodes=num_nodes,
            provider="aws",
            image=img,
        )
        .up_if_not()
        .save()
    )

    # gpus.restart_server() # to restart the Runhouse server, does not tear down the actual underlying compute
    gpus.sync_secrets(["aws"])  # sends our AWS secret to the remote compute

    # Now that the cluster is up, we will send our trainer to the remote compute, instantiate a remote instance of it, and run the training.
    # Note that we call .distribute("pytorch") to set up the PyTorch distributed backend. We run the training for 15 epochs with a batch size of 32,
    # calling methods on the remote instance as if it were local.
    trainer = rh.module(ResNetTrainer).to(gpus)
    epochs = 15
    batch_size = 32

    remote_trainer = trainer(
        name="resnet_trainer",
        num_nodes=num_nodes,
        gpus_per_node=gpus_per_node,
        working_s3_bucket=working_s3_bucket,
        working_s3_path=working_s3_path,
    ).distribute("pytorch")

    remote_trainer.load_data(
        train_data_path=train_data_path,
        val_data_path=val_data_path,
        batch_size=batch_size,
    )
    remote_trainer.load_model(num_classes=1000, pretrained=False)
    remote_trainer.load_trainer(epochs=epochs, strategy="ddp")
    remote_trainer.fit()
