# ## 
# This script demonstrates how to set up a distributed training pipeline using PyTorch, DLRM, MovieLens, and AWS S3.
# The training pipeline involves initializing a distributed model, loading data from S3, and saving model checkpoints back to S3.
import boto3

import runhouse as rh

import os
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

import logging
import ray 
import ray.train
import ray.train.torch
from ray.train import ScalingConfig



# ### DLRM Model Class
# Define the DLRM model class, with support for loading pretrained weights from S3.
# This is used by the trainer class to initialize the model.
class DLRM(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim, load_from_s3=False, s3_bucket=None, s3_key=None):
        super(DLRM, self).__init__()
        
        self.user_embedding = nn.Embedding(num_users, embedding_dim,padding_idx=0)
        self.item_embedding = nn.Embedding(num_items, embedding_dim,padding_idx=0)
        
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        self.num_users = num_users
        self.num_items = num_items

        if load_from_s3:
            self.load_model(s3_bucket, s3_key)
    
    def forward(self, users, items):
        users = users.clamp(min=0, max=self.num_users - 1)
        items = items.clamp(min=0, max=self.num_items - 1)

        user_embed = self.user_embedding(users)
        item_embed = self.item_embedding(items)

        x = torch.cat([user_embed, item_embed], dim=-1) 
        return self.fc(x).squeeze(1)
    
    def load_model(self, s3_bucket, s3_key): 
        from io import BytesIO
        s3 = boto3.client("s3")
        response = s3.get_object(Bucket=s3_bucket, Key=s3_key)
        model_state_dict = torch.load(BytesIO(response['Body'].read()))  # Load the state_dict from S3
        self.load_state_dict(model_state_dict)

# ### Reads the dataset that was preprocessed in the prior step using Ray Data
# This is a simple example so we will train with just userId, movieId, and rating columns.
def read_preprocessed_dlrm(data_path):
        import pyarrow as pa
        schema = pa.schema([
            ("userId", pa.int64()),
            ("movieId", pa.int64()),
            ("rating", pa.float32()),
            ])
        return ray.data.read_parquet(data_path, schema=schema, columns=["userId", "movieId","rating"])


# ### Trainer Function
# This function is the main training that gets distributed and run by Ray
# - Sets up the model, data loaders, and optimizer
# - Implementing training and validation loops
# - Saving model checkpoints to S3
def dlrm_train(config):
    
    logging.info('Starting dlrm training')

    s3_bucket, s3_path, unique_users, unique_movies, embedding_dim, lr, weight_decay, step_size, gamma, epochs, save_every_epochs = config["s3_bucket"], config["s3_path"], config["unique_users"], config["unique_movies"], config["embedding_dim"], config["lr"], config["weight_decay"], config["step_size"], config["gamma"], config["epochs"], config["save_every_epochs"]
    
    # Initialize model, criterion, optimizer, and scheduler
    model = ray.train.torch.prepare_model(DLRM(
            num_users=unique_users,
            num_items=unique_movies,
            embedding_dim=embedding_dim,
        )
    )
    
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    # Define per epoch training function 
    def train_epoch(batch_size = 128):
    
        model.train()
        running_loss = 0.0
        train_data_shard = ray.train.get_dataset_shard("train")
    
        for batch_idx, batch in enumerate(train_data_shard.iter_torch_batches(batch_size=batch_size)):
            optimizer.zero_grad()
            
            user_ids, item_ids, labels = batch['userId'], batch['movieId'], batch['rating']
            outputs = model(user_ids, item_ids)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if (batch_idx + 1) % 250 == 0:
                print(f"Batch {batch_idx + 1} loss: {loss.item():.4f}")
    
        avg_loss = running_loss / batch_idx
        return avg_loss
    
    # Define validation function 
    def validate_epoch(batch_size):
        model.eval()
        val_loss = 0.0
        eval_data_shard = ray.train.get_dataset_shard("val")
    
        with torch.no_grad():
            for batch_idx, batch in enumerate(eval_data_shard.iter_torch_batches(batch_size=batch_size)):
                user_ids, item_ids, labels = batch['userId'], batch['movieId'], batch['rating']
                outputs = model(user_ids, item_ids)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
    
        val_loss /= batch_idx
        return val_loss
    
    # Define a helper function to save checkpoints to s3
    def save_checkpoint(name):
        torch.save(model.module.state_dict(), name)
    
        s3 = boto3.client("s3")
        s3.upload_file(name, s3_bucket, s3_path + "checkpoints/" + name)
        print(f"Model saved to s3://{s3_bucket}/{s3_path}checkpoints/{name}")
    
    # Run the training for `epochs` epochs, saving every fifth epoch
    save_checkpoint(f"dlrm_model.pth")
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}")
    
        train_loss = train_epoch(batch_size = 500)
        val_loss = validate_epoch(batch_size = 500)
        
        print(f"Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}")
        
        scheduler.step()
        
        if (epoch + 1) % save_every_epochs == 0:
            save_checkpoint(f"dlrm_model_epoch_{epoch + 1}.pth")

        save_checkpoint(f"dlrm_model.pth")

def ray_trainer(num_nodes, gpus_per_node, s3_bucket, s3_path, train_data_path, val_data_path, embedding_dim, lr, weight_decay, step_size, gamma, epochs, save_every_epochs): 
    # Load data 
    logging.info("Loading data")

    train_dataset = read_preprocessed_dlrm(train_data_path)
    val_dataset = read_preprocessed_dlrm(val_data_path)
    
    logging.info("Data loaded")
    
    unique_users = 330975 #len(train_dataset.unique("userId"))
    unique_movies = 86000 #len(train_dataset.unique("movieId")) 
    
    scaling_config = ScalingConfig(
        num_workers=num_nodes, 
        use_gpu=True,
        resources_per_worker = {
            "CPU": 3,  
            "GPU": gpus_per_node   
        },)

    ray_train = ray.train.torch.TorchTrainer(
            train_loop_per_worker=dlrm_train,
            train_loop_config= { "s3_bucket": s3_bucket, "s3_path": s3_path, "unique_users": unique_users, "unique_movies": unique_movies, "embedding_dim": embedding_dim, "lr": lr, "weight_decay": weight_decay, "step_size": step_size, "gamma": gamma, "epochs": epochs, "save_every_epochs": save_every_epochs},
            datasets={"train": train_dataset, "val": val_dataset},
            scaling_config=scaling_config,
        ) 
    
    ray_train.fit()
    
# ### Run distributed training with Runhouse
# The following code snippet demonstrates how to create a Runhouse cluster and run the distributed training pipeline on the cluster.
# - We define a 3 node cluster with GPUs where we will do the training. 
# - Then we dispatch the Ray trainer function to the remote cluster and call .distribute('ray') to properly setup Ray. It's that easy. 
if __name__ == "__main__":
    train_data_path = "s3://rh-demo-external/dlrm-training-example/preprocessed_data/train/"
    val_data_path ="s3://rh-demo-external/dlrm-training-example/preprocessed_data/eval/"

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
                    "ray[data,train]"
                ],
            ),
        )
        .up_if_not()
        .save()
    )
    gpu_cluster.restart_server()
    gpu_cluster.sync_secrets(["aws"])

    epochs = 15
    remote_trainer = rh.function(ray_trainer).to(gpu_cluster, name = "ray_trainer").distribute('ray')
    
    remote_trainer(
        num_nodes, 
        gpus_per_node,
        s3_bucket=working_s3_bucket,
        s3_path=working_s3_path,
        train_data_path=train_data_path, 
        val_data_path=val_data_path, 
        embedding_dim=64,
        lr=0.001,
        weight_decay=0.0001,
        step_size=5,
        gamma=0.5,
        epochs=epochs,
        save_every_epochs=5)
