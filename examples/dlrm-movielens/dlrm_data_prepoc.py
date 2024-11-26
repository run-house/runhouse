import runhouse as rh 


import ray
import ray.data

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from typing import Dict, Any
import subprocess 

def download_data(s3_path: str, local_path: str):
    import subprocess  
    subprocess.run(f"aws s3 sync {s3_path} {local_path}", shell=True)


def preprocess_data(local_path: str, s3_write_path: str):
    """
    Preprocess MovieLens dataset using Ray Data for distributed processing.
    
    Args:
        local_path (str): Local path to the directory containing CSV files.
        s3_write_path (str): Path to save the processed dataset in S3.
    """
    # Load datasets using Ray Data
    def load_dataset(filename: str):
        print('Reading file:', filename)
        return ray.data.read_csv(f"{local_path}/{filename}")

    # Load datasets
    print('Loading datasets')
    ratings = load_dataset("ratings.csv")
    print("Schema:", ratings.schema())

    # Preprocess data for DLRM model
    print('Preprocessing data for DLRM')
    def normalize_ratings(batch):
        min_rating = batch["rating"].min()
        max_rating = batch["rating"].max()
        batch["rating"] = (batch["rating"] - min_rating) / (max_rating - min_rating)
        return batch
    
    ratings = ratings.map_batches(normalize_ratings, batch_format="pandas")
    train_ds, remaining_ds = ratings.train_test_split(test_size=0.3, shuffle=True, seed=42)
    eval_ds, test_ds = remaining_ds.train_test_split(test_size=0.5, shuffle=True, seed=42)

    def prepare_features_labels(batch):
        import torch
        # Prepare dense features as (userId, movieId) and labels as normalized ratings
        dense_features = torch.tensor(list(zip(batch["userId"], batch["movieId"])), dtype=torch.float32)
        labels = torch.tensor(batch["rating"].values, dtype=torch.float32)
        return {"dense_features": dense_features, "labels": labels}

    train_ds = train_ds.map_batches(prepare_features_labels, batch_format="pandas")
    eval_ds = eval_ds.map_batches(prepare_features_labels, batch_format="pandas")
    test_ds = test_ds.map_batches(prepare_features_labels, batch_format="pandas")

    def write_to_s3(ds, s3_path):
        ds.write_parquet(s3_path)   
        print(f"Processed data saved to {s3_path}")
    
    # Save processed data to S3
    datasets = {'train': train_ds
                , 'eval': eval_ds
                , 'test': test_ds}

    for dataset_name, dataset in datasets.items():
        s3_path = f"{s3_write_path}/{dataset_name}/processed_movielens_data.parquet"
        write_to_s3(dataset, s3_path)

    print('Preprocessing complete')



if __name__ == "__main__":
    num_nodes = 2
    
    cluster = rh.cluster(
        name="rh-preprocessing",
        instance_type="CPU:4+",
        #memory = "32+",
        provider="aws",
        region="us-east-1",
        num_nodes = num_nodes,
        default_env=rh.env(
            reqs=[
                "scikit-learn",
                "pandas",
                "ray[data]",
                "awscli",
            ],
        ),
    ).up_if_not()
    
    cluster.sync_secrets(["aws"])   

    cluster.restart_server()
    remote_download = rh.function(download_data).to(cluster)
    remote_download(s3_path = 's3://rh-demo-external/dlrm-training-example/raw_data'
                    , local_path = '~/dlrm')

    remote_preprocess = rh.function(preprocess_data).to(cluster, name = 'preprocess_data').distribute('ray')
    print('sent function to cluster')
    remote_preprocess(local_path= '~/dlrm'
                      , s3_write_path= 's3://rh-demo-external/dlrm-training-example/preprocessed_data/')