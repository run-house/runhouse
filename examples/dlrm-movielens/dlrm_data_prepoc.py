import os
from typing import Any, Dict

import numpy as np

import ray
import ray.data
import runhouse as rh
import torch
from ray.data.preprocessors import StandardScaler


def preprocess_data(s3_read_path: str, s3_write_path: str):
    """
    Preprocess MovieLens dataset using Ray Data for distributed processing.

    Args:
        s3_path (str): Path to the s3 directory containing CSV files.
        s3_write_path (str): Path to save the processed dataset in S3.
    """
    # Load datasets using Ray Data
    def load_dataset(filename: str):
        print("Reading file:", f"{s3_read_path}/{filename}")
        return ray.data.read_csv(f"{s3_read_path}/{filename}")

    # Load datasets
    ratings = load_dataset("ratings.csv")

    # Preprocess data for DLRM model
    print("Preprocessing data for DLRM")
    ratings = StandardScaler(columns=["rating"]).fit_transform(ratings)

    # Split the dataset into train, eval, and test sets
    train_ds, remaining_ds = ratings.train_test_split(
        test_size=0.3, shuffle=True, seed=42
    )
    eval_ds, test_ds = remaining_ds.train_test_split(
        test_size=0.5, shuffle=True, seed=42
    )
    print(train_ds.schema())

    # Save processed data to S3
    def write_to_s3(ds, s3_path):
        print("Processing data", s3_path)
        ds.write_parquet(s3_path)
        print(f"Processed data saved to {s3_path}")

    print("Writing datasets to S3")
    datasets = {"train": train_ds, "eval": eval_ds, "test": test_ds}

    for dataset_name, dataset in datasets.items():
        print(dataset_name)
        print(dataset)
        s3_path = f"{s3_write_path}/{dataset_name}/processed_movielens_data.parquet"
        write_to_s3(dataset, s3_path)

    print("Preprocessing complete")


if __name__ == "__main__":
    num_nodes = 2

    cluster = rh.cluster(
        name="rh-preprocessing",
        instance_type="CPU:4+",
        # memory = "32+",
        provider="aws",
        region="us-east-1",
        num_nodes=num_nodes,
        autostop_minutes=45,
        default_env=rh.env(
            reqs=["scikit-learn", "pandas", "ray[data]", "awscli", "torch"],
        ),
    ).up_if_not()

    # cluster.restart_server()
    cluster.sync_secrets(["aws"])

    s3_raw = "s3://rh-demo-external/dlrm-training-example/raw_data"
    local_path = "~/dlrm"
    s3_preprocessed = "s3://rh-demo-external/dlrm-training-example/preprocessed_data"

    # remote_download = rh.function(download_data).to(cluster)
    # remote_download(s3_path = s3_raw, local_path = local_path)

    remote_preprocess = (
        rh.function(preprocess_data)
        .to(cluster, name="preprocess_data")
        .distribute("ray")
    )
    print("sent function to cluster")
    remote_preprocess(s3_read_path=s3_raw, s3_write_path=s3_preprocessed)

    cluster.teardown()
