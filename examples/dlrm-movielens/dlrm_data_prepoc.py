from typing import Any, Dict

import ray
import ray.data
import runhouse as rh
from ray.data.preprocessors import StandardScaler

# ## Preprocessing data for DLRM
# The following function is regular, undecorated Python that uses Ray Data for processing. 
# It is sent to the remote cluster for execution.
def preprocess_data(s3_read_path: str, s3_write_path: str, filename: str):
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
    ratings = load_dataset(filename)

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

    # Define an image which will be installed on each node of the cluste.r 
    # An image can include a base Docker image, package installations, setup commands, env vars, and secrets. 
    img = rh.Image('ray-data').install_packages(
        [
            "ray[data]",
            "pandas",
            "scikit-learn",
            "torch",
            "awscli",
        ]
    ).sync_secrets([
        "aws"
        ]
    )

    num_nodes = 2

    cluster = rh.cluster(
        name="rh-preprocessing",
        cpu="CPU:4+",
        memory = "15+",
        provider="aws",
        region="us-east-1",
        num_nodes=num_nodes,
        autostop_minutes=120,
        image = img, 
    ).up_if_not()

    remote_preprocess = (
        rh.function(preprocess_data)
        .to(cluster, name="preprocess_data")
        .distribute("ray")
        .save()
    )
    print("sent function to cluster")

    s3_raw = "s3://rh-demo-external/dlrm-training-example/raw_data"
    filename = "ratings.csv"
    s3_preprocessed = "s3://rh-demo-external/dlrm-training-example/preprocessed_data"

    remote_preprocess(s3_read_path=s3_raw, s3_write_path=s3_preprocessed, filename=filename)

    # cluster.teardown() # to teardown the cluster after the job is done
