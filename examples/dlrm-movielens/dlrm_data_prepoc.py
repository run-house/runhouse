import kubetorch as kt
import ray
import ray.data
from ray.data.preprocessors import StandardScaler

# ## Preprocessing data for DLRM
# The following function until line 35 is regular, undecorated Python that uses Ray Data for processing.
# It is sent to the remote compute for execution.
def preprocess_data(s3_read_path, s3_write_path, filename):
    # Load datasets using Ray Data
    ratings = ray.data.read_csv(f"{s3_read_path}/{filename}")

    # Preprocess data - standard scaler as an example
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
        print("Writing: ", s3_path)
        ds.write_parquet(s3_path)
        print(f"Processed data saved to {s3_path}")

    datasets = {"train": train_ds, "eval": eval_ds, "test": test_ds}

    for dataset_name, dataset in datasets.items():
        s3_path = f"{s3_write_path}/{dataset_name}/processed_movielens_data.parquet"
        write_to_s3(dataset, s3_path)


# ## Launch compute and execute
# Here, we launch compute, dispatch the preprocessing function to the compute, and call that function.
# Whether launching elastic compute or from Kubernetes, Runhouse wires up the Ray cluster for you and downs the cluster when complete.
# Runhouse syncs the code the code across and makes it a callable "service" on the remote.
# This code can be identically placed within an orchestrator (e.g. my_pipeline.yaml) and identical execution will occur.
if __name__ == "__main__":

    # Define an image which will be installed on each node of compute.
    # An image can include a base Docker image, package installations, setup commands, env vars, and secrets.
    img = (
        kt.images.pytorch()
        .pip_install(
            [
                "ray[data]",
                "pandas",
                "scikit-learn",
                "awscli",
            ]
        )
        .sync_secrets(["aws"])
    )

    # Create Kubetorch compute with 8 CPUs and 32GB of memory
    # Launch from AWS (EC2) on US East 1 region
    compute = kt.Compute(
        num_cpus="8",
        memory="32",  # Also `gpus` `disk_size`
        image=img,
    )

    # Send the preprocess_data function to the remote compute and distribute it with Ray over 4 nodes
    remote_preprocess = (
        kt.function(preprocess_data)
        .to(compute)
        .distribute(
            "ray",
            num_nodes=4,
        )
    )

    # Call the remote function (which uses Ray Data on the Ray cluster we formed)
    s3_raw = "s3://rh-demo-external/dlrm-training-example/raw_data"
    filename = "ratings.csv"
    s3_preprocessed = "s3://rh-demo-external/dlrm-training-example/preprocessed_data"

    remote_preprocess(
        s3_read_path=s3_raw, s3_write_path=s3_preprocessed, filename=filename
    )
