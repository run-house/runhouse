import runhouse as rh
from helper_dask import launch_dask_cluster

if __name__ == "__main__":
    # ## Create a Runhouse cluster with 3 nodes
    num_nodes = 3
    cluster_name = f"py-{num_nodes}"

    cluster = rh.cluster(
        name=cluster_name,
        instance_type="r5d.xlarge",
        num_instances=num_nodes,
        provider="aws",
    ).up_if_not()

    # ## Launch a Dask cluster on the Runhouse cluster
    # You can do this here if you want the Dask cluster to be ephemeral, or in a separate notebook/script if long lived
    launch_dask_cluster(cluster)

    # ## Setup the remote training
    from lightgbm_training import LightGBMModelTrainer

    # The environment for the remote cluster
    env = rh.env(
        reqs=[
            "dask-ml",
            "dask[distributed]",
            "dask[dataframe]",
            "boto3",
            "s3fs",
            "lightgbm",
            "cloudpickle",
        ],
        secrets=["aws"],
    )

    # ## Send the trainer class to the remote cluster and instantiate a remote object named 'my_trainer'
    # You can interact with this trainer class in a different notebook / elsewhere using cluster.get('trainer', remote = True) to get the remote object
    remote_dask_trainer = rh.module(LightGBMModelTrainer).to(cluster, env=env)
    dask_trainer = remote_dask_trainer(
        name="my_trainer"
    )  # This is a locally callable, but remote instance of the trainer class

    # ## Do the processing and training on the remote cluster
    # Access the Dask client, data, and preprocess the data
    dataset_path = "s3://rh-demo-external/taxi/*.parquet"  # 2024 NYC Taxi Data
    X_vars = ["passenger_count", "trip_distance", "fare_amount"]
    y_var = "tip_amount"

    dask_trainer.load_client()
    dask_trainer.load_data(dataset_path)
    new_date_columns = dask_trainer.preprocess(date_column="tpep_pickup_datetime")
    X_vars = X_vars + new_date_columns
    dask_trainer.train_test_split(target_var=y_var, features=X_vars)

    # Train, test, and save the model
    dask_trainer.train_model()
    dask_trainer.test_model()
    dask_trainer.save_model("model.pkl")

    # cluster.teardown() # Optionally, automatically teardown the cluster after training
