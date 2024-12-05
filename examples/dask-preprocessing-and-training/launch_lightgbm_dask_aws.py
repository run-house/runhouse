import runhouse as rh

if __name__ == "__main__":
    # ## Create a Runhouse cluster with multiple nodes
    num_nodes = 2
    cluster_name = f"py-new-{num_nodes}"

    img = (
        rh.Image("dask-env")
        .install_packages(
            [
                "dask[distributed,dataframe]",
                "dask-ml",
                "gcsfs",
                "lightgbm",
            ],
        )
        .set_env_vars(
            {
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
            }
        )
    )

    cluster = rh.cluster(
        name=cluster_name,
        instance_type="r5d.xlarge",
        num_nodes=num_nodes,
        provider="aws",
        launcher_type="local",
        image = img, 
    ).up_if_not()

    # ## Send the trainer class to the remote cluster and instantiate a remote object named 'my_trainer'
    # LightGBMModelTrainer is a completely normal class encapsulating training, that a researcher would also be able to use locally as-is
    from lightgbm_training import LightGBMModelTrainer
    remote_dask_trainer = rh.module(LightGBMModelTrainer).to(cluster)

    # Create is a locally callable, but remote instance of the trainer class
    # You can interact with this trainer class in a different notebook / elsewhere using
    # cluster.get('trainer', remote = True) to get the remote object
    dask_trainer = remote_dask_trainer(name="my_trainer").distribute("dask")

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

    cluster.teardown()  # Optionally, automatically teardown the cluster after training
