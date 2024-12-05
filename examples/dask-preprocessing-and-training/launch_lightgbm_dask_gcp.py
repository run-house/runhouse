import runhouse as rh
from runhouse.resources.images.image import Image

# ## Dask + LightGBM Training
if __name__ == "__main__":
    # ## Create a Runhouse cluster with 2 nodes
    num_nodes = 3
    cluster_name = f"rh-{num_nodes}-dask-gcp"

    # The environment for the remote cluster
    img = rh.Image("dask-env").install_packages(
            [
                "dask[distributed,dataframe]",
                "dask-ml",
                "gcsfs",
                "lightgbm",
                "bokeh",
            ],
        ).set_env_vars(
            {
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
            }
        )

    cluster = rh.ondemand_cluster(
        name=cluster_name,
        instance_type="n2-highmem-4",
        num_nodes=num_nodes,
        provider="gcp",
        region="us-east1",
        image=img,
        launcher_type="den",
    ).up_if_not()

    # cluster.restart_server(resync_rh=True)

    # ## Setup the remote training
    # LightGBMModelTrainer is a completely normal class that contains our training methods,
    # that a researcher would also be able to use locally as-is as well (on non-distributed Dask)
    from lightgbm_training import LightGBMModelTrainer

    remote_dask_trainer = rh.module(LightGBMModelTrainer).to(cluster)

    # Create is a locally callable, but remote instance of the trainer class
    # You can interact with this trainer class in a different notebook / elsewhere using
    # cluster.get('trainer', remote = True) to get the remote object
    # We also use .distribute("dask") to start the Dask cluster and indicate this will be used with Dask
    dask_trainer = remote_dask_trainer(name="my_trainer").distribute("dask")

    # ## Do the processing and training on the remote cluster
    # Access the Dask client, data, and preprocess the data
    data_path = "gs://rh-demo-external/output_parquet" # 2024 NYC Taxi Data
    X_vars = ["passenger_count", "trip_distance", "fare_amount"]
    y_var = "tip_amount"
    cluster.connect_dask()
    dask_trainer.load_client()
    dask_trainer.load_data(data_path)
    new_date_columns = dask_trainer.preprocess(date_column="tpep_pickup_datetime")
    X_vars = X_vars + new_date_columns
    dask_trainer.train_test_split(target_var=y_var, features=X_vars)

    # Train, test, and save the model
    dask_trainer.train_model()
    print("Model trained")
    dask_trainer.test_model()
    print("Model tested")
    dask_trainer.save_model("gs://rh-model-checkpoints/lightgbm_dask/model.pkl")
    print("Model saved")

    # cluster.teardown() # Optionally, automatically teardown the cluster after training
    
    print("Launching notebook")
    # cluster.notebook()  # Optionally, open a Jupyter notebook on the cluster to interact with the trained model
