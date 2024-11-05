import runhouse as rh

# ## Using Dask on Ray for data processing
# Dask on Ray works out of the box with Runhouse clusters. Simply use enable_dask_on_ray()
def read_taxi_df_dask(dataset_path, X_vars, y_vars):
    import dask.dataframe as dd
    from ray.util.dask import disable_dask_on_ray, enable_dask_on_ray  # , ray_dask_get

    enable_dask_on_ray()

    # Read the dataset
    df = dd.read_parquet(dataset_path)
    print(df.head())

    X = df[X_vars].to_dask_array(lengths=True)
    y = df[y_vars].to_dask_array(lengths=True)

    from dask_ml.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("First few rows of X_train:")
    print(
        X_train[:5].compute()
    )  # Limit to first 5 rows and compute to bring it to memory

    disable_dask_on_ray()


# ## Using Dask-XGBoost to train a model on a Dask cluster
def xgboost_dask(dataset_path, X_vars, y_vars):
    import xgboost as xgb
    from dask import dataframe as dd
    from dask.distributed import Client

    client = Client("tcp://localhost:8786")  # Dask and Runhouse share a head node

    df = dd.read_parquet(dataset_path)

    X = df[X_vars].to_dask_array(lengths=True)
    y = df[y_vars].to_dask_array(lengths=True)

    dtrain = xgb.dask.DaskDMatrix(client, X, y)

    output = xgb.dask.train(
        client,
        {"verbosity": 2, "tree_method": "hist", "objective": "reg:squarederror"},
        dtrain,
        num_boost_round=4,
        evals=[(dtrain, "train")],
    )

    print(output)


# ## Launch a Dask cluster on the Runhouse cluster
# We start a scheduler on the head node and workers on the other nodes
def launch_dask_cluster(cluster):
    import threading

    def start_scheduler():
        cluster.run("dask scheduler --port 8786")

    def start_worker(head_node_ip, node):
        cluster.run(f"dask worker tcp://{head_node_ip}:8786", node=node)

    stable_ips = [ip[0] for ip in cluster.stable_internal_external_ips]
    head_node_ip = stable_ips[0]

    # Start the scheduler on the head node
    scheduler_thread = threading.Thread(target=start_scheduler)
    scheduler_thread.daemon = True
    scheduler_thread.start()

    # Start workers and connect to the scheduler
    import time

    for ip in cluster.ips:
        time.sleep(3)
        cluster.run('pip install "dask[distributed]"', node=ip)
        worker_thread = threading.Thread(target=start_worker, args=(head_node_ip, ip))
        worker_thread.daemon = True
        worker_thread.start()


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

    # cluster.restart_server()

    # ## Example of using Dask on Ray to read data and minimally preprocess the data
    # Use one slice of the NYC taxi data as an example
    dataset_path = "s3://rh-demo-external/taxi/yellow_tripdata_2024-01.parquet"
    X_vars = ["passenger_count", "trip_distance", "fare_amount"]
    y_vars = ["tip_amount"]

    env = rh.env(
        name="dask-env",
        reqs=[
            "dask-ml",
            "dask[distributed]",
            "dask[dataframe]",
            "boto3",
            "s3fs",
            "xgboost",
        ],
        secrets=["aws"],
    )
    remote_read_taxi_df_dask = rh.function(read_taxi_df_dask).to(cluster, env=env)
    remote_read_taxi_df_dask(dataset_path, X_vars, y_vars)
    print("***** done with Dask on Ray example *****")

    # ## Example of starting a Dask cluster on the Runhouse cluster to make use of Dask-XGBoost

    # Use cluster commands to launch the Dask cluster from the Runhouse cluster
    launch_dask_cluster(cluster)
    print("***** Dask cluster launched *****")

    # Send XGB Training to the remote cluster and run it there
    env = rh.env(
        reqs=[
            "dask-ml",
            "dask[distributed]",
            "dask[dataframe]",
            "boto3",
            "s3fs",
            "xgboost",
        ],
        secrets=["aws"],
    )
    remote_xgb = rh.function(xgboost_dask).to(cluster, env=env)
    remote_xgb(dataset_path, X_vars, y_vars)
    print("***** Done with XGB Training *****")

    # This will keep the Dask cluster up until the user interrupts the program (otherwise, it will terminate at the end of the script)
    try:
        while True:
            pass
    except KeyboardInterrupt:
        print("Tore down Dask cluster")
