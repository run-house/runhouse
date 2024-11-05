import runhouse as rh


# ## Using Dask on Ray for data processing
# Dask on Ray works out of the box with Runhouse clusters. Simply use enable_dask_on_ray()
def read_taxi_df_dask(dataset_path, X_vars, y_vars):
    import dask.dataframe as dd

    # Read the dataset
    df = dd.read_parquet(dataset_path)
    print("First few rows of the dataset:")
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


# ## Using Dask-XGBoost to train a model on a Dask cluster
def xgboost_dask(dataset_path, X_vars, y_vars):
    import xgboost as xgb

    from dask import dataframe as dd

    # Alternatively, we could use dask's get_client
    client = rh.here.connect_dask()
    print(f"Client info: {client}")

    print("Reading the dataset")
    df = dd.read_parquet(dataset_path)

    X = df[X_vars].to_dask_array(lengths=True)
    y = df[y_vars].to_dask_array(lengths=True)

    dtrain = xgb.dask.DaskDMatrix(client, X, y)

    print("Training the model")
    output = xgb.dask.train(
        client,
        {"verbosity": 2, "tree_method": "hist", "objective": "reg:squarederror"},
        dtrain,
        num_boost_round=4,
        evals=[(dtrain, "train")],
    )

    print(output)


if __name__ == "__main__":
    # ## Create a Runhouse cluster with 3 nodes
    num_nodes = 3
    env = rh.env(
        name="dask-env",
        load_from_den=False,
        reqs=[
            "dask-ml",
            "dask[distributed]",
            "dask[dataframe]",
            "boto3",
            "s3fs",
            "xgboost",
        ],
    )
    cluster = (
        rh.cluster(
            name=f"rh-{num_nodes}",
            instance_type="r5d.xlarge",
            num_instances=num_nodes,
            provider="aws",
            default_env=env,
            launcher_type="local",
        )
        .save()
        .up_if_not()
    )
    cluster.sync_secrets(["aws"])

    # ## Example of using Dask on Ray to read data and minimally preprocess the data
    # Use one slice of the NYC taxi data as an example
    dataset_path = "s3://rh-demo-external/taxi/yellow_tripdata_2024-01.parquet"
    X_vars = ["passenger_count", "trip_distance", "fare_amount"]
    y_vars = ["tip_amount"]

    remote_read_taxi_df_dask = (
        rh.function(read_taxi_df_dask).to(cluster).distribute("dask")
    )
    remote_read_taxi_df_dask(dataset_path, X_vars, y_vars)
    print("***** done with Dask preproc example *****")

    remote_xgb = rh.function(xgboost_dask).to(cluster).distribute("dask")
    remote_xgb(dataset_path, X_vars, y_vars)
    print("***** Done with XGB Training *****")
