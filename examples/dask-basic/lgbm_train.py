import kubetorch as kt


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


def xgboost_dask(dataset_path, X_vars, y_vars):
    import xgboost as xgb
    from dask import dataframe as dd
    from dask.distributed import Client

    client = Client("tcp://localhost:8786")
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
    img = kt.images.dask().pip_install(
        [
            "dask-ml",
            "boto3",
            "s3fs",
            "xgboost",
        ],
    )
    compute = kt.Compute(num_cpus="8", memory="32", image=img).sync_secrets(["aws"])

    # ## Example of using Dask on Ray to read data and minimally preprocess the data
    # Use one slice of the NYC taxi data as an example
    dataset_path = "s3://rh-demo-external/taxi/yellow_tripdata_2024-01.parquet"
    X_vars = ["passenger_count", "trip_distance", "fare_amount"]
    y_vars = ["tip_amount"]

    remote_read_taxi_df_dask = (
        kt.function(read_taxi_df_dask)
        .to(compute)
        .distribute("dask", num_nodes=num_nodes)
    )
    remote_read_taxi_df_dask(dataset_path, X_vars, y_vars)
    print("***** done with Dask preproc example *****")

    remote_xgb = (
        kt.function(xgboost_dask).to(compute).distribute("dask", num_nodes=num_nodes)
    )
    remote_xgb(dataset_path, X_vars, y_vars)
    print("***** Done with XGB Training *****")
