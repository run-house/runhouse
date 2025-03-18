import kubetorch as kt

# ## Using Dask on Ray for data processing
# Dask on Ray works out of the box when KT sets up the Ray cluster; simply use enable_dask_on_ray()
def read_taxi_df_dask(dataset_path, X_vars, y_vars):
    import dask.dataframe as dd
    from ray.util.dask import disable_dask_on_ray, enable_dask_on_ray

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


if __name__ == "__main__":
    num_nodes = 3

    img = kt.images.ray().pip_install(
        [
            "dask-ml",
            "dask[distributed]",
            "dask[dataframe]",
            "boto3",
            "s3fs",
            "xgboost",
        ]
    )

    cluster = kt.compute(cpus="4+", image=img)

    cluster.sync_secrets(["aws"])

    # ## Example of using Dask on Ray to read data and minimally preprocess the data
    # Use one slice of the NYC taxi data as an example
    dataset_path = "s3://rh-demo-external/taxi/yellow_tripdata_2024-01.parquet"
    X_vars = ["passenger_count", "trip_distance", "fare_amount"]
    y_var = ["tip_amount"]

    remote_read_taxi_df_dask = (
        kt.function(read_taxi_df_dask)
        .to(cluster)
        .distribute("ray", num_nodes=num_nodes)
    )
    remote_read_taxi_df_dask(dataset_path, X_vars, y_var)
