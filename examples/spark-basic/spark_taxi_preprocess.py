def spark_test(spark):
    df = spark.createDataFrame([("Alice", 1), ("Bob", 2)], ["name", "id"])
    df.show()


def clean_up(data):
    from pyspark.sql.functions import col

    # Replace any null or missing values in numeric columns with default values (example: 0 or mean values)
    data = data.fillna(
        {
            "extra": 0.0,
            "tip_amount": 0.0,
            "tolls_amount": 0.0,
            "improvement_surcharge": 0.0,
            "congestion_surcharge": 0.0,
            "Airport_fee": 0.0,
        }
    )

    # Remove rows where essential columns are missing (e.g., no pickup or dropoff times)
    data = data.filter(
        col("tpep_pickup_datetime").isNotNull()
        & col("tpep_dropoff_datetime").isNotNull()
    )

    return data


def add_features(data):
    from pyspark.sql.functions import col, hour, unix_timestamp

    data = data.withColumn(
        "trip_duration_minutes",
        (
            unix_timestamp("tpep_dropoff_datetime")
            - unix_timestamp("tpep_pickup_datetime")
        )
        / 60,
    )
    data = data.withColumn(
        "trip_speed", col("trip_distance") / col("trip_duration_minutes")
    )
    data = data.withColumn(
        "rush_hour_trip",
        (hour(col("tpep_pickup_datetime")) >= 7)
        & (hour(col("tpep_pickup_datetime")) <= 9),
    )

    return data


def nyc_taxi_preprocess(spark, data_path):
    import os

    data = spark.read.parquet(os.path.expanduser(data_path))

    print("Rows", data.count())

    data = clean_up(data)
    data = add_features(data)

    print(data.head())

    data.write.mode("overwrite").parquet(
        os.path.expanduser("~/nyc_taxi_preprocessed.parquet")
    )


if __name__ == "__main__":
    import runhouse as rh

    num_nodes = 2
    cluster_name = f"rh-{num_nodes}-aws-spark"

    # The environment for the remote cluster
    img = (
        rh.Image()
        .install_packages(
            ["raydp", "pyspark"],
        )
        .from_docker("openjdk:17-slim")
    )

    cluster = rh.cluster(
        name=cluster_name,
        num_cpus="6+",
        memory="24+",
        num_nodes=num_nodes,
        provider="aws",
        use_spot=False,
        autostop_mins=120,
        image=img,
        launcher="local",
    ).up_if_not()

    local_path = "~/yellow_tripdata_2024-12.parquet"
    cluster.run_bash(
        f"wget -P {local_path} https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-12.parquet",
        node="all",
    )

    remote_spark_test = (
        rh.function(spark_test).to(cluster).distribute(distribution="spark")
    )
    remote_spark_test(
        spark_init_options={
            "num_executors": 1,
            "executor_cores": 2,
            "executor_memory": "2GB",
        }
    )

    remote_spark = (
        rh.function(nyc_taxi_preprocess)
        .to(cluster)
        .distribute(
            distribution="spark",
            spark_init_options={
                "num_executors": 4,
                "executor_cores": 2,
                "executor_memory": "2GB",
            },
        )
    )
    remote_spark(data_path=local_path)
