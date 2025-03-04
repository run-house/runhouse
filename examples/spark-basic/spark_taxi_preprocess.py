# # Launch Spark Workloads on Runhouse
# Here, we offer minimal examples of launching Spark workloads on Runhouse. With Runhouse, you can
# launch ephemeral compute that is ready to run Spark workloads like a serverless offering, but
# with the flexibility to choose any cloud provider and compute type, plus use spot instances.

# ## Hello World Spark Example
def spark_test(spark):
    df = spark.createDataFrame([("Alice", 1), ("Bob", 2)], ["name", "id"])
    df.show()


# ## NYC Taxi Preprocessing Example
# We define two helper functions `clean_up` and `add_features` to clean up the data and add new features to the dataset.
# Then we define the function `nyc_taxi_preprocess` that reads the data from a parquet file, cleans it up, adds features, and writes the preprocessed data back to a parquet file.
# That function is sent to a Runhouse cluster, and we call `.distribute("spark")` to make the functionr eady to run.
def clean_up(data):
    from pyspark.sql.functions import col

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


# This is the function we send to Runhouse compute to preprocess; note that we require an
# argument `spark` to be passed to the function. Runhouse will provide the Spark client to the function.
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


# ## Defining the Runhouse Cluster and Running the Spark Workloads
# We define a Runhouse cluster with multiple nodes and launch it on AWS. You can see some of the options you have
# to initialize the cluster. You can also run this on your own Kubernetes cluster with a Kubeconfig as well.
# Currently, Runhouse does require the pre-installation of RayDP and Java in the image. This may change in the future.
if __name__ == "__main__":

    import runhouse as rh

    num_nodes = 2
    cluster_name = f"rh-{num_nodes}-aws-spark"

    img = (
        rh.Image()
        .install_packages(
            ["raydp", "pyspark"],
        )
        .from_docker("openjdk:17-slim")
    )  # To distribute workloads with Runhouse, we need `raydp` and Java installed. PySpark, and any other packages our Spark job needs should also be included

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

    remote_spark_test = (
        rh.function(spark_test).to(cluster).distribute(distribution="spark")
    )
    remote_spark_test(
        spark_init_options={
            "num_executors": 1,
            "executor_cores": 2,
            "executor_memory": "2GB",
        }
    )  # We can define the number of executors, cores, and memory for the Spark job when calling the function

    local_path_on_remote_compute = "~/yellow_tripdata_2024-12.parquet"
    data_url = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-12.parquet"
    cluster.run_bash_over_ssh(
        f"wget -P {local_path_on_remote_compute} {data_url}", node="all"
    )  # Download the dataset

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
    )  # We can also define the Spark resources when sending the function to the cluster. This is overriden if we define the resources when calling the function.
    remote_spark(data_path=local_path_on_remote_compute)

    cluster.teardown()  # To teardown the cluster after the run, comment out to keep cluster alive
