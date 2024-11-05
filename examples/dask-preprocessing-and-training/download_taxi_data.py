import runhouse as rh


def remote_download_to_s3(year, month, s3_bucket="nyc-tlc", s3_path="trip-data/"):
    import boto3
    import requests

    filename = f"yellow_tripdata_{month}-{year}.parquet"
    url = f"https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{month}-{year}.parquet"

    response = requests.get(url)
    response.raise_for_status()  # Raise an error for bad responses

    # Step 2: Save the file locally
    with open(filename, "wb") as local_file:
        local_file.write(response.content)

    s3 = boto3.client("s3")
    s3.upload_file(filename, s3_bucket, s3_path + filename)

    import os

    if os.path.exists(filename):
        os.remove(filename)


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

    remote_download = rh.function(remote_download_to_s3).to(cluster)
    remote_download("2024", "01")
