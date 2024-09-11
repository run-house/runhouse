import os

import boto3


# Download data from S3
def download_folder_from_s3(bucket_name, s3_folder_prefix, local_folder_path):
    s3 = boto3.client("s3")

    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket_name, Prefix=s3_folder_prefix):
        if "Contents" in page:
            for obj in page["Contents"]:
                s3_key = obj["Key"]
                relative_path = os.path.relpath(s3_key, s3_folder_prefix)
                local_path = os.path.join(local_folder_path, relative_path)

                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                s3.download_file(bucket_name, s3_key, local_path)
                print(f"Downloaded {s3_key} to {local_path}")


# download_folder_from_s3('rh-demo-external', 'your/s3/folder/prefix', '/path/to/local/folder', 'your-access-key-id', 'your-secret-access-key')


# Upload data to S3 bucket
def upload_folder_to_s3(local_folder_path, bucket_name, s3_folder_prefix):
    s3 = boto3.client("s3")

    for root, dirs, files in os.walk(local_folder_path):
        for file in files:
            local_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_path, local_folder_path)
            s3_path = os.path.join(s3_folder_prefix, relative_path)

            s3.upload_file(local_path, bucket_name, s3_path)
            print(f"Uploaded {local_path} to s3://{bucket_name}/{s3_path}")


# upload_folder_to_s3('/path/to/local/folder', 'rh-demo-external', 'your/s3/folder/prefix', 'your-access-key-id', 'your-secret-access-key')
