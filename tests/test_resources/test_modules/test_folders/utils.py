def create_s3_bucket(bucket_name: str):
    """Create bucket in S3 if it does not already exist."""
    from sky.data.storage import S3Store

    s3_store = S3Store(name=bucket_name, source="")
    return s3_store


def create_gcs_bucket(bucket_name: str):
    """Create bucket in GS if it does not already exist."""
    from sky.data.storage import GcsStore

    gcs_store = GcsStore(name=bucket_name, source="")
    return gcs_store
