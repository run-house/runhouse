import ast
import datetime
import json
import uuid

from requests import Response


def timing(func):
    def wrapper(*args, **kwargs):
        import time

        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"Finished {func.__name__.title()} in {int((end - start))} seconds")
        return result

    return wrapper


def remove_null_values_from_dict(source_dic: dict) -> dict:
    return {k: v for k, v in source_dic.items() if v is not None}


def load_resp_content(resp: Response) -> dict:
    return json.loads(resp.content)


def read_resp_data(resp: Response):
    return load_resp_content(resp).get("data", {})


def to_bool(value):
    try:
        return ast.literal_eval(value)
    except:
        return value


def is_jsonable(myjson):
    try:
        json.dumps(myjson)
    except ValueError:
        return False
    return True


def generate_uuid():
    return uuid.uuid4().hex


def log_timestamp():
    return int(datetime.datetime.now(datetime.timezone.utc).timestamp() * 1e9)


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
