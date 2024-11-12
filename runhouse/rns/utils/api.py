import ast
import datetime
import functools
import json
import os
import uuid
from enum import Enum
from typing import Tuple

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
    except (TypeError, OverflowError):
        return False
    return True


def generate_uuid():
    return uuid.uuid4().hex


def utc_now():
    """Current time as datetime object."""
    return datetime.datetime.now(datetime.timezone.utc)


def log_timestamp():
    """Return as timestamp in nanoseconds."""
    return int(utc_now().timestamp() * 1e9)


def log_datetime():
    """Current time as readable datetime string.
    Example: '2023-04-23'"""
    return utc_now().strftime("%Y-%m-%d")


def resolve_absolute_path(path: str):
    return os.path.abspath(os.path.expanduser(path))


def relative_file_path(file_path: str):
    """Convert to a relative path if it is not already one."""
    if file_path.startswith("~"):
        return file_path

    # Convert to a relative path
    relative_path = os.path.relpath(file_path, os.path.expanduser("~"))
    relative_path = relative_path.replace("\\", "/")

    if not relative_path.startswith("~"):
        relative_path = f"~/{relative_path}"

    return relative_path


def generate_ssh_keys() -> Tuple[str, str]:
    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    # Adapted from: https://github.com/skypilot-org/skypilot/blob/2c7419cdd22adb440bd37702be6f1ddc1c287684/sky/authentication.py#L106 # noqa
    private_key_path = os.path.expanduser("~/.ssh/sky-key")
    public_key_path = os.path.expanduser("~/.ssh/sky-key.pub")

    if os.path.exists(private_key_path):
        return private_key_path, public_key_path

    key = rsa.generate_private_key(
        backend=default_backend(), public_exponent=65537, key_size=2048
    )

    private_key = (
        key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),
        )
        .decode("utf-8")
        .strip()
    )

    public_key = (
        key.public_key()
        .public_bytes(
            serialization.Encoding.OpenSSH, serialization.PublicFormat.OpenSSH
        )
        .decode("utf-8")
        .strip()
    )

    key_dir = os.path.dirname(private_key_path)
    os.makedirs(key_dir, exist_ok=True, mode=0o700)

    with open(
        private_key_path,
        "w",
        encoding="utf-8",
        opener=functools.partial(os.open, mode=0o600),
    ) as f:
        f.write(private_key)

    with open(
        public_key_path,
        "w",
        encoding="utf-8",
        opener=functools.partial(os.open, mode=0o644),
    ) as f:
        f.write(public_key)

    return private_key_path, public_key_path


class ResourceAccess(str, Enum):
    WRITE = "write"
    READ = "read"
    DENIED = "denied"


class ResourceVisibility(str, Enum):
    PRIVATE = "private"
    UNLISTED = "unlisted"
    PUBLIC = "public"
