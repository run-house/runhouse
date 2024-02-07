import codecs
import json
import logging
import re
import sys
from typing import Any, Dict, List, Optional

from pydantic import BaseModel
from ray import cloudpickle as pickle

logger = logging.getLogger(__name__)


class Message(BaseModel):
    data: Any = None
    serialization: str = None
    env: str = None
    key: Optional[str] = None
    stream_logs: Optional[bool] = True
    save: Optional[bool] = False
    remote: Optional[bool] = False
    run_async: Optional[bool] = False


class ServerSettings(BaseModel):
    den_auth: Optional[bool] = None
    flush_auth_cache: Optional[bool] = None


class PutResourceParams(BaseModel):
    serialized_data: Any
    serialization: Optional[str] = None
    env_name: Optional[str] = None


class PutObjectParams(BaseModel):
    key: str
    serialized_data: Any
    serialization: Optional[str] = None
    env_name: Optional[str] = None


class RenameObjectParams(BaseModel):
    key: str
    new_key: str


class DeleteObjectParams(BaseModel):
    keys: List[str]


class Args(BaseModel):
    args: Optional[List[Any]]
    kwargs: Optional[Dict[str, Any]]


class Response(BaseModel):
    data: Any = None
    error: Optional[str] = None
    traceback: Optional[str] = None
    output_type: str
    serialization: Optional[str] = None


class OutputType:
    EXCEPTION = "exception"
    STDOUT = "stdout"
    STDERR = "stderr"
    SUCCESS = "success"  # No output
    NOT_FOUND = "not_found"
    CANCELLED = "cancelled"
    RESULT = "result"
    RESULT_LIST = "result_list"
    RESULT_STREAM = "result_stream"
    RESULT_SERIALIZED = "result_serialized"
    SUCCESS_STREAM = "success_stream"  # No output, but with generators
    CONFIG = "config"


def pickle_b64(picklable):
    return codecs.encode(pickle.dumps(picklable), "base64").decode()


def b64_unpickle(b64_pickled):
    return pickle.loads(codecs.decode(b64_pickled.encode(), "base64"))


def deserialize_data(data: Any, serialization: Optional[str]):
    if data is None:
        return None

    if serialization == "json":
        return json.loads(data)
    elif serialization == "pickle":
        return b64_unpickle(data)
    else:
        return data


def serialize_data(data: Any, serialization: Optional[str]):
    if data is None:
        return None

    if serialization == "json":
        return json.dumps(data)
    elif serialization == "pickle":
        return pickle_b64(data)
    else:
        return data


def handle_exception_response(exception, traceback):
    logger.exception(exception)
    return Response(
        output_type=OutputType.EXCEPTION,
        error=pickle_b64(exception),
        traceback=pickle_b64(traceback),
    )


def get_token_from_request(request):
    auth_headers = request.headers.get("Authorization", "")
    return auth_headers.split("Bearer ")[-1] if auth_headers else None


def load_current_cluster_rns_address():
    from runhouse.resources.hardware import _current_cluster, _get_cluster_from

    current_cluster = _get_cluster_from(_current_cluster("config"))
    return current_cluster.rns_address if current_cluster else None


def handle_response(response_data, output_type, err_str):
    if output_type == OutputType.RESULT_SERIALIZED:
        return deserialize_data(response_data["data"], response_data["serialization"])
    if output_type in [OutputType.RESULT, OutputType.RESULT_STREAM]:
        return b64_unpickle(response_data["data"])
    elif output_type == OutputType.CONFIG:
        # No need to unpickle since this was just sent as json
        return response_data["data"]
    elif output_type == OutputType.RESULT_LIST:
        # Map, starmap, and repeat return lists of results
        return [b64_unpickle(val) for val in response_data["data"]]
    elif output_type == OutputType.NOT_FOUND:
        raise KeyError(f"{err_str}: key {response_data['data']} not found")
    elif output_type == OutputType.CANCELLED:
        raise RuntimeError(f"{err_str}: task was cancelled")
    elif output_type in [OutputType.SUCCESS, OutputType.SUCCESS_STREAM]:
        return
    elif output_type == OutputType.EXCEPTION:
        fn_exception = b64_unpickle(response_data["error"])
        fn_traceback = b64_unpickle(response_data["traceback"])
        logger.error(f"{err_str}: {fn_exception}")
        logger.error(f"Traceback: {fn_traceback}")
        raise fn_exception
    elif output_type == OutputType.STDOUT:
        res = response_data["data"]
        # Regex to match tqdm progress bars
        tqdm_regex = re.compile(r"(.+)%\|(.+)\|\s+(.+)/(.+)")
        for line in res:
            if tqdm_regex.match(line):
                # tqdm lines are always preceded by a \n, so we can use \x1b[1A to move the cursor up one line
                # For some reason, doesn't work in PyCharm's console, but works in the terminal
                print("\x1b[1A\r" + line, end="", flush=True)
            else:
                print(line, end="", flush=True)
    elif output_type == OutputType.STDERR:
        res = response_data["data"]
        print(res, file=sys.stderr)
