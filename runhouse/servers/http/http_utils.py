import codecs
import json
import logging
import re
import sys
from typing import Any, Dict, List, Optional

from fastapi import HTTPException
from pydantic import BaseModel
from ray import cloudpickle as pickle
from ray.exceptions import RayTaskError

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


class RequestContext(BaseModel):
    request_id: str
    username: Optional[
        str
    ]  # TODO refactor the auth cache to use usernames instead of token hashes
    token_hash: Optional[str]


class ServerSettings(BaseModel):
    den_auth: Optional[bool] = None
    flush_auth_cache: Optional[bool] = None


class CallParams(BaseModel):
    data: Any = None
    serialization: Optional[str] = None
    run_name: Optional[str] = None
    stream_logs: Optional[bool] = False
    save: Optional[bool] = False
    remote: Optional[bool] = False
    run_async: Optional[bool] = False


class PutResourceParams(BaseModel):
    serialized_data: Any
    serialization: Optional[str] = None
    env_name: Optional[str] = None


class PutObjectParams(BaseModel):
    key: str
    serialized_data: Any
    serialization: Optional[str] = None
    env_name: Optional[str] = None


class GetObjectParams(BaseModel):
    key: str
    serialization: Optional[str] = None
    remote: Optional[bool] = False


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
    elif serialization is None:
        return data
    else:
        raise ValueError(f"Invalid serialization type {serialization}")


def serialize_data(data: Any, serialization: Optional[str]):
    if data is None:
        return None

    if serialization == "json":
        return json.dumps(data)
    elif serialization == "pickle":
        return pickle_b64(data)
    elif serialization is None:
        return data
    else:
        raise ValueError(f"Invalid serialization type {serialization}")


def handle_exception_response(
    exception: Exception, traceback, serialization="pickle", from_http_server=False
):
    if not (
        isinstance(exception, StopIteration) or isinstance(exception, GeneratorExit)
    ):
        logger.exception(exception)

    # We need to be selective about when we convert errors to HTTPExceptions because HTTPExceptions are not
    # picklable and can't be passed over the actor boundary. We should only be converting in the HTTPServer right
    # before we send the response back to the client from the http_server.
    if isinstance(exception, RayTaskError) and from_http_server:
        cause = exception.args[0]
        detail = cause.args[0] if cause.args else ""
        if isinstance(cause, PermissionError):
            if "No Runhouse token provided." in detail:
                raise HTTPException(status_code=401, detail=detail)
            if (
                "Unauthorized access to resource" in detail
            ):
                raise HTTPException(status_code=403, detail=detail)
        if isinstance(cause, ValueError):
            if "Invalid serialization type" in detail:
                raise HTTPException(status_code=400, detail=detail)

    if isinstance(exception, PermissionError):
        # If we're raising from inside the obj store, we should still raise this error so the HTTPServer can
        # catch it and convert it to an HTTPException.
        raise exception

    if serialization not in ["json", "pickle"] or isinstance(exception, HTTPException):
        # If the exception is an HTTPException, we should let it flow back through the server so it gets
        # handled by FastAPI's exception handling and returned to the user with the correct status code
        raise exception

    return Response(
        output_type=OutputType.EXCEPTION,
        error=serialize_data(exception, serialization=serialization),
        traceback=serialize_data(traceback, serialization=serialization),
    )


def get_token_from_request(request):
    auth_headers = request.headers.get("Authorization", "")
    return auth_headers.split("Bearer ")[-1] if auth_headers else None


def load_current_cluster():
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
        if not (
            isinstance(fn_exception, StopIteration)
            or isinstance(fn_exception, GeneratorExit)
        ):
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
