import codecs
import json
import re
import sys
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import requests

from fastapi import HTTPException
from pydantic import BaseModel
from ray import cloudpickle as pickle
from ray.exceptions import RayTaskError

from runhouse.logger import ClusterLogsFormatter, logger
from runhouse.servers.obj_store import RunhouseStopIteration


class RequestContext(BaseModel):
    request_id: str
    token: Optional[str]


class ServerSettings(BaseModel):
    cluster_name: Optional[str] = None
    den_auth: Optional[bool] = None
    flush_auth_cache: Optional[bool] = None
    autostop_mins: Optional[int] = None
    status_check_interval: Optional[int] = None


class CallParams(BaseModel):
    data: Any = None
    serialization: Optional[str] = "none"
    run_name: Optional[str] = None
    stream_logs: Optional[bool] = False
    save: Optional[bool] = False
    remote: Optional[bool] = False


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
    output_type: str
    serialization: Optional[str] = None


class OutputType:
    EXCEPTION = "exception"
    STDOUT = "stdout"
    STDERR = "stderr"
    SUCCESS = "success"  # No output
    CANCELLED = "cancelled"
    RESULT_SERIALIZED = "result_serialized"
    CONFIG = "config"


class FolderMethod(str, Enum):
    GET = "get"
    LS = "ls"
    PUT = "put"
    MKDIR = "mkdir"
    RM = "rm"


class FolderParams(BaseModel):
    operation: FolderMethod
    path: Optional[str] = None
    mode: Optional[str] = None
    serialization: Optional[str] = None
    overwrite: Optional[bool] = False
    recursive: Optional[bool] = False
    contents: Optional[Any] = None


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
    elif serialization is None or serialization == "none":
        return data
    else:
        raise ValueError(f"Invalid serialization type {serialization}")


def serialize_data(data: Any, serialization: Optional[str]):
    if data is None:
        return None

    if serialization == "json":
        try:
            return json.dumps(data)
        except TypeError:
            data["error"] = str(data["error"])
            return json.dumps(data)
    elif serialization == "pickle":
        return pickle_b64(data)
    elif serialization is None or serialization == "none":
        return data
    else:
        raise ValueError(f"Invalid serialization type {serialization}")


def handle_exception_response(
    exception: Exception, traceback: str, serialization="pickle", from_http_server=False
):
    if not (
        isinstance(exception, RunhouseStopIteration)
        or isinstance(exception, StopIteration)
        or isinstance(exception, GeneratorExit)
        or isinstance(exception, StopAsyncIteration)
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
            if "Unauthorized access to resource" in detail:
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

    if isinstance(exception, RunhouseStopIteration):
        exception = StopIteration()

    exception_data = {
        "error": serialize_data(exception, serialization),
        "traceback": traceback,
        "exception_as_str": str(exception),
    }
    return Response(
        output_type=OutputType.EXCEPTION,
        data=exception_data,
        serialization=serialization,
    )


def username_from_token(token: str) -> Union[str, None]:
    """Get the username from the provided cluster subtoken."""
    from runhouse.globals import rns_client
    from runhouse.rns.utils.api import read_resp_data

    resp = requests.get(
        f"{rns_client.api_server_url}/user",
        headers={"Authorization": f"Bearer {token}"},
    )
    if resp.status_code != 200:
        return None

    user_data = read_resp_data(resp)
    username = user_data.get("username")
    if not username:
        return None

    return username


def get_token_from_request(request):
    auth_headers = auth_headers_from_request(request)
    return auth_headers.split("Bearer ")[-1] if auth_headers else None


def auth_headers_from_request(request):
    return request.headers.get("Authorization", "")


def load_current_cluster_rns_address():
    from runhouse.resources.hardware import _current_cluster, _get_cluster_from

    current_cluster = _get_cluster_from(_current_cluster("config"))
    return current_cluster.rns_address if current_cluster else None


def handle_response(
    response_data: Dict[Any, Any],
    output_type: OutputType,
    err_str: str,
    log_formatter: ClusterLogsFormatter,
):
    system_color, reset_color = log_formatter.format(output_type)

    if output_type == OutputType.RESULT_SERIALIZED:
        return deserialize_data(response_data["data"], response_data["serialization"])
    elif output_type == OutputType.CONFIG:
        # No need to unpickle since this was just sent as json
        return response_data["data"]
    elif output_type == OutputType.CANCELLED:
        raise RuntimeError(f"{err_str}: task was cancelled")
    elif output_type == OutputType.SUCCESS:
        return
    elif output_type == OutputType.EXCEPTION:
        # Here for compatibility before we stopped serializing the traceback
        if not isinstance(response_data["data"], dict):
            exception_dict = deserialize_data(
                response_data["data"], response_data["serialization"]
            )
            fn_exception = exception_dict["error"]
            fn_traceback = exception_dict["traceback"]
        else:
            fn_traceback = response_data["data"]["traceback"]
            fn_exception = None
            fn_exception_as_str = response_data["data"].get("exception_as_str", None)
            try:
                fn_exception = deserialize_data(
                    response_data["data"]["error"], response_data["serialization"]
                )
            except Exception as e:
                logger.error(
                    f"{system_color}{err_str}: Failed to unpickle exception. Please check the logs for more "
                    f"information.{reset_color}"
                )
                if fn_exception_as_str:
                    logger.error(
                        f"{system_color}{err_str} Exception as string: {fn_exception_as_str}{reset_color}"
                    )
                logger.error(f"{system_color}Traceback: {fn_traceback}{reset_color}")
                raise e
        if not (
            isinstance(fn_exception, StopIteration)
            or isinstance(fn_exception, GeneratorExit)
            or isinstance(fn_exception, StopAsyncIteration)
        ):
            logger.error(f"{system_color}{err_str}: {fn_exception}{reset_color}")
            logger.error(f"{system_color}Traceback: {fn_traceback}{reset_color}")
        raise fn_exception
    elif output_type == OutputType.STDOUT:
        res = response_data["data"]
        # Regex to match tqdm progress bars
        tqdm_regex = re.compile(r"(.+)%\|(.+)\|\s+(.+)/(.+)")
        for line in res:
            if tqdm_regex.match(line):
                # tqdm lines are always preceded by a \n, so we can use \x1b[1A to move the cursor up one line
                # For some reason, doesn't work in PyCharm's console, but works in the terminal
                print(
                    f"{system_color}\x1b[1A\r" + line + reset_color, end="", flush=True
                )
            else:
                print(system_color + line + reset_color, end="", flush=True)
    elif output_type == OutputType.STDERR:
        res = response_data["data"]
        print(system_color + res + reset_color, file=sys.stderr)


###########################
#### Folder Operations ####
###########################
def folder_mkdir(path: Path):
    if not path.parent.is_dir():
        raise ValueError(
            f"Parent path {path.parent} does not exist or is not a directory"
        )

    path.mkdir(parents=True, exist_ok=True)

    return Response(output_type=OutputType.SUCCESS)


def folder_get(path: Path, folder_params: FolderParams):
    mode = folder_params.mode or "rb"
    serialization = folder_params.serialization
    binary_mode = "b" in mode

    with open(path, mode=mode) as f:
        file_contents = f.read()

    if binary_mode and isinstance(file_contents, bytes):
        file_contents = file_contents.decode()

    output_type = OutputType.RESULT_SERIALIZED
    serialization = serialization or ("pickle" if binary_mode else None)

    return Response(
        data=file_contents,
        output_type=output_type,
        serialization=serialization,
    )


def folder_put(path: Path, folder_params: FolderParams):
    overwrite = folder_params.overwrite
    mode = folder_params.mode or "wb"
    serialization = folder_params.serialization
    contents = folder_params.contents

    if not path.is_dir():
        raise ValueError(f"Path {path} is not a directory")

    path.mkdir(exist_ok=True)

    if overwrite is False:
        existing_files = {str(item.name) for item in path.iterdir()}
        intersection = existing_files.intersection(set(contents.keys()))
        if intersection:
            raise FileExistsError(
                f"File(s) {intersection} already exist(s) at path: {path}"
            )

    for filename, file_obj in contents.items():
        binary_mode = "b" in mode
        if binary_mode:
            serialization = serialization or "pickle"

        file_obj = serialize_data(file_obj, serialization)

        if binary_mode and not isinstance(file_obj, bytes):
            file_obj = file_obj.encode()

        file_path = path / filename
        if not overwrite and file_path.exists():
            raise FileExistsError(f"File {file_path} already exists.")

        try:
            with open(file_path, mode) as f:
                f.write(file_obj)
        except Exception as e:
            raise e

    return Response(output_type=OutputType.SUCCESS)


def folder_ls(path: Path, folder_params: FolderParams):
    recursive: bool = folder_params.recursive
    if path is None:
        raise ValueError("Path is required for ls operation")

    if not path.exists():
        raise FileNotFoundError(f"Path {path} does not exist")

    if not path.is_dir():
        raise ValueError(f"Path {path} is not a directory")

    files = list(path.rglob("*")) if recursive else [item for item in path.iterdir()]
    return Response(
        data=files,
        output_type=OutputType.RESULT_SERIALIZED,
        serialization=None,
    )


def folder_rm(path: Path, folder_params: FolderParams):
    import shutil

    recursive: bool = folder_params.recursive
    contents = folder_params.contents
    if contents:
        for content in contents:
            content_path = path / content
            if content_path.exists():
                if content_path.is_file():
                    content_path.unlink()
                elif content_path.is_dir() and recursive:
                    shutil.rmtree(content_path)
                else:
                    raise ValueError(
                        f"Path {content_path} is a directory and recursive is set to False"
                    )
        return Response(output_type=OutputType.SUCCESS)

    if not path.is_dir():
        path.unlink()
        return Response(output_type=OutputType.SUCCESS)

    if recursive:
        shutil.rmtree(path)
        return Response(output_type=OutputType.SUCCESS)

    items = path.iterdir()
    if not items:
        # Empty dir
        path.rmdir()
        return Response(output_type=OutputType.SUCCESS)

    for item in items:
        if item.is_file():
            item.unlink()
        else:
            raise ValueError(
                f"Folder {item} found in {path}, recursive is set to False"
            )

    return Response(output_type=OutputType.SUCCESS)
