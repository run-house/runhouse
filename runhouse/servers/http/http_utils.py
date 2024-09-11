import codecs
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import requests

from fastapi import HTTPException
from pydantic import BaseModel, field_validator
from ray import cloudpickle as pickle
from ray.exceptions import RayTaskError

from runhouse.logger import get_logger

from runhouse.servers.obj_store import RunhouseStopIteration
from runhouse.utils import ClusterLogsFormatter

logger = get_logger(__name__)


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
    run_name: str
    data: Any = None
    serialization: Optional[str] = "none"
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


class FolderParams(BaseModel):
    path: str
    is_file: bool = False

    @field_validator("path", mode="before")
    def convert_path_to_string(cls, v):
        return str(v) if v is not None else v


class FolderLsParams(FolderParams):
    full_paths: Optional[bool] = True
    sort: Optional[bool] = False


class FolderGetParams(FolderParams):
    encoding: Optional[str] = None
    mode: Optional[str] = None


class FolderPutParams(FolderParams):
    contents: Any
    overwrite: Optional[bool] = False
    mode: Optional[str] = None
    serialization: Optional[str] = None


class FolderRmParams(FolderParams):
    contents: Optional[List] = None
    recursive: Optional[bool] = False


class FolderMvParams(FolderParams):
    dest_path: str
    overwrite: Optional[bool] = True

    @field_validator("dest_path", mode="before")
    def convert_path_to_string(cls, v):
        return str(v) if v is not None else v


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


def resolve_folder_path(path: str):
    return (
        None
        if path is None
        else Path(path).expanduser()
        if path.startswith("~")
        else Path(path).resolve()
    )


def folder_mkdir(path: Path):
    if not path.parent.is_dir():
        raise ValueError(
            f"Parent path {path.parent} does not exist or is not a directory"
        )

    path.mkdir(parents=True, exist_ok=True)

    return Response(output_type=OutputType.SUCCESS)


def folder_get(path: Path, encoding: str = None, mode: str = None):
    mode = mode or "rb"
    binary_mode = "b" in mode

    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Path {path} does not exist")

    try:
        with open(path, mode=mode, encoding=encoding) as f:
            file_contents = f.read()

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"File {path} not found")

    except PermissionError:
        raise HTTPException(
            status_code=403, detail=f"Permission denied for file in path {path}"
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error reading file {path}: {str(e)}"
        )

    if binary_mode and isinstance(file_contents, bytes):
        file_contents = file_contents.decode()

    return Response(
        data=file_contents,
        output_type=OutputType.RESULT_SERIALIZED,
        serialization=None,
    )


def folder_put(
    path: Path,
    contents: Dict[str, Any],
    overwrite: bool,
    mode: str = None,
    serialization: str = None,
):
    mode = mode or "wb"

    if contents and not isinstance(contents, dict):
        raise HTTPException(
            status_code=422,
            detail="`contents` argument must be a dict mapping filenames to file-like objects",
        )

    if overwrite is False:
        existing_files = {str(item.name) for item in path.iterdir()}
        intersection = existing_files.intersection(set(contents.keys()))
        if intersection:
            raise HTTPException(
                status_code=409,
                detail=f"File(s) {intersection} already exist(s) at path: {path}",
            )

    path.mkdir(parents=True, exist_ok=True)

    for filename, file_obj in contents.items():
        binary_mode = "b" in mode

        if serialization:
            file_obj = serialize_data(file_obj, serialization)

        if binary_mode and not isinstance(file_obj, bytes):
            file_obj = file_obj.encode()

        file_path = path / filename
        if not overwrite and file_path.exists():
            raise HTTPException(
                status_code=409, detail=f"File {file_path} already exists"
            )

        try:
            with open(file_path, mode) as f:
                f.write(file_obj)
        except Exception as e:
            HTTPException(
                status_code=500,
                detail=f"Failed to write file with mode '{mode}': {str(e)}",
            )

    return Response(output_type=OutputType.SUCCESS)


def folder_ls(path: Path, full_paths: bool, sort: bool):
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Path {path} does not exist")

    if not path.is_dir():
        raise HTTPException(status_code=400, detail=f"Path {path} is not a directory")

    paths = [p for p in path.iterdir()]

    # Sort the paths by modification time if sort is True
    if sort:
        paths.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    # Convert paths to strings and format them based on full_paths
    if full_paths:
        files = [str(p.resolve()) for p in paths]
    else:
        files = [p.name for p in paths]

    return Response(
        data=files,
        output_type=OutputType.RESULT_SERIALIZED,
        serialization=None,
    )


def folder_rm(path: Path, contents: List[str], recursive: bool):
    if contents:
        from runhouse import Folder

        try:
            Folder._delete_contents(contents, path, recursive)
        except ValueError as e:
            raise HTTPException(
                status_code=400,
                detail=str(e),
            )

        return Response(output_type=OutputType.SUCCESS)

    if not path.is_dir():
        path.unlink()
        return Response(output_type=OutputType.SUCCESS)

    if recursive:
        shutil.rmtree(path)
        return Response(output_type=OutputType.SUCCESS)

    items = list(path.iterdir())
    if not items:
        # Remove the empty directory
        path.rmdir()
        return Response(output_type=OutputType.SUCCESS)

    # Remove file contents, but not the directory itself (since recursive not set to `True`)
    for item in items:
        if item.is_file():
            item.unlink()
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Folder {item} found in {path}, recursive is set to `False`",
            )

    return Response(output_type=OutputType.SUCCESS)


def folder_mv(src_path: Path, dest_path: str, overwrite: bool):
    dest_path = resolve_folder_path(dest_path)

    if not src_path.exists():
        raise HTTPException(
            status_code=404, detail=f"The source path {src_path} does not exist"
        )

    if not overwrite and dest_path.exists():
        raise HTTPException(
            status_code=409,
            detail=f"The destination path {dest_path} already exists. Set `overwrite` to `True` to "
            f"overwrite the destination path",
        )

    # Create the destination directory if it doesn't exist and overwrite is set to `True`
    dest_path.parent.mkdir(parents=True, exist_ok=overwrite)

    # Move the directory
    shutil.move(str(src_path), str(dest_path))

    return Response(output_type=OutputType.SUCCESS)


def folder_exists(path: Path):
    folder_exists_resp = path.exists()
    if not path.is_file():
        folder_exists_resp = folder_exists_resp and path.is_dir()
    return Response(
        data=folder_exists_resp,
        output_type=OutputType.RESULT_SERIALIZED,
        serialization=None,
    )
