import asyncio
import json
import time
import warnings

from concurrent.futures import ThreadPoolExecutor
from functools import wraps
from pathlib import Path
from random import randrange
from typing import Any, Dict, List, Optional, Union

import httpx

import requests

from runhouse.globals import rns_client
from runhouse.logger import get_logger

from runhouse.resources.resource import Resource
from runhouse.servers.http.http_utils import (
    CallParams,
    CreateProcessParams,
    DeleteObjectParams,
    FolderGetParams,
    FolderLsParams,
    FolderMvParams,
    FolderParams,
    FolderPutParams,
    FolderRmParams,
    GetObjectParams,
    handle_response,
    InstallPackageParams,
    KillProcessParams,
    LogsParams,
    OutputType,
    PutObjectParams,
    PutResourceParams,
    RenameObjectParams,
    RunBashParams,
    serialize_data,
    SetProcessEnvVarsParams,
)

from runhouse.utils import ClusterLogsFormatter, generate_default_name, thread_coroutine


# Make this global so connections are pooled across instances of HTTPClient
session = requests.Session()
session.timeout = None

logger = get_logger(__name__)


def retry_with_exponential_backoff(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        MAX_RETRIES = 5
        retries = 0
        while True:
            try:
                return func(*args, **kwargs)
            except ConnectionError as e:
                retries += 1
                if retries == MAX_RETRIES:
                    raise e
                sleep_time = randrange(1, 2 ** (retries + 1) + 1)
                time.sleep(sleep_time)

    return wrapper


class HTTPClient:
    """
    Client for cluster RPCs
    """

    CHECK_TIMEOUT_SEC = 10

    def __init__(
        self,
        host: str,
        port: Optional[int],
        resource_address: str,
        auth=None,
        cert_path=None,
        use_https=False,
        system=None,
    ):
        self.host = host
        self.port = port
        self.auth = auth
        self.cert_path = cert_path
        self.use_https = use_https
        self.resource_address = resource_address
        self.system = system
        self.verify = False

        if self.use_https:
            # https://requests.readthedocs.io/en/latest/user/advanced/#ssl-cert-verification
            # Only verify with the specific cert path if the cert itself is self-signed, otherwise we use the default
            # setting of "True", which will verify the cluster's SSL certs
            self.verify = self.cert_path if self._certs_are_self_signed() else True

        self.async_session = httpx.AsyncClient(
            auth=self.auth, verify=self.verify, timeout=None
        )

        self.log_formatter = ClusterLogsFormatter(self.system)
        self._request_headers = rns_client.request_headers(self.resource_address)

    def _certs_are_self_signed(self) -> bool:
        """Checks whether the cert provided is self-signed. If it is, all client requests will include the path
        to the cert to be used for verification."""
        from cryptography import x509
        from cryptography.hazmat.backends import default_backend

        if not self.cert_path:
            # No cert path is specified, assume certs will be configured on the server (ex: via Caddy)
            return False

        cert_path = Path(self.cert_path)
        if not cert_path.exists():
            return False

        # Check whether the cert is self-signed
        with open(cert_path, "rb") as cert_file:
            cert = x509.load_pem_x509_certificate(cert_file.read(), default_backend())

        if cert.issuer == cert.subject:
            warnings.warn(
                f"Cert in use ({cert_path}) is self-signed, cannot independently verify it."
            )
            return True

        return False

    @staticmethod
    def from_endpoint(endpoint: str, resource_address: str, auth=None, cert_path=None):
        protocol, uri = endpoint.split("://")
        if protocol not in ["http", "https"]:
            raise ValueError(f"Invalid protocol: {protocol}")
        port = None
        if ":" in uri:
            host, port_and_route = uri.split(":", 1)
            port, _ = port_and_route.split("/", 1)
        else:
            host, _ = uri.split("/", 1)
        use_https = protocol == "https"

        if port is None:
            port = 443 if use_https else 80
        else:
            port = int(port)

        client = HTTPClient(
            host,
            port=port,
            auth=auth,
            cert_path=cert_path,
            resource_address=resource_address,
            use_https=False,
        )
        client.use_https = use_https
        return client

    def _formatted_url(self, endpoint: str):
        prefix = "https" if self.use_https else "http"
        if self.port:
            return f"{prefix}://{self.host}:{self.port}/{endpoint}"
        return f"{prefix}://{self.host}/{endpoint}"

    def request(
        self,
        endpoint,
        req_type="post",
        data=None,
        process=None,
        stream_logs=True,
        save=False,
        key=None,
        err_str=None,
        timeout=None,
        headers: Union[Dict, None] = None,
    ):
        headers = headers or self._request_headers
        json_dict = {
            "data": data,
            "process": process,
            "stream_logs": stream_logs,
            "save": save,
            "key": key,
        }
        return self.request_json(
            endpoint,
            req_type=req_type,
            json_dict=json_dict,
            err_str=err_str,
            timeout=timeout,
            headers=headers,
        )

    def request_json(
        self,
        endpoint: str,
        req_type: str = "post",
        json_dict: Any = None,
        err_str=None,
        timeout=None,
        headers: Union[Dict, None] = None,
    ):
        # Support use case where we explicitly do not want to provide headers (e.g. requesting a cert)
        headers = self._request_headers if headers != {} else headers
        req_fn = (
            session.get
            if req_type == "get"
            else session.put
            if req_type == "put"
            else session.delete
            if req_type == "delete"
            else session.post
        )
        # Note: For localhost (e.g. docker) do not add trailing slash (will lead to connection errors)
        endpoint = endpoint.strip("/")
        if (
            self.host not in ["localhost", "127.0.0.1", "0.0.0.0"]
            and "?" not in endpoint
        ):
            endpoint += "/"

        if req_type == "get":
            response = retry_with_exponential_backoff(req_fn)(
                self._formatted_url(endpoint),
                params=json_dict,
                headers=headers,
                auth=self.auth,
                verify=self.verify,
                timeout=timeout,
            )
        else:
            response = retry_with_exponential_backoff(req_fn)(
                self._formatted_url(endpoint),
                json=json_dict,
                headers=headers,
                auth=self.auth,
                verify=self.verify,
            )
        if response.status_code != 200:
            raise ValueError(
                f"Error calling {endpoint} on server: {response.content.decode()}"
            )
        resp_json = response.json()
        if isinstance(resp_json, dict) and "output_type" in resp_json:
            return handle_response(
                resp_json,
                resp_json["output_type"],
                err_str,
                log_formatter=self.log_formatter,
            )
        return resp_json

    def check_server(self):
        resp = session.get(
            self._formatted_url("check"),
            timeout=self.CHECK_TIMEOUT_SEC,
            verify=self.verify,
        )

        if resp.status_code != 200:
            raise ValueError(
                f"Error checking server: {resp.content.decode()}. Is the server running?"
            )

        rh_version = resp.json().get("rh_version", None)
        import runhouse

        if rh_version and not runhouse.__version__ == rh_version:
            logger.warning(
                f"Server was started with Runhouse version ({rh_version}), "
                f"but local Runhouse version is ({runhouse.__version__})"
            )

    def status(self, send_to_den: bool = False):
        """Load the remote cluster's status."""
        return self.request(
            f"status?send_to_den={send_to_den}",
            req_type="get",
        )

    def folder_ls(self, path: Union[str, Path], full_paths: bool, sort: bool):
        folder_params = FolderLsParams(
            path=path, full_paths=full_paths, sort=sort
        ).model_dump()
        return self.request_json(
            "/folder/method/ls", req_type="post", json_dict=folder_params
        )

    def folder_mkdir(self, path: Union[str, Path]):
        folder_params = FolderParams(path=path).model_dump()
        return self.request_json(
            "/folder/method/mkdir", req_type="post", json_dict=folder_params
        )

    def folder_mv(
        self, path: Union[str, Path], dest_path: Union[str, Path], overwrite: bool
    ):
        folder_params = FolderMvParams(
            path=path, dest_path=dest_path, overwrite=overwrite
        ).model_dump()
        return self.request_json(
            "/folder/method/mv", req_type="post", json_dict=folder_params
        )

    def folder_get(self, path: Union[str, Path], encoding: str, mode: str):
        folder_params = FolderGetParams(
            path=path, encoding=encoding, mode=mode
        ).model_dump()
        return self.request_json(
            "/folder/method/get", req_type="post", json_dict=folder_params
        )

    def folder_put(
        self,
        path: Union[str, Path],
        contents: Union[Dict[str, Any], Resource, List[Resource]],
        overwrite: bool,
        mode: str,
        serialization: str,
    ):
        folder_params = FolderPutParams(
            path=path,
            contents=serialize_data(contents, serialization),
            mode=mode,
            overwrite=overwrite,
            serialization=serialization,
        ).model_dump()
        return self.request_json(
            "/folder/method/put", req_type="post", json_dict=folder_params
        )

    def folder_rm(self, path: Union[str, Path], contents: List, recursive: bool):
        folder_params = FolderRmParams(
            path=path, recursive=recursive, contents=contents
        ).model_dump()
        return self.request_json(
            "/folder/method/rm", req_type="post", json_dict=folder_params
        )

    def folder_exists(self, path: str):
        folder_params = FolderParams(path=path).model_dump()
        return self.request_json(
            "/folder/method/exists", req_type="post", json_dict=folder_params
        )

    def get_certificate(self):
        cert: bytes = self.request(
            "cert",
            req_type="get",
            headers={},
        )
        # Create parent directory to store the cert
        Path(self.cert_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.cert_path, "wb") as file:
            file.write(cert)

    def _process_call_result(
        self,
        result,
        system,
        output_type,
    ):

        from runhouse.resources.module import Module

        if isinstance(result, Module):
            if (
                system
                and result.system
                and system.rns_address == result.system.rns_address
            ):
                result.system = system
        elif output_type == OutputType.CONFIG:
            if system and "system" in result and system.rns_address == result["system"]:
                result["system"] = system
            result = Resource.from_config(result, dryrun=True)

        return result

    def call(
        self,
        key: str,
        method_name: str,
        data: Any = None,
        serialization: Optional[str] = None,
        run_name: Optional[str] = None,
        stream_logs: bool = True,
        remote: bool = False,
        save=False,
        headers=None,
    ):
        """wrapper to temporarily support cluster's call signature"""
        return self.call_module_method(
            key,
            method_name,
            data=data,
            serialization=serialization,
            run_name=run_name,
            stream_logs=stream_logs,
            remote=remote,
            save=save,
            system=self.system,
            headers=headers,
        )

    def call_module_method(
        self,
        key: str,
        method_name: str,
        data: Any = None,
        serialization: Optional[str] = None,
        run_name: Optional[str] = None,
        stream_logs: bool = True,
        remote: bool = False,
        save=False,
        system=None,
        headers=None,
    ):
        """
        Client function to call the rpc for call_module_method
        """
        if run_name is None:
            run_name = generate_default_name(
                prefix=key if method_name == "__call__" else f"{key}_{method_name}",
                precision="ms",  # Higher precision because we see collisions within the same second
                sep="--",
            )

        # Measure the time it takes to send the message
        start = time.time()
        logger.info(
            f"{'Calling' if method_name else 'Getting'} {key}"
            + (f".{method_name}" if method_name else "")
        )
        serialization = serialization or "pickle"
        error_str = f"Error calling {method_name} on {key} on server"

        with ThreadPoolExecutor() as executor:
            # Run logs request in separate thread. Can start it before because it'll wait 5 seconds for the
            # calls request to begin.
            if stream_logs:
                logs_future = executor.submit(
                    thread_coroutine,
                    self._alogs_request(
                        run_name=run_name,
                        key=key,
                        serialization=serialization,
                        error_str=error_str,
                        create_async_client=True,
                    ),
                )
            response = retry_with_exponential_backoff(session.post)(
                self._formatted_url(f"{key}/{method_name}"),
                json=CallParams(
                    data=serialize_data(data, serialization),
                    serialization=serialization,
                    run_name=run_name,
                    stream_logs=stream_logs,
                    save=save,
                    remote=remote,
                ).model_dump(),
                headers=headers or self._request_headers,
                auth=self.auth,
                verify=self.verify,
            )

            if response.status_code != 200:
                raise ValueError(
                    f"Error calling {method_name} on server: {response.content.decode()}"
                )

            resp_json = response.json()
            function_result = handle_response(
                resp_json,
                resp_json["output_type"],
                error_str,
                log_formatter=self.log_formatter,
            )
            output_type = resp_json["output_type"]
            if stream_logs:
                _ = logs_future.result()

        end = time.time()

        function_result = self._process_call_result(
            function_result, system, output_type
        )

        if method_name:
            log_str = (
                f"Time to call {key}.{method_name}: {round(end - start, 2)} seconds"
            )
        else:
            log_str = f"Time to get {key}: {round(end - start, 2)} seconds"

        logger.info(log_str)
        return function_result

    async def acall(
        self,
        key: str,
        method_name: str,
        data: Any = None,
        serialization: Optional[str] = None,
        run_name: Optional[str] = None,
        stream_logs: bool = True,
        remote: bool = False,
        run_async=False,
        save=False,
    ):
        """wrapper to temporarily support cluster's call signature"""
        return await self.acall_module_method(
            key,
            method_name,
            data=data,
            serialization=serialization,
            run_name=run_name,
            stream_logs=stream_logs,
            remote=remote,
            run_async=run_async,
            save=save,
            system=self.system,
        )

    async def _acall_request(
        self,
        key: str,
        method_name: str,
        run_name: str,
        serialization: str,
        stream_logs: bool,
        run_async: bool,
        save: bool,
        remote: bool,
        data: Any = None,
        resource_address=None,
    ):
        response = await self.async_session.post(
            self._formatted_url(f"{key}/{method_name}"),
            json=CallParams(
                data=serialize_data(data, serialization),
                serialization=serialization,
                run_name=run_name,
                stream_logs=stream_logs,
                save=save,
                remote=remote,
                run_async=run_async,
            ).model_dump(),
            headers=self._request_headers,
        )
        if response.status_code != 200:
            raise ValueError(
                f"Error calling {method_name} on server: {response.content.decode()}"
            )

        resp_json = response.json()
        return resp_json

    async def _alogs_request(
        self,
        run_name: str,
        key: Optional[str] = None,
        node_ip_or_idx: Optional[Union[str, int]] = None,
        process: Optional[str] = None,
        serialization: Optional[str] = None,
        error_str: Optional[str] = None,
        create_async_client=False,
    ) -> None:
        # When running this in another thread, we need to explicitly create an async client here. When running within
        # the main thread, we can use the client that was passed in.
        if sum(arg is not None for arg in [key, node_ip_or_idx, process]) != 1:
            raise ValueError(
                "Exactly one of key, node_ip_or_idx, or process must be provided to get logs"
            )

        if create_async_client:
            client = httpx.AsyncClient(auth=self.auth, verify=self.verify, timeout=None)
        else:
            client = self.async_session

        if error_str is None:
            error_str = f"Error calling logs function on server for {run_name}"

        async with client.stream(
            "POST",
            self._formatted_url("logs"),
            headers=self._request_headers,
            json=LogsParams(
                run_name=run_name,
                node_ip_or_idx=node_ip_or_idx,
                process=process,
                key=key,
                serialization=serialization,
            ).model_dump(),
        ) as res:
            if res.status_code != 200:
                error_resp = await res.aread()
                raise ValueError(
                    f"Error calling logs function on server: {error_resp.decode()}"
                )
            async for response_json in res.aiter_lines():
                resp = json.loads(response_json)
                output_type = resp["output_type"]
                if output_type not in [
                    OutputType.EXCEPTION,
                    OutputType.STDOUT,
                    OutputType.STDERR,
                ]:
                    raise ValueError(
                        f"Unexpected output type from logs function: {output_type}"
                    )
                handle_response(
                    resp, output_type, error_str, log_formatter=self.log_formatter
                )

    async def acall_module_method(
        self,
        key: str,
        method_name: str,
        data: Any = None,
        serialization: Optional[str] = None,
        run_name: Optional[str] = None,
        stream_logs: bool = True,
        remote: bool = False,
        run_async=False,
        save=False,
        system=None,
    ):
        """
        Client function to call the rpc for call_module_method
        """
        if run_name is None:
            run_name = generate_default_name(
                prefix=key if method_name == "__call__" else f"{key}_{method_name}",
                precision="ms",  # Higher precision because we see collisions within the same second
                sep="--",
            )

        # Measure the time it takes to send the message
        start = time.time()
        logger.info(
            f"{'Calling' if method_name else 'Getting'} {key}"
            + (f".{method_name}" if method_name else "")
        )
        serialization = serialization or "pickle"
        error_str = f"Error calling {method_name} on {key} on server"

        acall_request = asyncio.create_task(
            self._acall_request(
                key=key,
                method_name=method_name,
                run_name=run_name,
                serialization=serialization,
                stream_logs=stream_logs,
                run_async=run_async,
                save=save,
                remote=remote,
                data=data,
            )
        )
        alogs_request = asyncio.create_task(
            self._alogs_request(
                run_name=run_name,
                key=key,
                serialization=serialization,
                error_str=error_str,
            )
        )

        output_type = None
        function_result = None
        for fut_result in asyncio.as_completed([acall_request, alogs_request]):
            resp_json = await fut_result
            # alogs_request returns None, acall_request returns a legitimate result
            if resp_json is not None:
                function_result = handle_response(
                    resp_json,
                    resp_json["output_type"],
                    error_str,
                    log_formatter=self.log_formatter,
                )
                output_type = resp_json["output_type"]

        end = time.time()

        function_result = self._process_call_result(
            function_result, system, output_type
        )

        if method_name:
            log_str = (
                f"Time to call {key}.{method_name}: {round(end - start, 2)} seconds"
            )
        else:
            log_str = f"Time to get {key}: {round(end - start, 2)} seconds"
        logger.info(log_str)
        return function_result

    def put_object(self, key: str, value: Any, process=None):
        return self.request_json(
            "object",
            req_type="post",
            json_dict=PutObjectParams(
                key=key,
                serialized_data=serialize_data(value, "pickle"),
                process_name=process,
                serialization="pickle",
            ).model_dump(),
            err_str=f"Error putting object {key}",
        )

    def put_resource(
        self, resource, process: Optional[str] = None, state=None, dryrun=False
    ):
        config = resource.config(condensed=False)
        return self.request_json(
            "resource",
            req_type="post",
            # TODO wire up dryrun properly
            json_dict=PutResourceParams(
                serialized_data=serialize_data([config, state, dryrun], "pickle"),
                process_name=process,
                serialization="pickle",
            ).model_dump(),
            err_str=f"Error putting resource {resource.name or type(resource)}",
        )

    def get(
        self,
        key: str,
        default: Any = None,
        remote=False,
        system=None,
    ):
        """Provides compatibility with cluster's get."""
        try:
            res = self.request_json(
                "object",
                req_type="get",
                json_dict=GetObjectParams(
                    key=key,
                    serialization="pickle",
                    remote=remote,
                ).model_dump(),
                err_str=f"Error getting object {key}",
            )
            if remote and isinstance(res, dict) and "resource_type" in res:
                # Reconstruct the resource from the config
                if "system" in res:
                    res["system"] = system
                res = Resource.from_config(res, dryrun=True)

        except KeyError as e:
            if default == KeyError:
                raise e
            return default
        return res

    def rename(self, old_key: str, new_key: str):
        """Provides compatibility with cluster's rename."""
        return self.rename_object(old_key, new_key)

    def rename_object(self, old_key, new_key):
        self.request_json(
            "rename",
            req_type="post",
            json_dict=RenameObjectParams(key=old_key, new_key=new_key).model_dump(),
            err_str=f"Error renaming object {old_key}",
        )

    def set_settings(self, new_settings: Dict[str, Any]):
        res = retry_with_exponential_backoff(session.post)(
            self._formatted_url("settings"),
            json=new_settings,
            headers=self._request_headers,
            auth=self.auth,
            verify=self.verify,
        )
        if res.status_code != 200:
            raise ValueError(
                f"Error switching to new settings: {new_settings} on server: {res.content.decode()}"
            )
        return res

    def set_cluster_name(self, name: str):
        resp = self.set_settings({"cluster_name": name})
        self.resource_address = name
        return resp

    def delete(self, keys=None, process=None):
        return self.request_json(
            "delete_object",
            req_type="post",
            json_dict=DeleteObjectParams(keys=keys or []).model_dump(),
            err_str=f"Error deleting keys {keys}",
        )

    def keys(self, process=None):
        return self.request(
            f"keys/?process_name={process}" if process else "keys", req_type="get"
        )

    ################################################################################################
    # Process related functions
    ################################################################################################

    def list_processes(self):
        return self.request_json("/processes", req_type="get")

    def create_process(
        self,
        params: CreateProcessParams,
    ):
        return self.request_json(
            "/create_process",
            req_type="post",
            json_dict=params.model_dump(),
        )

    def set_process_env_vars(
        self,
        process_name: str,
        env_vars: Dict[str, str],
    ):
        return self.request_json(
            "/process_env_vars",
            req_type="post",
            json_dict=SetProcessEnvVarsParams(
                process_name=process_name, env_vars=env_vars
            ).model_dump(),
        )

    def kill_process(self, process_name: str):
        return self.request_json(
            "/kill_process",
            req_type="post",
            json_dict=KillProcessParams(process_name=process_name).model_dump(),
        )

    def install_package(
        self,
        package: "Package",
        conda_env_name: Optional[str] = None,
        force_sync_local: bool = False,
    ):
        return self.request_json(
            "/install_package",
            req_type="post",
            json_dict=InstallPackageParams(
                package_config=package.config(),
                conda_env_name=conda_env_name,
                force_sync_local=force_sync_local,
            ).model_dump(),
        )

    def run_bash(
        self,
        command: str,
        node_ip_or_idx: Optional[Union[str, int]] = None,
        process: Optional[str] = None,
        run_name: Optional[str] = None,
        require_outputs: bool = False,
    ):
        if node_ip_or_idx is not None and process is not None:
            raise ValueError("Cannot specify both node and process")

        return self.request_json(
            "/run_bash",
            req_type="post",
            json_dict=RunBashParams(
                command=command,
                require_outputs=require_outputs,
                node_ip_or_idx=node_ip_or_idx,
                process=process,
                run_name=run_name,
            ).model_dump(),
        )
