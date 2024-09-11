import argparse
import asyncio
import inspect
import json
import traceback
import uuid
from functools import wraps
from pathlib import Path
from typing import Optional

import ray
import requests
import yaml
from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.openapi.docs import get_redoc_html, get_swagger_ui_html
from fastapi.responses import StreamingResponse

from runhouse.constants import (
    DEFAULT_HTTP_PORT,
    DEFAULT_HTTPS_PORT,
    DEFAULT_SERVER_HOST,
    DEFAULT_SERVER_PORT,
    EMPTY_DEFAULT_ENV_NAME,
    LOGGING_WAIT_TIME,
    RH_LOGFILE_PATH,
)
from runhouse.globals import configs, obj_store, rns_client
from runhouse.logger import get_logger
from runhouse.rns.utils.api import resolve_absolute_path, ResourceAccess
from runhouse.servers.caddy.config import CaddyConfig
from runhouse.servers.http.auth import averify_cluster_access
from runhouse.servers.http.certs import TLSCertConfig
from runhouse.servers.http.http_utils import (
    CallParams,
    DeleteObjectParams,
    deserialize_data,
    folder_exists,
    folder_get,
    folder_ls,
    folder_mkdir,
    folder_mv,
    folder_put,
    folder_rm,
    FolderGetParams,
    FolderLsParams,
    FolderMvParams,
    FolderParams,
    FolderPutParams,
    FolderRmParams,
    get_token_from_request,
    handle_exception_response,
    OutputType,
    PutObjectParams,
    PutResourceParams,
    RenameObjectParams,
    resolve_folder_path,
    Response,
    serialize_data,
    ServerSettings,
)
from runhouse.servers.obj_store import (
    ClusterServletSetupOption,
    ObjStoreError,
    RaySetupOption,
)
from runhouse.utils import generate_default_name, sync_function

app = FastAPI(docs_url=None, redoc_url=None)

logger = get_logger(__name__)


def validate_cluster_access(func):
    """If using Den auth, validate the user's cluster subtoken and access to the cluster before continuing."""

    @wraps(func)
    async def wrapper(*args, **kwargs):
        request: Request = kwargs.get("request")
        den_auth_enabled: bool = await HTTPServer.get_den_auth()
        is_coro = inspect.iscoroutinefunction(func)

        func_call: bool = func.__name__ in ["post_call", "get_call"]

        # restrict access for folder specific APIs
        access_level_required = (
            ResourceAccess.WRITE if func.__name__.startswith("folder") else None
        )
        token = get_token_from_request(request)

        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        ctx_token = obj_store.set_ctx(request_id=request_id, token=token)

        try:
            if den_auth_enabled and not func_call:
                if token is None:
                    raise HTTPException(
                        status_code=401,
                        detail="No Runhouse token provided. Try running `$ runhouse login` or visiting "
                        "https://run.house/login to retrieve a token. If calling via HTTP, please "
                        "provide a valid token in the Authorization header.",
                    )
                cluster_uri = (await obj_store.aget_cluster_config()).get("name")
                cluster_access = await averify_cluster_access(
                    cluster_uri, token, access_level_required
                )
                if not cluster_access:
                    # Must have cluster access for all the non func calls
                    # Note: for func calls we handle the auth in the object store
                    raise HTTPException(
                        status_code=403,
                        detail="Cluster access is required for this operation.",
                    )

            if is_coro:
                res = await func(*args, **kwargs)
            else:
                res = func(*args, **kwargs)
        except Exception as e:
            raise e
        finally:
            if ctx_token:
                obj_store.unset_ctx(ctx_token)
        return res

    return wrapper


class HTTPServer:
    SKY_YAML = str(Path("~/.sky/sky_ray.yml").expanduser())
    memory_exporter = None

    @classmethod
    async def ainitialize(
        cls,
        default_env_name=None,
        conda_env=None,
        from_test: bool = False,
        *args,
        **kwargs,
    ):
        runtime_env = {"conda": conda_env} if conda_env else None

        default_env_name = default_env_name or EMPTY_DEFAULT_ENV_NAME

        # Ray and ClusterServlet should already be
        # initialized by the start script (see below)
        # But if the HTTPServer was started standalone in a test,
        # We still want to make sure the cluster servlet is initialized
        if from_test:
            await obj_store.ainitialize(
                default_env_name,
                setup_ray=RaySetupOption.TEST_PROCESS,
                runtime_env=runtime_env,
            )

        # TODO disabling due to latency, figure out what to do with this
        # try:
        #     # Collect metadata for the cluster immediately on init
        #     self._collect_cluster_stats()
        # except Exception as e:
        #     logger.error(f"Failed to collect cluster stats: {str(e)}")

        # We initialize a default env servlet where some things may run.
        _ = obj_store.get_env_servlet(
            env_name=default_env_name,
            create=True,
            runtime_env=runtime_env,
        )

        if default_env_name == EMPTY_DEFAULT_ENV_NAME:
            from runhouse.resources.envs import Env

            default_env = Env(name=default_env_name)
            data = (default_env.config(condensed=False), {}, False)
            obj_store.put_resource(
                serialized_data=data, serialization=None, env_name=default_env_name
            )

    @classmethod
    def initialize(
        cls,
        default_env_name=None,
        conda_env=None,
        from_test: bool = False,
        *args,
        **kwargs,
    ):
        return sync_function(cls.ainitialize)(
            default_env_name,
            conda_env,
            from_test,
            *args,
            **kwargs,
        )

    @classmethod
    async def get_den_auth(cls):
        return obj_store.is_den_auth_enabled()

    @classmethod
    async def aenable_den_auth(cls, flush: Optional[bool] = True):
        await obj_store.aenable_den_auth()

        if flush:
            await obj_store.aclear_auth_cache()

    def enable_den_auth(flush: Optional[bool] = True):
        return sync_function(HTTPServer.aenable_den_auth)(flush)

    @classmethod
    async def adisable_den_auth(cls):
        await obj_store.adisable_den_auth()

    @classmethod
    def disable_den_auth(cls):
        return sync_function(HTTPServer.adisable_den_auth)()

    @staticmethod
    @app.get("/cert")
    def get_cert():
        """Download the certificate file for this server necessary for enabling HTTPS.
        User must have access to the cluster in order to download the certificate."""

        # Default for this endpoint is "pickle" serialization
        serialization = "pickle"

        try:
            certs_config = TLSCertConfig()
            cert_path = certs_config.cert_path

            if not Path(cert_path).exists():
                raise FileNotFoundError(
                    f"No certificate found on cluster in path: {cert_path}"
                )

            with open(cert_path, "rb") as cert_file:
                cert = cert_file.read()

            return Response(
                data=serialize_data(cert, serialization),
                output_type=OutputType.RESULT_SERIALIZED,
                serialization=serialization,
            )

        except Exception as e:
            logger.exception(e)
            exception_data = {
                "error": serialize_data(e, serialization),
                "traceback": traceback.format_exc(),
            }
            return Response(
                output_type=OutputType.EXCEPTION,
                data=exception_data,
                serialization=serialization,
            )

    @staticmethod
    @app.get("/check")
    def check_server():

        # Default for this endpoint is "pickle" serialization
        serialization = "pickle"

        try:
            if not ray.is_initialized():
                raise Exception("Ray is not initialized, restart the server.")
            logger.info("Server is up.")

            import runhouse

            return {"rh_version": runhouse.__version__}
        except Exception as e:
            logger.exception(e)
            exception_data = {
                "error": serialize_data(e, serialization),
                "traceback": traceback.format_exc(),
            }
            return Response(
                output_type=OutputType.EXCEPTION,
                data=exception_data,
                serialization=serialization,
            )

    @staticmethod
    @app.post("/settings")
    @validate_cluster_access
    async def update_settings(request: Request, message: ServerSettings) -> Response:
        if message.cluster_name:
            await obj_store.aset_cluster_config_value("name", message.cluster_name)

        if message.den_auth:
            await HTTPServer.aenable_den_auth(flush=message.flush_auth_cache)
        elif message.den_auth is not None and not message.den_auth:
            await HTTPServer.adisable_den_auth()

        if message.autostop_mins:
            obj_store.set_cluster_config_value("autostop_mins", message.autostop_mins)

        if message.status_check_interval:
            obj_store.set_cluster_config_value(
                "status_check_interval", message.status_check_interval
            )

        return Response(output_type=OutputType.SUCCESS)

    @staticmethod
    async def _call(
        key: str,
        method_name: Optional[str] = None,
        params: CallParams = Body(default=None),
    ):
        # Allow an empty body
        params = params or CallParams()

        try:
            if not params.run_name:
                raise ValueError("run_name is required for all calls.")
            # Call async so we can loop to collect logs until the result is ready

            fut = asyncio.create_task(
                obj_store.acall(
                    key=key,
                    method_name=method_name,
                    data=params.data,
                    stream_logs=params.stream_logs,
                    serialization=params.serialization,
                    run_name=params.run_name,
                    remote=params.remote,
                )
            )
            # If stream_logs is False, we'll wait for the result and return it
            if not params.stream_logs:
                return await fut

            return StreamingResponse(
                HTTPServer._get_results_and_logs_generator(
                    fut=fut,
                    run_name=params.run_name,
                    serialization=params.serialization,
                ),
                media_type="application/json",
            )
        except Exception as e:
            return handle_exception_response(
                e,
                traceback.format_exc(),
                serialization=params.serialization,
                from_http_server=True,
            )

    # Open API docs routes
    @staticmethod
    @app.get("/{key}/openapi.json")
    @validate_cluster_access
    async def get_openapi_spec(request: Request, key: str):
        try:
            module_openapi_spec = await obj_store.acall(
                key, method_name="openapi_spec", serialization=None
            )
        except (AttributeError, ObjStoreError):
            # The object put on the server is not an `rh.Module`, so it doesn't have an openapi_spec method
            # OR
            # The object is not found in the object store at all
            module_openapi_spec = None

        if not module_openapi_spec:
            raise HTTPException(status_code=404, detail=f"Module {key} not found.")

        return module_openapi_spec

    @staticmethod
    @app.get("/{key}/redoc")
    @validate_cluster_access
    async def get_redoc(request: Request, key: str):
        return get_redoc_html(
            openapi_url=f"/{key}/openapi.json", title="Developer Documentation"
        )

    @staticmethod
    @app.get("/{key}/docs")
    @validate_cluster_access
    async def get_swagger_ui_html(request: Request, key: str):
        return get_swagger_ui_html(
            openapi_url=f"/{key}/openapi.json",
            title=f"{key} - Swagger UI",
            oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
            init_oauth=app.swagger_ui_init_oauth,
        )

    # TODO match "/{key}/{method_name}/{path:more_path}" for asgi / proxy requests
    @staticmethod
    @app.post("/{key}/{method_name}")
    @validate_cluster_access
    async def post_call(
        request: Request,
        key: str,
        method_name: str = None,
        params: CallParams = Body(default=None),
    ):
        return await HTTPServer._call(key, method_name, params)

    @staticmethod
    @app.get("/{key}/{method_name}")
    @validate_cluster_access
    async def get_call(
        request: Request,
        key: str,
        method_name: Optional[str] = None,
        serialization: Optional[str] = None,
        run_name: Optional[str] = None,
        stream_logs: Optional[bool] = False,
        save: Optional[bool] = False,
        remote: Optional[bool] = False,
    ):
        # Default argument to json doesn't allow a user to pass in a serialization string if they want
        # But, if they didn't pass anything, we want it to be `json` by default.
        serialization = serialization or "json"
        try:
            if run_name is None and stream_logs:
                raise ValueError(
                    "run_name is required for all calls when stream_logs is True."
                )

            if run_name is None:
                run_name = generate_default_name(
                    prefix=key if method_name == "__call__" else f"{key}_{method_name}",
                    precision="ms",  # Higher precision because we see collisions within the same second
                    sep="@",
                )

            # The types need to be explicitly specified as parameters first so that
            # we can cast Query params to the right type.
            params = CallParams(
                serialization=serialization,
                run_name=run_name,
                stream_logs=stream_logs,
                save=save,
                remote=remote,
            )

            query_params_remaining = dict(request.query_params)
            call_params_dict = params.model_dump()
            for k, v in dict(request.query_params).items():
                # If one of the query_params matches an arg in CallParams, set it
                # And also remove it from the query_params dict, so the rest
                # of the args will be passed as kwargs
                if k in call_params_dict:
                    del query_params_remaining[k]

            data = {
                "args": [],
                "kwargs": query_params_remaining,
            }
            params.data = serialize_data(data, serialization)

            return await HTTPServer._call(key, method_name, params)
        except Exception as e:
            logger.exception(e)
            exception_data = {
                "error": serialize_data(e, serialization),
                "traceback": traceback.format_exc(),
            }
            return Response(
                output_type=OutputType.EXCEPTION,
                data=exception_data,
                serialization=serialization,
            )

    @staticmethod
    def _get_logfiles(log_key, log_type=None):
        if not log_key:
            return None
        key_logs_path = Path(RH_LOGFILE_PATH) / log_key
        if key_logs_path.exists():
            # Logs are like: `.rh/logs/key/key.[out|err]`
            glob_pattern = (
                "*.out"
                if log_type == "stdout"
                else "*.err"
                if log_type == "stderr"
                else "*.[oe][ur][tr]"
            )
            return [str(f.absolute()) for f in key_logs_path.glob(glob_pattern)]
        else:
            return None

    @staticmethod
    def open_new_logfiles(key, open_files):
        logfiles = HTTPServer._get_logfiles(key)
        if logfiles:
            for f in logfiles:
                if f not in [o.name for o in open_files]:
                    logger.info(f"Streaming logs from {f}")
                    open_files.append(open(f, "r"))
        return open_files

    @staticmethod
    async def _get_results_and_logs_generator(fut, run_name, serialization=None):
        logger.debug(f"Streaming logs for key {run_name}")
        open_logfiles = []
        waiting_for_results = True

        try:
            while waiting_for_results:
                if fut.done():
                    waiting_for_results = False
                    ret_val = fut.result()
                    yield json.dumps(jsonable_encoder(ret_val)) + "\n"
                else:
                    await asyncio.sleep(LOGGING_WAIT_TIME)
                # Grab all the lines written to all the log files since the last time we checked, including
                # any new log files that have been created
                open_logfiles = HTTPServer.open_new_logfiles(run_name, open_logfiles)
                ret_lines = []
                for i, f in enumerate(open_logfiles):
                    file_lines = f.readlines()
                    if file_lines:
                        # TODO [DG] handle .out vs .err, and multiple workers
                        # if len(logfiles) > 1:
                        #     ret_lines.append(f"Process {i}:")
                        ret_lines += file_lines
                if ret_lines:
                    logger.debug(f"Yielding logs for key {run_name}")
                    yield json.dumps(
                        jsonable_encoder(
                            Response(
                                data=ret_lines,
                                output_type=OutputType.STDOUT,
                            )
                        )
                    ) + "\n"

        except Exception as e:
            logger.exception(e)
            # NOTE: We do not convert the exception to an HTTPException here, because once we're inside this
            # generator starlette has already returned the StreamingResponse and there is no way to halt the stream
            # to return a 403 instead. Users shouldn't notice much of a difference, because if working through the
            # client the exception will be serialized and returned to appear as a native python exception, and if
            # working through an HTTP call stream_logs is False by default, so a normal HTTPException will be raised
            # above before entering this generator.
            exception_data = {
                "error": serialize_data(e, serialization),
                "traceback": traceback.format_exc(),
            }
            yield json.dumps(
                jsonable_encoder(
                    Response(
                        output_type=OutputType.EXCEPTION,
                        data=exception_data,
                        serialization=serialization,
                    )
                )
            )
        finally:
            if not open_logfiles:
                logger.warning(f"No logfiles found for call {run_name}")
            for f in open_logfiles:
                f.close()

    @staticmethod
    @app.post("/resource")
    @validate_cluster_access
    async def put_resource(request: Request, params: PutResourceParams):
        try:
            env_name = params.env_name
            return await obj_store.aput_resource(
                serialized_data=params.serialized_data,
                serialization=params.serialization,
                env_name=env_name,
            )
        except Exception as e:
            return handle_exception_response(
                e, traceback.format_exc(), from_http_server=True
            )

    @staticmethod
    @app.post("/object")
    @validate_cluster_access
    async def put_object(request: Request, params: PutObjectParams):
        try:
            await obj_store.aput(
                key=params.key,
                value=params.serialized_data,
                env=params.env_name,
                serialization=params.serialization,
                create_env_if_not_exists=True,
            )
            return Response(output_type=OutputType.SUCCESS)
        except Exception as e:
            return handle_exception_response(
                e,
                traceback.format_exc(),
                serialization=params.serialization,
                from_http_server=True,
            )

    @staticmethod
    @app.get("/object")
    @validate_cluster_access
    async def get_object(
        request: Request,
        key: str,
        serialization: Optional[str] = "json",
        remote: bool = False,
    ):
        try:
            return await obj_store.aget(
                key=key,
                serialization=serialization,
                remote=remote,
            )
        except Exception as e:
            return handle_exception_response(
                e,
                traceback.format_exc(),
                serialization=serialization,
                from_http_server=True,
            )

    @staticmethod
    @app.post("/rename")
    @validate_cluster_access
    async def rename_object(request: Request, params: RenameObjectParams):
        try:
            await obj_store.arename(
                old_key=params.key,
                new_key=params.new_key,
            )
            return Response(output_type=OutputType.SUCCESS)
        except Exception as e:
            return handle_exception_response(
                e, traceback.format_exc(), from_http_server=True
            )

    @staticmethod
    @app.post("/folder/method/ls")
    @validate_cluster_access
    async def folder_ls_cmd(request: Request, ls_params: FolderLsParams):
        try:
            path = resolve_folder_path(ls_params.path)
            return folder_ls(path, full_paths=ls_params.full_paths, sort=ls_params.sort)

        except Exception as e:
            return handle_exception_response(
                e, traceback.format_exc(), from_http_server=True
            )

    @staticmethod
    @app.post("/folder/method/mkdir")
    @validate_cluster_access
    async def folder_mkdir_cmd(request: Request, folder_params: FolderParams):
        try:
            path = resolve_folder_path(folder_params.path)
            return folder_mkdir(path)

        except Exception as e:
            return handle_exception_response(
                e, traceback.format_exc(), from_http_server=True
            )

    @staticmethod
    @app.post("/folder/method/get")
    @validate_cluster_access
    async def folder_get_cmd(request: Request, get_params: FolderGetParams):
        try:
            path = resolve_folder_path(get_params.path)
            return folder_get(path, mode=get_params.mode, encoding=get_params.encoding)

        except Exception as e:
            return handle_exception_response(
                e, traceback.format_exc(), from_http_server=True
            )

    @staticmethod
    @app.post("/folder/method/put")
    @validate_cluster_access
    async def folder_put_cmd(request: Request, put_params: FolderPutParams):
        try:
            path = resolve_folder_path(put_params.path)
            serialization = put_params.serialization
            serialized_contents = put_params.contents
            contents = deserialize_data(serialized_contents, serialization)

            return folder_put(
                path,
                contents=contents,
                overwrite=put_params.overwrite,
                mode=put_params.mode,
                serialization=serialization,
            )

        except Exception as e:
            return handle_exception_response(
                e, traceback.format_exc(), from_http_server=True
            )

    @staticmethod
    @app.post("/folder/method/rm")
    @validate_cluster_access
    async def folder_rm_cmd(request: Request, rm_params: FolderRmParams):
        try:
            path = resolve_folder_path(rm_params.path)
            return folder_rm(
                path, contents=rm_params.contents, recursive=rm_params.recursive
            )

        except Exception as e:
            return handle_exception_response(
                e, traceback.format_exc(), from_http_server=True
            )

    @staticmethod
    @app.post("/folder/method/mv")
    @validate_cluster_access
    async def folder_mv_cmd(request: Request, mv_params: FolderMvParams):
        try:
            path = resolve_folder_path(mv_params.path)
            return folder_mv(
                src_path=path,
                dest_path=mv_params.dest_path,
                overwrite=mv_params.overwrite,
            )

        except Exception as e:
            return handle_exception_response(
                e, traceback.format_exc(), from_http_server=True
            )

    @staticmethod
    @app.post("/folder/method/exists")
    @validate_cluster_access
    async def folder_exists_cmd(request: Request, folder_params: FolderParams):
        try:
            path = resolve_folder_path(folder_params.path)
            return folder_exists(path=path)

        except Exception as e:
            return handle_exception_response(
                e, traceback.format_exc(), from_http_server=True
            )

    @staticmethod
    @app.post("/delete_object")
    @validate_cluster_access
    async def delete_obj(request: Request, params: DeleteObjectParams):
        try:
            if len(params.keys) == 0:
                cleared = await obj_store.akeys()
                await obj_store.aclear()
            else:
                cleared = []
                for key in params.keys:
                    await obj_store.adelete(key)
                    cleared.append(key)

            # Expicitly tell the client not to attempt to deserialize the output
            return Response(
                data=cleared,
                output_type=OutputType.RESULT_SERIALIZED,
                serialization=None,
            )
        except Exception as e:
            return handle_exception_response(
                e, traceback.format_exc(), from_http_server=True
            )

    @staticmethod
    @app.get("/keys")
    @validate_cluster_access
    async def get_keys(request: Request, env_name: Optional[str] = None):
        try:
            if not env_name:
                output = await obj_store.akeys()
            else:
                output = await obj_store.akeys_for_env_servlet_name(env_name)

            # Expicitly tell the client not to attempt to deserialize the output
            return Response(
                data=output,
                output_type=OutputType.RESULT_SERIALIZED,
                serialization=None,
            )
        except Exception as e:
            return handle_exception_response(
                e, traceback.format_exc(), from_http_server=True
            )

    @staticmethod
    @app.get("/status")
    @validate_cluster_access
    def get_status(request: Request, send_to_den: bool = False):

        return obj_store.status(send_to_den=send_to_den)

    @staticmethod
    def _collect_cluster_stats():
        """Collect cluster metadata and send to Grafana Loki"""
        if not configs.data_collection_enabled():
            return

        cluster_data = HTTPServer._cluster_status_report()
        sky_data = HTTPServer._cluster_sky_report()

        HTTPServer._log_cluster_data(
            {**cluster_data, **sky_data},
            labels={"username": configs.username, "environment": "prod"},
        )

    @staticmethod
    def _cluster_status_report():
        import ray._private.usage.usage_lib as ray_usage_lib
        from ray._raylet import GcsClient

        gcs_client = GcsClient(address="127.0.0.1:6379", nums_reconnect_retry=20)

        # fields : ['ray_version', 'python_version']
        cluster_metadata = ray_usage_lib.get_cluster_metadata(gcs_client)

        # fields: ['total_num_cpus', 'total_num_gpus', 'total_memory_gb', 'total_object_store_memory_gb']
        cluster_status_report = ray_usage_lib.get_cluster_status_to_report(
            gcs_client
        ).__dict__

        return {**cluster_metadata, **cluster_status_report}

    @staticmethod
    def _cluster_sky_report():
        try:
            with open(HTTPServer.SKY_YAML, "r") as stream:
                sky_ray_data = yaml.safe_load(stream)
        except FileNotFoundError:
            # For non on-demand clusters we won't have sky data
            return {}

        provider = sky_ray_data["provider"]
        node_config = sky_ray_data["available_node_types"].get("ray.head.default", {})

        return {
            "cluster_name": sky_ray_data.get("cluster_name"),
            "region": provider.get("region"),
            "provider": provider.get("module"),
            "instance_type": node_config.get("node_config", {}).get("InstanceType"),
        }

    @staticmethod
    def _log_cluster_data(data: dict, labels: dict):
        from runhouse.rns.utils.api import log_timestamp

        payload = {
            "streams": [
                {"stream": labels, "values": [[str(log_timestamp()), json.dumps(data)]]}
            ]
        }

        payload = json.dumps(payload)
        resp = requests.post(
            f"{rns_client.api_server_url}/admin/logs", data=json.dumps(payload)
        )

        if resp.status_code == 405:
            # api server not configured to receive grafana logs
            return

        if resp.status_code != 200:
            logger.error(
                f"({resp.status_code}) Failed to send logs to Grafana Loki: {resp.text}"
            )


async def main():
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help=f"Host to run server on. By default will run on {DEFAULT_SERVER_HOST}",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port to run daemon on on. If provided and Caddy is not enabled, "
        f"will attempt to run the daemon on this port, defaults to {DEFAULT_SERVER_PORT}",
    )
    parser.add_argument(
        "--conda-env", type=str, default=None, help="Conda env to run server in"
    )
    parser.add_argument(
        "--use-https",
        action="store_true",  # if providing --use-https will be set to True
        default=argparse.SUPPRESS,  # If user didn't specify, attribute will not be present (not False)
        help="Start an HTTPS server with new TLS certs",
    )
    parser.add_argument(
        "--use-den-auth",
        action="store_true",  # if providing --use-den-auth will be set to True
        default=argparse.SUPPRESS,  # If user didn't specify, attribute will not be present (not False)
        help="Whether to authenticate requests with a Runhouse token",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default=None,
        help="Server domain name",
    )
    parser.add_argument(
        "--ssl-keyfile",
        type=str,
        default=None,
        help="Path to SSL key file on the cluster to use for HTTPS",
    )
    parser.add_argument(
        "--ssl-certfile",
        type=str,
        default=None,
        help="Path to SSL cert file on the cluster to use for HTTPS",
    )
    parser.add_argument(
        "--restart-proxy",
        action="store_true",  # if providing --restart-proxy will be set to True
        help="Reconfigure Caddy",
    )
    parser.add_argument(
        "--use-caddy",
        action="store_true",  # if providing --use-caddy will be set to True
        default=argparse.SUPPRESS,  # If user didn't specify, attribute will not be present (not False)
        help="Configure Caddy as a reverse proxy",
    )
    parser.add_argument(
        "--certs-address",
        type=str,
        default=None,
        help="Address to use for generating self-signed certs and enabling HTTPS. (e.g. public IP address)",
    )
    parser.add_argument(
        "--default-env-name",
        type=str,
        default=None,
        help="Name of env where the HTTP server is started.",
    )
    parser.add_argument(
        "--api-server-url",
        type=str,
        default=rns_client.api_server_url,
        help="URL of Runhouse Den",
    )
    parser.add_argument(
        "--from-python",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Whether HTTP server is called from Python rather than CLI.",
    )

    parse_args = parser.parse_args()

    conda_name = parse_args.conda_env
    restart_proxy = parse_args.restart_proxy
    api_server_url = parse_args.api_server_url
    default_env_name = parse_args.default_env_name

    if not hasattr(parse_args, "from_python") and not default_env_name:
        # detect default env if called from runhouse cli (start/restart) and cluster_config exists
        from runhouse.resources.hardware.utils import load_cluster_config_from_file

        cluster_config = load_cluster_config_from_file()
        if cluster_config.get("default_env"):
            default_env_name = cluster_config.get("default_env")["name"]

    # The object store and the cluster servlet within it need to be
    # initialized in order to call `obj_store.get_cluster_config()`, which
    # uses the object store to load the cluster config from Ray.
    # When setting up the server, we always want to create a new ClusterServlet.
    # We only want to forcibly start a Ray cluster if asked.
    # We connect this to the "base" env, which we'll initialize later,
    # so writes to the obj_store within the server get proxied to the "base" env.
    await obj_store.ainitialize(
        default_env_name,
        setup_cluster_servlet=ClusterServletSetupOption.FORCE_CREATE,
    )

    cluster_config = await obj_store.aget_cluster_config()
    if not cluster_config:
        logger.warning(
            "Cluster config is not set. Using default values where possible."
        )
    else:
        logger.info("Loaded cluster config from Ray.")

    ########################################
    # Handling args that could be specified in the
    # cluster_config.json or via CLI args
    ########################################

    # Den URL
    cluster_config["api_server_url"] = api_server_url

    # Server port
    if parse_args.port is not None and parse_args.port != cluster_config.get(
        "server_port"
    ):
        logger.warning(
            f"CLI provided server port: {parse_args.port} is different from the server port specified in "
            f"cluster_config.json: {cluster_config.get('server_port')}. Prioritizing CLI provided port."
        )

    port_arg = (
        parse_args.port or cluster_config.get("server_port") or DEFAULT_SERVER_PORT
    )

    if port_arg is not None:
        cluster_config["server_port"] = port_arg

    # Den auth enabled
    if hasattr(
        parse_args, "use_den_auth"
    ) and parse_args.use_den_auth != cluster_config.get("den_auth", False):
        logger.warning(
            f"CLI provided den_auth: {parse_args.use_den_auth} is different from the den_auth specified in "
            f"cluster_config.json: {cluster_config.get('den_auth')}. Prioritizing CLI provided den_auth."
        )

    den_auth = getattr(parse_args, "use_den_auth", False) or cluster_config.get(
        "den_auth", False
    )
    cluster_config["den_auth"] = den_auth

    if den_auth:
        await obj_store.aenable_den_auth()

    domain = parse_args.domain or cluster_config.get("domain", None)
    cluster_config["domain"] = domain

    # Keyfile
    if (
        parse_args.ssl_keyfile is not None
        and parse_args.ssl_keyfile != cluster_config.get("ssl_keyfile")
    ):
        logger.warning(
            f"CLI provided ssl_keyfile: {parse_args.ssl_keyfile} is different from the ssl_keyfile specified in "
            f"cluster_config.json: {cluster_config.get('ssl_keyfile')}. Prioritizing CLI provided ssl_keyfile."
        )

    ssl_keyfile_path_arg = parse_args.ssl_keyfile or cluster_config.get("ssl_keyfile")
    if ssl_keyfile_path_arg is not None:
        cluster_config["ssl_keyfile"] = ssl_keyfile_path_arg
    parsed_ssl_keyfile = (
        resolve_absolute_path(ssl_keyfile_path_arg) if ssl_keyfile_path_arg else None
    )

    # Certfile
    if (
        parse_args.ssl_certfile is not None
        and parse_args.ssl_certfile != cluster_config.get("ssl_certfile")
    ):
        logger.warning(
            f"CLI provided ssl_certfile: {parse_args.ssl_certfile} is different from the ssl_certfile specified in "
            f"cluster_config.json: {cluster_config.get('ssl_certfile')}. Prioritizing CLI provided ssl_certfile."
        )

    ssl_certfile_path_arg = parse_args.ssl_certfile or cluster_config.get(
        "ssl_certfile"
    )
    if ssl_certfile_path_arg is not None:
        cluster_config["ssl_certfile"] = ssl_certfile_path_arg
    parsed_ssl_certfile = (
        resolve_absolute_path(ssl_certfile_path_arg) if ssl_certfile_path_arg else None
    )

    # Host
    if parse_args.host is not None and parse_args.host != cluster_config.get(
        "server_host"
    ):
        logger.warning(
            f"CLI provided server_host: {parse_args.host} is different from the server_host specified in "
            f"cluster_config.json: {cluster_config.get('server_host')}. Prioritizing CLI provided server_host."
        )

    host = parse_args.host or cluster_config.get("server_host") or DEFAULT_SERVER_HOST
    cluster_config["server_host"] = host

    # Address in the case we're a TLS server
    if (
        parse_args.certs_address is not None
        and parse_args.certs_address != cluster_config.get("ips", [None])[0]
    ):
        logger.warning(
            f"CLI provided certs_address: {parse_args.certs_address} is different from the certs_address specified in "
            f"cluster_config.json: {cluster_config.get('ips', [None])[0]}. Prioritizing CLI provided certs_address."
        )

    certs_addresses = parse_args.certs_address or cluster_config.get("ips", None)

    # We don't want to unset multiple addresses if they were set in the cluster config
    if certs_addresses is not None:
        cluster_config["ips"] = certs_addresses
    else:
        cluster_config["ips"] = ["0.0.0.0"]

    certs_address = certs_addresses[0] if certs_addresses else None

    config_conn = cluster_config.get("server_connection_type")

    # Use caddy as reverse proxy
    if hasattr(parse_args, "use_caddy") and parse_args.use_caddy != (
        config_conn in ["tls", "none"]
    ):
        logger.warning(
            f"CLI provided use_caddy: {parse_args.use_caddy} is different from the server_connection_type specified in "
            f"cluster_config.json: {config_conn}. Prioritizing CLI provided use_caddy."
        )

    # Use HTTPS
    if hasattr(parse_args, "use_https") and parse_args.use_https != (
        config_conn == "tls"
    ):
        logger.warning(
            f"CLI provided use_https: {parse_args.use_https} is different from the server_connection_type specified in "
            f"cluster_config.json: {config_conn}. Prioritizing CLI provided use_https."
        )

    use_caddy = getattr(parse_args, "use_caddy", False) or (
        config_conn in ["tls", "none"]
    )
    use_https = getattr(parse_args, "use_https", False) or (config_conn == "tls")
    cluster_config["server_connection_type"] = (
        "tls" if use_https else "none" if use_caddy else config_conn
    )

    # If there was no `cluster_config.json`, then server was created
    # simply with `runhouse start`.
    # A real `cluster_config` that was loaded
    # from json would have this set for sure
    if not cluster_config.get("resource_subtype"):
        # This is needed for the Cluster object to be created
        # in rh.here.
        cluster_config["resource_subtype"] = "Cluster"

        # server_connection_type is not set up if this is done through
        # a local `runhouse start`
        cluster_config["server_connection_type"] = "tls" if use_https else "none"

    await obj_store.aset_cluster_config(cluster_config)

    await HTTPServer.ainitialize(
        default_env_name=default_env_name,
        conda_env=conda_name,
    )

    if den_auth:
        # Update den auth if enabled - keep as a class attribute to be referenced by the validator decorator
        await HTTPServer.aenable_den_auth()

    if use_https and not domain:
        # If using https (whether or not Caddy is being used) and no domain is specified, need to provide both
        # key and cert files - they should both already exist on the cluster
        if (
            not parsed_ssl_keyfile
            or not Path(parsed_ssl_keyfile).exists()
            or not parsed_ssl_certfile
            or not Path(parsed_ssl_certfile).exists()
        ):
            # Custom certs should already be on the cluster if their file paths are provided
            raise FileNotFoundError(
                f"Could not find SSL private key and cert files on the cluster, which are required when specifying "
                f"a port ({port_arg}) or enabling HTTPS. Please specify the paths using the --ssl-certfile and "
                f"--ssl-keyfile flags."
            )

    # If the daemon port was not specified, it should be the default RH port
    daemon_port = port_arg
    if not daemon_port or daemon_port in [
        DEFAULT_HTTP_PORT,
        DEFAULT_HTTPS_PORT,
    ]:
        # Since one of HTTP_PORT or HTTPS_PORT was specified, Caddy is set up to forward requests
        # from the daemon to that port
        daemon_port = DEFAULT_SERVER_PORT

    # Note: running the FastAPI app on a higher, non-privileged port (8000) and using Caddy as a reverse
    # proxy to forward requests from port 80 (HTTP) or 443 (HTTPS) to the app's port.
    if use_caddy:
        logger.debug("Using Caddy as a reverse proxy")
        if certs_address is None and domain is None:
            raise ValueError(
                "Must provide the server address or domain to configure Caddy. No address or domain found in the "
                "server start command (--certs-address or --domain) or in the cluster config YAML saved on the cluster."
            )

        cc = CaddyConfig(
            address=certs_address,
            domain=domain,
            rh_server_port=daemon_port,
            ssl_key_path=parsed_ssl_keyfile,
            ssl_cert_path=parsed_ssl_certfile,
            use_https=use_https,
            force_reinstall=restart_proxy,
        )
        cc.configure()

        caddy_port = DEFAULT_HTTPS_PORT if use_https else DEFAULT_HTTP_PORT
        logger.info(f"Caddy is proxying requests from {caddy_port} to {daemon_port}.")

    logger.info(
        f"Launching Runhouse API server with den_auth={den_auth} and "
        + f"on host={host} and use_https={use_https} and port_arg={daemon_port}"
    )

    cluster = None
    if not hasattr(parse_args, "from_python") and default_env_name:
        from runhouse.resources.hardware import Cluster

        cluster = Cluster.from_config(cluster_config)
        cluster._sync_default_env_to_cluster()

    # Only launch uvicorn with certs if HTTPS is enabled and not using Caddy
    uvicorn_cert = parsed_ssl_certfile if not use_caddy and use_https else None
    uvicorn_key = parsed_ssl_keyfile if not use_caddy and use_https else None

    config = uvicorn.Config(
        app,
        host=host,
        port=daemon_port,
        ssl_certfile=uvicorn_cert,
        ssl_keyfile=uvicorn_key,
        loop="uvloop",
    )
    server = uvicorn.Server(config)
    await server.serve()

    if cluster:
        cluster.put_resource(cluster.default_env)

        from runhouse.resources.envs.utils import _process_env_vars

        env_vars = _process_env_vars(cluster.default_env.env_vars)
        if env_vars:
            cluster.call(cluster.default_env.name, "_set_env_vars", env_vars)


if __name__ == "__main__":
    asyncio.run(main())
