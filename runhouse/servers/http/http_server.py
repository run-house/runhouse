import argparse
import json
import logging
import subprocess
import time
import traceback
from functools import wraps
from pathlib import Path
from typing import Optional, Union

import ray
import requests
from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from sky.skylet.autostop_lib import set_last_active_time_to_now

from runhouse.globals import configs, env_servlets, rns_client
from runhouse.resources.hardware.utils import _current_cluster, _get_cluster_from
from runhouse.rns.utils.api import (
    load_resp_content,
    resolve_absolute_path,
    ResourceAccess,
)
from runhouse.rns.utils.names import _generate_default_name
from runhouse.servers.http.http_utils import (
    b64_unpickle,
    Message,
    OutputType,
    pickle_b64,
    Response,
    ServerCache,
    TLSCertConfig,
)
from runhouse.servers.nginx.config import NginxConfig
from runhouse.servers.servlet import EnvServlet

logger = logging.getLogger(__name__)

app = FastAPI()

# **NOTE**: This is a temp in-mem cache, we will move this out into the object store for future permissions support
server_cache = ServerCache()

global den_auth


def validate_user(func):
    """If using an HTTPS server, validate the user's Runhouse token before continuing with the API request execution"""

    @wraps(func)
    async def wrapper(*args, **kwargs):
        request: Request = kwargs.get("request")
        is_https: bool = request.url.scheme == "https"
        func_call: bool = func.__name__ in ["call_module_method", "call"]
        use_den_auth: bool = den_auth

        if not is_https and not use_den_auth:
            # Skip validation if not using HTTPS or not validating the user's token
            return func(*args, **kwargs)

        token = request.headers.get("Authorization", "").split("Bearer ")[-1]
        if not token:
            raise HTTPException(
                status_code=404,
                detail="No token found in request auth headers. Expected in "
                f"format: {json.dumps({'Authorization': 'Bearer <token>'})}",
            )

        # Check if user's token has already been validated and saved to cache on the cluster
        cached_resources: dict = server_cache.get_resources(token)
        if not cached_resources:
            resp = requests.get(
                f"{rns_client.api_server_url}/user",
                headers={"Authorization": f"Bearer {token}"},
            )
            if resp.status_code != 200:
                raise HTTPException(
                    status_code=403,
                    detail=f"Failed to validate Runhouse user: {load_resp_content(resp)}",
                )

        if use_den_auth or func_call:
            # Note: currently providing cluster level access, not at the resource level
            # but will be used in the future for validating access to other resources if provided in the request
            cluster_uri = _load_current_cluster(kwargs)
            if cluster_uri is None:
                logger.error(
                    "Failed to load cluster RNS address. Make sure cluster config YAML has been saved "
                    "on the cluster in path: ~/.rh/cluster_config.yaml"
                )
                raise HTTPException(
                    status_code=403,
                    detail="Failed validate access to cluster",
                )

            verify_resource_access(token, cluster_uri, cached_resources, func_call)

        return func(*args, **kwargs)

    return wrapper


def verify_resource_access(
    token: str, resource_uri: str, cached_resources: dict, func_call: bool
) -> None:
    """Verify the user has access to a particular resource. Use global HTTP Server cache to check before pinging
    Den for the most current list of resources the user has access to."""
    # e.g. {"/jlewitt1/bert-preproc": "read"}
    resource_access_type = cached_resources.get(resource_uri)
    if resource_access_type:
        if func_call and resource_access_type != ResourceAccess.WRITE.value:
            raise HTTPException(
                status_code=403,
                detail=f"Write access is required for resource: {resource_uri}",
            )
        return

    resp = requests.get(
        f"{rns_client.api_server_url}/resource",
        headers={"Authorization": f"Bearer {token}"},
    )
    if resp.status_code != 200:
        raise HTTPException(
            status_code=403,
            detail=f"Failed to load resources: {load_resp_content(resp)}",
        )

    resp_data = json.loads(resp.content)
    all_resources: dict = {
        resource["name"]: resource["access_type"] for resource in resp_data["data"]
    }

    # Update the cache with the latest list of resources
    # Note: This will only persist for as long as the server is up
    server_cache.put_resources(token, all_resources)

    logger.info(f"Updated cache with {len(all_resources)} resources for user")

    if resource_uri not in all_resources:
        raise HTTPException(
            status_code=404,
            detail=f"Could not validate access to resource: {resource_uri}",
        )


def _load_current_cluster(kwargs) -> Union[str, None]:
    current_cluster = _get_cluster_from(_current_cluster("config"))
    if current_cluster:
        return current_cluster.rns_address

    # If no cluster config saved yet on the cluster try getting the cluster uri from the message object
    # included in the request
    message: Message = kwargs.get("message")
    resource = json.loads(message.data)
    return resource.get("name")


class HTTPServer:
    MAX_MESSAGE_LENGTH = 1 * 1024 * 1024 * 1024  # 1 GB
    LOGGING_WAIT_TIME = 1
    DEFAULT_APP_PORT = 32300  # Port where Fast API app is running
    # Allow external devices to access the app by default
    DEFAULT_SERVER_HOST = "0.0.0.0"
    SKY_YAML = str(Path("~/.sky/sky_ray.yml").expanduser())
    memory_exporter = None

    def __init__(
        self, conda_env=None, enable_local_span_collection=None, *args, **kwargs
    ):
        runtime_env = {"conda": conda_env} if conda_env else {}

        # If enable_local_span_collection flag is passed, setup the span exporter and related functionality
        if enable_local_span_collection:
            from opentelemetry import trace
            from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
            from opentelemetry.instrumentation.requests import RequestsInstrumentor
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import SimpleSpanProcessor
            from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
                InMemorySpanExporter,
            )

            trace.set_tracer_provider(TracerProvider())
            self.memory_exporter = InMemorySpanExporter()
            trace.get_tracer_provider().add_span_processor(
                SimpleSpanProcessor(self.memory_exporter)
            )
            # Instrument the app object
            FastAPIInstrumentor.instrument_app(app)

            # Instrument the requests library
            RequestsInstrumentor().instrument()

            @staticmethod
            @app.get("/spans")
            @validate_user
            def get_spans(request: Request):
                print("Calling get_spans")
                return {
                    "spans": [
                        span.to_json()
                        for span in self.memory_exporter.get_finished_spans()
                    ]
                }

        if not ray.is_initialized():
            ray.init(
                ignore_reinit_error=True,
                runtime_env=runtime_env,
                namespace="runhouse",
            )

        try:
            # Collect metadata for the cluster immediately on init
            self._collect_cluster_stats()
        except Exception as e:
            logger.error(f"Failed to collect cluster stats: {str(e)}")

        base_env = self.get_env_servlet(
            env_name="base",
            create=True,
            runtime_env=runtime_env,
        )
        env_servlets["base"] = base_env
        from runhouse.globals import obj_store

        obj_store.set_name("server")

        HTTPServer.register_activity()

    @staticmethod
    def register_activity():
        set_last_active_time_to_now()

    @staticmethod
    @app.get("/cert")
    def get_cert():
        """Download the certificate file for this server necessary for enabling HTTPS.
        User must have access to the cluster in order to download the certificate."""
        try:
            certs_config = TLSCertConfig()
            cert_path = certs_config.cert_path

            if not Path(cert_path).exists():
                raise FileNotFoundError(
                    f"No certificate found on cluster in path: {cert_path}"
                )

            with open(cert_path, "rb") as cert_file:
                cert = cert_file.read()

            return Response(data=pickle_b64(cert), output_type=OutputType.RESULT)

        except Exception as e:
            logger.exception(e)
            return Response(
                error=pickle_b64(e),
                traceback=pickle_b64(traceback.format_exc()),
                output_type=OutputType.EXCEPTION,
            )

    @staticmethod
    @app.post("/check")
    @validate_user
    def check_server(request: Request, message: Message):
        HTTPServer.register_activity()
        cluster_config = message.data
        try:
            if cluster_config:
                logger.info(
                    f"Message received from client to check server: {cluster_config}"
                )
                rh_dir = Path("~/.rh").expanduser()
                rh_dir.mkdir(exist_ok=True)
                (rh_dir / "cluster_config.yaml").write_text(cluster_config)
                # json.dump(cluster_config, open(rh_dir / "cluster_config.yaml", "w"), indent=4)

            # Check if Ray is deadlocked
            # Get `ray status` from command line
            status = subprocess.check_output(["ray", "status"]).decode("utf-8")

            import runhouse

            # Reset here in case it was set before the config was written down, making here=="file"
            runhouse.here = _get_cluster_from(_current_cluster("config"))

            return Response(data=pickle_b64(status), output_type=OutputType.RESULT)
        except Exception as e:
            logger.exception(e)
            HTTPServer.register_activity()
            return Response(
                error=pickle_b64(e),
                traceback=pickle_b64(traceback.format_exc()),
                output_type=OutputType.EXCEPTION,
            )

    @staticmethod
    def get_env_servlet(env_name, create=False, runtime_env=None):
        if env_name in env_servlets.keys():
            return env_servlets[env_name]

        if create:
            new_env = (
                ray.remote(EnvServlet)
                .options(
                    name=env_name,
                    get_if_exists=True,
                    runtime_env=runtime_env,
                    lifetime="detached",
                    namespace="runhouse",
                    max_concurrency=1000,
                )
                .remote(env_name=env_name)
            )
            env_servlets[env_name] = new_env
            return new_env

        else:
            raise Exception(
                f"Environment {env_name} does not exist. Please send it to the cluster first."
            )

    @staticmethod
    def call_servlet_method(servlet, method, args, block=True):
        if isinstance(servlet, ray.actor.ActorHandle):
            obj_ref = getattr(servlet, method).remote(*args)
            if block:
                return ray.get(obj_ref)
            else:
                return obj_ref
        else:
            return getattr(servlet, method)(*args)

    @staticmethod
    def call_in_env_servlet(
        method,
        args=None,
        env=None,
        create=False,
        lookup_env_for_name=None,
        block=True,
    ):
        HTTPServer.register_activity()
        try:
            if lookup_env_for_name:
                env = env or HTTPServer.lookup_env_for_name(lookup_env_for_name)
            servlet = HTTPServer.get_env_servlet(env or "base", create=create)
            # If servlet is a RayActor, call with .remote
            return HTTPServer.call_servlet_method(servlet, method, args, block=block)
        except Exception as e:
            logger.exception(e)
            HTTPServer.register_activity()
            return Response(
                error=pickle_b64(e),
                traceback=pickle_b64(traceback.format_exc()),
                output_type=OutputType.EXCEPTION,
            )

    @staticmethod
    def lookup_env_for_name(name, check_rns=False):
        from runhouse.globals import obj_store

        env = obj_store.get_env(name)
        if env:
            return env

        # Load the resource config from rns and see if it has an "env" field
        if check_rns:
            resource_config = rns_client.load_config(name)
            if resource_config and "env" in resource_config:
                return resource_config["env"]

        return None

    @staticmethod
    @app.post("/resource")
    @validate_user
    def put_resource(request: Request, message: Message):
        # if resource is env and not yet a servlet, construct env servlet
        if message.env and message.env not in env_servlets.keys():
            resource = b64_unpickle(message.data)[0]
            if resource["resource_type"] == "env":
                runtime_env = (
                    {"conda_env": resource["env_name"]}
                    if resource["resource_subtype"] == "CondaEnv"
                    else {}
                )

                new_env = HTTPServer.get_env_servlet(
                    env_name=message.env,
                    create=True,
                    runtime_env=runtime_env,
                )

                env_servlets[message.env] = new_env

        return HTTPServer.call_in_env_servlet(
            "put_resource",
            [message],
            env=message.env,
            create=True,
            lookup_env_for_name=message.key,
        )

    @staticmethod
    @app.post("/{module}/{method}")
    @validate_user
    def call_module_method(
        request: Request, module, method=None, message: dict = Body(...)
    ):
        # Stream the logs and result (e.g. if it's a generator)
        HTTPServer.register_activity()
        try:
            # This translates the json dict into an object that we can can access with dot notation, e.g. message.key
            message = argparse.Namespace(**message) if message else None
            method = None if method == "None" else method
            # If this is a "get" request to just return the module, do not stream logs or save by default
            message = message or (
                Message(stream_logs=False, key=module) if not method else Message()
            )
            env = message.env or HTTPServer.lookup_env_for_name(module)
            persist = message.run_async or message.remote or message.save or not method
            if method:
                # TODO fix the way we generate runkeys, it's ugly
                message.key = message.key or _generate_default_name(
                    prefix=module if method == "__call__" else f"{module}_{method}",
                    precision="ms",  # Higher precision because we see collisions within the same second
                )
                # If certain conditions are met, we can return a response immediately
                fast_resp = not persist and not message.stream_logs

                # Unless we're returning a fast response, we discard this obj_ref
                obj_ref = HTTPServer.call_in_env_servlet(
                    "call_module_method",
                    [module, method, message],
                    env=env,
                    create=True,
                    block=False,
                )

                if fast_resp:
                    res = ray.get(obj_ref)
                    logger.info(f"Returning fast response for {message.key}")
                    return res

            else:
                message.key = module
                # If this is a "get" call, don't wait for the result, it's either there or not.
                from runhouse.globals import obj_store

                if not obj_store.contains(message.key):
                    return Response(output_type=OutputType.NOT_FOUND, data=message.key)

            if message.run_async:
                return Response(
                    data=pickle_b64(message.key),
                    output_type=OutputType.RESULT,
                )

            return StreamingResponse(
                HTTPServer._get_results_and_logs_generator(
                    message.key,
                    env=env,
                    stream_logs=message.stream_logs,
                    remote=message.remote,
                    pop=not persist,
                ),
                media_type="application/json",
            )
        except Exception as e:
            logger.exception(e)
            HTTPServer.register_activity()
            return Response(
                error=pickle_b64(e),
                traceback=pickle_b64(traceback.format_exc()),
                output_type=OutputType.EXCEPTION,
            )

    @staticmethod
    def _get_logfiles(log_key, log_type=None):
        if not log_key:
            return None
        key_logs_path = Path(EnvServlet.RH_LOGFILE_PATH) / log_key
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
    def _get_results_and_logs_generator(key, env, stream_logs, remote=False, pop=False):
        from runhouse.globals import obj_store

        open_logfiles = []
        waiting_for_results = True

        try:
            obj_ref = None
            while waiting_for_results:

                while not obj_store.contains(key):
                    time.sleep(0.1)

                if not obj_ref:
                    obj_ref = HTTPServer.call_in_env_servlet(
                        "get",
                        [key, remote, True],
                        env=env,
                        block=False,
                    )
                try:
                    ret_val = ray.get(obj_ref, timeout=HTTPServer.LOGGING_WAIT_TIME)
                    # Last result in a stream will have type RESULT to indicate the end
                    if ret_val is None:
                        # Still waiting for results in queue
                        obj_ref = None
                        # time.sleep(HTTPServer.LOGGING_WAIT_TIME)
                        raise ray.exceptions.GetTimeoutError
                    if not ret_val.output_type == OutputType.RESULT_STREAM:
                        waiting_for_results = False
                    ret_resp = json.dumps(jsonable_encoder(ret_val))
                    logger.info(f"Yielding response for key {key}")
                    yield ret_resp
                except ray.exceptions.GetTimeoutError:
                    pass

                # Reset the obj_ref so we make a fresh request for the result next time around
                obj_ref = None

                # Grab all the lines written to all the log files since the last time we checked, including
                # any new log files that have been created
                open_logfiles = (
                    HTTPServer.open_new_logfiles(key, open_logfiles)
                    if stream_logs
                    else []
                )
                ret_lines = []
                for i, f in enumerate(open_logfiles):
                    file_lines = f.readlines()
                    if file_lines:
                        # TODO [DG] handle .out vs .err, and multiple workers
                        # if len(logfiles) > 1:
                        #     ret_lines.append(f"Process {i}:")
                        ret_lines += file_lines
                if ret_lines:
                    lines_resp = Response(
                        data=ret_lines,
                        output_type=OutputType.STDOUT,
                    )
                    logger.debug(f"Yielding logs for key {key}")
                    yield json.dumps(jsonable_encoder(lines_resp))

        except Exception as e:
            logger.exception(e)
            yield json.dumps(
                jsonable_encoder(
                    Response(
                        error=pickle_b64(e),
                        traceback=pickle_b64(traceback.format_exc()),
                        output_type=OutputType.EXCEPTION,
                    )
                )
            )
        finally:
            if stream_logs and not open_logfiles:
                logger.warning(f"No logfiles found for call {key}")
            for f in open_logfiles:
                f.close()

            logger.debug(f"Deleting {key}")
            if pop:
                obj_store.delete(key)
                HTTPServer.call_in_env_servlet("delete_obj", [[key], True], env=env)

    @staticmethod
    @app.post("/object")
    @validate_user
    def put_object(request: Request, message: Message):
        return HTTPServer.call_in_env_servlet(
            "put_object", [message.key, message.data], env=message.env, create=True
        )

    @staticmethod
    @app.put("/object")
    @validate_user
    def rename_object(request: Request, message: Message):
        return HTTPServer.call_in_env_servlet(
            "rename_object", [message], env=message.env
        )

    @staticmethod
    @app.delete("/object")
    @validate_user
    def delete_obj(request: Request, message: Message):
        return HTTPServer.call_in_env_servlet(
            "delete_obj", [message], env=message.env, lookup_env_for_name=message.key
        )

    @staticmethod
    @app.post("/cancel")
    @validate_user
    def cancel_run(request: Request, message: Message):
        return HTTPServer.call_in_env_servlet(
            "cancel_run", [message], env=message.env, lookup_env_for_name=message.key
        )

    @staticmethod
    @app.get("/keys")
    @validate_user
    def get_keys(request: Request, env: Optional[str] = None):
        from runhouse.globals import obj_store

        if not env:
            return Response(
                output_type=OutputType.RESULT, data=pickle_b64(obj_store.keys())
            )
        return HTTPServer.call_in_env_servlet("get_keys", [], env=env)

    @staticmethod
    @app.post("/secrets")
    @validate_user
    def add_secrets(request: Request, message: Message):
        return HTTPServer.call_in_env_servlet(
            "add_secrets", [message], env=message.env, create=True
        )

    @staticmethod
    @app.post("/call/{module}/{method}")
    @validate_user
    async def call(
        request: Request,
        module,
        method=None,
        args: dict = Body(),
        serialization="json",
    ):
        kwargs = args.get("kwargs", {})
        args = args.get("args", [])
        resp = HTTPServer.call_in_env_servlet(
            "call",
            [module, method, args, kwargs, serialization],
            create=True,
            lookup_env_for_name=module,
        )

        return JSONResponse(content=resp)

    @staticmethod
    def _collect_cluster_stats():
        """Collect cluster metadata and send to Grafana Loki"""
        if configs.get("disable_data_collection") is True:
            return

        cluster_data = HTTPServer._cluster_status_report()
        sky_data = HTTPServer._cluster_sky_report()

        HTTPServer._log_cluster_data(
            {**cluster_data, **sky_data},
            labels={"username": configs.get("username"), "environment": "prod"},
        )

    @staticmethod
    def _cluster_status_report():
        import ray._private.usage.usage_lib as ray_usage_lib
        from ray._private import gcs_utils

        gcs_client = gcs_utils.GcsClient(
            address="127.0.0.1:6379", nums_reconnect_retry=20
        )

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
            from runhouse import Secrets

            sky_ray_data = Secrets.read_yaml_file(HTTPServer.SKY_YAML)
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


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help=f"Host to run server on. By default will run on {HTTPServer.DEFAULT_SERVER_HOST}",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help=f"Port to run server on. By default will run on {HTTPServer.DEFAULT_APP_PORT}",
    )
    parser.add_argument(
        "--conda-env", type=str, default=None, help="Conda env to run server in"
    )
    parser.add_argument(
        "--enable-local-span-collection",
        type=bool,
        default=None,
        help="Enable local span collection",
    )
    parser.add_argument(
        "--use-https",
        action="store_true",  # if providing --use-https will be set to True
        help="Start an HTTPS server with new TLS certs",
    )
    parser.add_argument(
        "--use-den-auth",
        action="store_true",  # if providing --use-den-auth will be set to True
        help="Whether to authenticate requests with a Runhouse token",
    )
    parser.add_argument(
        "--ssl-keyfile",
        type=str,
        default=None,
        help="Path to SSL key file to use for HTTPS",
    )
    parser.add_argument(
        "--ssl-certfile",
        type=str,
        default=None,
        help="Path to SSL cert file to use for HTTPS",
    )
    parser.add_argument(
        "--force-reinstall",
        action="store_true",  # if providing --force-reinstall will be set to True
        help="Reconfigure Nginx and reinstall certs",
    )
    parser.add_argument(
        "--skip-nginx",
        action="store_true",  # if providing --skip-nginx will be set to True
        help="Do not configure Nginx as a reverse proxy",
    )
    parser.add_argument(
        "--address",
        type=str,
        default=None,
        help="Public IP address of the server. Required for generating self-signed certs and enabling HTTPS",
    )

    parse_args = parser.parse_args()
    host = parse_args.host
    port = parse_args.port
    conda_name = parse_args.conda_env
    should_enable_local_span_collection = parse_args.enable_local_span_collection
    use_https = parse_args.use_https
    den_auth = parse_args.use_den_auth
    parsed_ssl_keyfile = parse_args.ssl_keyfile
    parsed_ssl_certfile = parse_args.ssl_certfile
    force_reinstall = parse_args.force_reinstall
    skip_nginx = parse_args.skip_nginx
    address = parse_args.address

    HTTPServer(
        conda_env=conda_name,
        enable_local_span_collection=should_enable_local_span_collection,
    )

    # Save down certs onto the cluster which is needed for Nginx and relevant when starting server with HTTPS
    if parsed_ssl_keyfile and not Path(parsed_ssl_keyfile).exists():
        raise FileNotFoundError(
            f"No SSL key file found on cluster in path: {parsed_ssl_keyfile}"
        )

    if parsed_ssl_certfile and not Path(parsed_ssl_certfile).exists():
        raise FileNotFoundError(
            f"No SSL cert file found on cluster in path: {parsed_ssl_certfile}"
        )

    cert_config = TLSCertConfig()
    ssl_keyfile = resolve_absolute_path(parsed_ssl_keyfile or cert_config.key_path)
    ssl_certfile = resolve_absolute_path(parsed_ssl_certfile or cert_config.cert_path)

    if not Path(ssl_keyfile).exists() and not Path(ssl_certfile).exists():
        cert_config.generate_certs(address=address)
        logger.info(
            f"Generated new self-signed cert and keyfile on the cluster in paths: {cert_config.cert_path} "
            f"and {cert_config.key_path}"
        )

    host = host or HTTPServer.DEFAULT_SERVER_HOST

    # Note: running the FastAPI app on a higher, non-privileged port (8000) and using Nginx as a reverse
    # proxy to forward requests from port 80 (HTTP) or 443 (HTTPS) to the app's port.
    if use_https:
        https_port = port or HTTPServer.DEFAULT_APP_PORT

        logger.info(
            f"Launching HTTPS server with den_auth={den_auth} on host: {host} and port 443."
        )
        if not skip_nginx:
            nc = NginxConfig(
                app_port=https_port,
                force_reinstall=force_reinstall,
                ssl_key_path=ssl_keyfile,
                ssl_cert_path=ssl_certfile,
                address=address,
            )
            nc.configure()

            if parsed_ssl_keyfile or parsed_ssl_certfile:
                # reload nginx in case the certs were updated
                nc.reload()

            logger.info(
                f"Nginx will forward all traffic to the Runhouse API server running on port {https_port}"
            )

        # https://fastapi.tiangolo.com/advanced/middleware/#httpsredirectmiddleware
        app.add_middleware(HTTPSRedirectMiddleware)

        uvicorn.run(
            app,
            host=host,
            port=https_port,
            ssl_keyfile=ssl_keyfile,
            ssl_certfile=ssl_certfile,
        )

    else:
        http_port = port or HTTPServer.DEFAULT_APP_PORT

        logger.info(
            f"Launching HTTP server with den_auth={den_auth} on host: {host} and port 80."
        )
        if not skip_nginx:
            NginxConfig(
                app_port=http_port,
                force_reinstall=force_reinstall,
                ssl_key_path=ssl_keyfile,
                ssl_cert_path=ssl_certfile,
                address=address,
            ).configure()
            logger.info(
                f"Nginx will forward all traffic to the Runhouse API server running on port {http_port}"
            )

        uvicorn.run(app, host=host, port=http_port)
