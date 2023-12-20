import argparse
import inspect
import json
import logging
import time
import traceback
from functools import wraps
from pathlib import Path
from typing import Optional

import ray
import requests
import yaml
from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, StreamingResponse

from sky.skylet.autostop_lib import set_last_active_time_to_now

from runhouse.globals import configs, env_servlets, rns_client
from runhouse.resources.hardware.utils import _load_cluster_config, CLUSTER_CONFIG_PATH
from runhouse.rns.utils.api import resolve_absolute_path
from runhouse.rns.utils.names import _generate_default_name
from runhouse.servers.http.auth import hash_token, verify_cluster_access
from runhouse.servers.http.certs import TLSCertConfig
from runhouse.servers.http.http_utils import (
    b64_unpickle,
    get_token_from_request,
    load_current_cluster,
    Message,
    OutputType,
    pickle_b64,
    Response,
)
from runhouse.servers.nginx.config import NginxConfig
from runhouse.servers.servlet import EnvServlet

logger = logging.getLogger(__name__)

app = FastAPI()


def validate_cluster_access(func):
    """If using Den auth, validate the user's Runhouse token and access to the cluster before continuing."""

    @wraps(func)
    async def wrapper(*args, **kwargs):
        request: Request = kwargs.get("request")
        use_den_auth: bool = HTTPServer.get_den_auth()
        is_coro = inspect.iscoroutinefunction(func)

        func_call: bool = func.__name__ in ["call_module_method", "call", "get_call"]
        if not use_den_auth or func_call:
            # If this is a func call, we'll handle the auth in the object store
            if is_coro:
                return await func(*args, **kwargs)

            return func(*args, **kwargs)

        token = get_token_from_request(request)
        if token is None:
            raise HTTPException(
                status_code=404,
                detail="No token found in request auth headers. Expected in "
                f"format: {json.dumps({'Authorization': 'Bearer <token>'})}",
            )

        cluster_uri = load_current_cluster()
        if cluster_uri is None:
            logger.error(
                f"Failed to load cluster RNS address. Make sure cluster config YAML has been saved "
                f"on the cluster in path: {CLUSTER_CONFIG_PATH}"
            )
            raise HTTPException(
                status_code=404,
                detail="Failed to load current cluster. Make sure cluster config YAML exists on the cluster.",
            )

        cluster_access = verify_cluster_access(cluster_uri, token)
        if not cluster_access:
            # Must have cluster access for all the non func calls
            # Note: for func calls will be handling the auth in the object store
            raise HTTPException(
                status_code=403,
                detail="Cluster access is required for API",
            )

        if is_coro:
            return await func(*args, **kwargs)

        return func(*args, **kwargs)

    return wrapper


class HTTPServer:
    MAX_MESSAGE_LENGTH = 1 * 1024 * 1024 * 1024  # 1 GB
    LOGGING_WAIT_TIME = 1
    DEFAULT_SERVER_HOST = "0.0.0.0"
    DEFAULT_SERVER_PORT = 32300
    DEFAULT_HTTP_PORT = 80
    DEFAULT_HTTPS_PORT = 443
    SKY_YAML = str(Path("~/.sky/sky_ray.yml").expanduser())
    memory_exporter = None
    _den_auth = False

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
            @validate_cluster_access
            def get_spans(request: Request):
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

    @classmethod
    def get_den_auth(cls):
        return cls._den_auth

    @classmethod
    def enable_den_auth(cls):
        cls._den_auth = True

    @classmethod
    def disable_den_auth(cls):
        cls._den_auth = False

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
    @app.get("/check")
    def check_server():
        try:
            HTTPServer.register_activity()
            if not ray.is_initialized():
                raise Exception("Ray is not initialized, restart the server.")
            logger.info("Server is up.")

            import runhouse

            return {"rh_version": runhouse.__version__}
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
    @validate_cluster_access
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
    @validate_cluster_access
    def call_module_method(
        request: Request, module, method=None, message: dict = Body(default=None)
    ):
        token = get_token_from_request(request)
        token_hash = hash_token(token) if den_auth and token else None
        # Stream the logs and result (e.g. if it's a generator)
        HTTPServer.register_activity()
        try:
            # This translates the json dict into an object that we can access with dot notation, e.g. message.key
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
                    [module, method, message, token_hash, den_auth],
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
    def _get_results_and_logs_generator(
        key, env, stream_logs, remote=False, pop=False, serialization=None
    ):
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
                        [key, remote, True, None, serialization],
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
                    yield ret_resp + "\n"
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
                    yield json.dumps(jsonable_encoder(lines_resp)) + "\n"

        except Exception as e:
            logger.exception(e)
            yield json.dumps(
                jsonable_encoder(
                    Response(
                        error=pickle_b64(e) if not serialization == "json" else str(e),
                        traceback=pickle_b64(traceback.format_exc())
                        if not serialization == "json"
                        else str(traceback.format_exc()),
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
    @validate_cluster_access
    def put_object(request: Request, message: Message):
        return HTTPServer.call_in_env_servlet(
            "put_object", [message.key, message.data], env=message.env, create=True
        )

    @staticmethod
    @app.put("/object")
    @validate_cluster_access
    def rename_object(request: Request, message: Message):
        return HTTPServer.call_in_env_servlet(
            "rename_object", [message], env=message.env, lookup_env_for_name=message.key
        )

    @staticmethod
    @app.delete("/object")
    @validate_cluster_access
    def delete_obj(request: Request, message: Message):
        return HTTPServer.call_in_env_servlet(
            "delete_obj", [message], env=message.env, lookup_env_for_name=message.key
        )

    @staticmethod
    @app.post("/cancel")
    @validate_cluster_access
    def cancel_run(request: Request, message: Message):
        return HTTPServer.call_in_env_servlet(
            "cancel_run", [message], env=message.env, lookup_env_for_name=message.key
        )

    @staticmethod
    @app.get("/keys")
    @validate_cluster_access
    def get_keys(request: Request, env: Optional[str] = None):
        from runhouse.globals import obj_store

        if not env:
            return Response(
                output_type=OutputType.RESULT, data=pickle_b64(obj_store.keys())
            )
        return HTTPServer.call_in_env_servlet("get_keys", [], env=env)

    @staticmethod
    @app.get("/{module}/{method}")
    @validate_cluster_access
    def get_call(request: Request, module, method=None, serialization="json"):
        token = get_token_from_request(request)
        token_hash = hash_token(token) if den_auth and token else None
        # Stream the logs and result (e.g. if it's a generator)
        HTTPServer.register_activity()
        try:
            kwargs = dict(request.query_params)
            kwargs.pop("serialization", None)
            method = None if method == "None" else method
            message = Message(stream_logs=True, data=kwargs)
            env = HTTPServer.lookup_env_for_name(module)
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
                    [module, method, message, token_hash, den_auth, serialization],
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
                    data=pickle_b64(message.key)
                    if not serialization == "json"
                    else message.key,
                    output_type=OutputType.RESULT,
                )

            return StreamingResponse(
                HTTPServer._get_results_and_logs_generator(
                    message.key,
                    env=env,
                    stream_logs=message.stream_logs,
                    remote=message.remote,
                    pop=not persist,
                    serialization=serialization,
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
    @app.post("/call/{module}/{method}")
    @validate_cluster_access
    async def call(
        request: Request,
        module,
        method=None,
        args: dict = Body(default={}),
        serialization="json",
    ):
        kwargs = args.get("kwargs", {})
        args = args.get("args", [])
        query_params = dict(request.query_params)
        query_params.pop("serialization", None)
        if query_params:
            kwargs.update(query_params)
        token = get_token_from_request(request)
        token_hash = hash_token(token) if den_auth and token else None
        resp = HTTPServer.call_in_env_servlet(
            "call",
            [module, method, args, kwargs, serialization, token_hash, den_auth],
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


if __name__ == "__main__":
    import uvicorn

    rh_server_host = HTTPServer.DEFAULT_SERVER_HOST
    rh_server_port = HTTPServer.DEFAULT_SERVER_PORT
    default_http_port = HTTPServer.DEFAULT_HTTP_PORT
    default_https_port = HTTPServer.DEFAULT_HTTPS_PORT
    http_port, https_port, ssl_keyfile, ssl_certfile = None, None, None, None

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help=f"Host to run server on. By default will run on {rh_server_host}",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help=f"Port to run server on. If not provided will run on {default_https_port} if HTTPS is enabled, "
        f"{default_http_port} if HTTP is enabled, and {rh_server_port} if connecting to the server via SSH",
    )
    parser.add_argument(
        "--conda-env", type=str, default=None, help="Conda env to run server in"
    )
    parser.add_argument(
        "--use-local-telemetry",
        action="store_true",  # if providing --use-local-telemetry will be set to True
        help="Enable local telemetry",
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
        help="Reconfigure Nginx",
    )
    parser.add_argument(
        "--use-nginx",
        action="store_true",  # if providing --use-nginx will be set to True
        help="Configure Nginx as a reverse proxy",
    )
    parser.add_argument(
        "--certs-address",
        type=str,
        default=None,
        help="Address to use for generating self-signed certs and enabling HTTPS. (e.g. public IP address)",
    )

    cluster_config = _load_cluster_config()
    parse_args = parser.parse_args()

    conda_name = parse_args.conda_env
    host = parse_args.host
    port = parse_args.port
    use_https = parse_args.use_https
    restart_proxy = parse_args.restart_proxy
    use_nginx = parse_args.use_nginx
    use_local_telemetry = parse_args.use_local_telemetry

    # Update globally inside the module based on the args passed in or the cluster config
    den_auth = parse_args.use_den_auth or cluster_config.get("den_auth")

    ips = cluster_config.get("ips", [])
    address = parse_args.certs_address or ips[0] if ips else None

    # If custom certs are provided explicitly use them
    keyfile_arg = parse_args.ssl_keyfile
    parsed_ssl_keyfile = resolve_absolute_path(keyfile_arg) if keyfile_arg else None

    certfile_arg = parse_args.ssl_certfile
    parsed_ssl_certfile = resolve_absolute_path(certfile_arg) if certfile_arg else None

    HTTPServer(
        conda_env=conda_name,
        enable_local_span_collection=use_local_telemetry,
    )

    if den_auth:
        # Update den auth if enabled - keep as a class attribute to be referenced by the validator decorator
        HTTPServer.enable_den_auth()

    # Custom certs should already be on the cluster if their file paths are provided
    if parsed_ssl_keyfile and not Path(parsed_ssl_keyfile).exists():
        raise FileNotFoundError(
            f"No SSL key file found on cluster in path: {parsed_ssl_keyfile}"
        )

    if parsed_ssl_certfile and not Path(parsed_ssl_certfile).exists():
        raise FileNotFoundError(
            f"No SSL cert file found on cluster in path: {parsed_ssl_certfile}"
        )

    if use_https:
        # If not using nginx and no port is specified use the default RH port
        https_port = port or (default_https_port if use_nginx else rh_server_port)
        logger.info(f"Launching HTTPS server on port: {https_port}.")

        cert_config = TLSCertConfig()
        ssl_keyfile = resolve_absolute_path(parsed_ssl_keyfile or cert_config.key_path)
        ssl_certfile = resolve_absolute_path(
            parsed_ssl_certfile or cert_config.cert_path
        )

        if not Path(ssl_keyfile).exists() and not Path(ssl_certfile).exists():
            if https_port != default_https_port:
                # if using a custom HTTPS port must provide private key file and certs explicitly
                raise FileNotFoundError(
                    f"Could not find SSL private key and cert files on the cluster, which are required when specifying "
                    f"a custom port ({https_port}). Please specify the paths using the --ssl-certfile and "
                    f"--ssl-keyfile flags."
                )

            cert_config.generate_certs(address=address)
            logger.info(
                f"Generated new self-signed cert and private key files on the cluster in "
                f"paths: {cert_config.cert_path} and {cert_config.key_path}"
            )
    else:
        # If not using nginx and no port is specified use the default RH port
        http_port = port or (default_http_port if use_nginx else rh_server_port)
        logger.info(f"Launching HTTP server on port: {http_port}.")

    # Note: running the FastAPI app on a higher, non-privileged port (8000) and using Nginx as a reverse
    # proxy to forward requests from port 80 (HTTP) or 443 (HTTPS) to the app's port.
    if use_nginx:
        logger.info("Configuring Nginx")
        if address is None:
            raise ValueError(
                "Must provide the server address to configure Nginx. No address found in the server "
                "start command (--certs-address) or in the cluster config YAML saved on the cluster."
            )

        nc = NginxConfig(
            address=address,
            rh_server_port=rh_server_port,
            ssl_key_path=ssl_keyfile,
            ssl_cert_path=ssl_certfile,
            use_https=use_https,
            force_reinstall=restart_proxy,
        )
        nc.configure()

        if use_https and (parsed_ssl_keyfile or parsed_ssl_certfile):
            # reload nginx in case updated certs were provided
            nc.reload()

        logger.info("Nginx will forward all traffic to the API server")

    host = host or rh_server_host
    logger.info(
        f"Launching Runhouse API server with den_auth={den_auth} and "
        + f"use_local_telemetry={use_local_telemetry} "
        + f"on host: {host} and port: {rh_server_port}"
    )

    # Only launch uvicorn with certs if HTTPS is enabled and not using Nginx
    uvicorn_cert = ssl_certfile if not use_nginx and use_https else None
    uvicorn_key = ssl_keyfile if not use_nginx and use_https else None

    uvicorn.run(
        app,
        host=host,
        port=rh_server_port,
        ssl_certfile=uvicorn_cert,
        ssl_keyfile=uvicorn_key,
    )
