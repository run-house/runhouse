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

from runhouse.constants import (
    CLUSTER_CONFIG_PATH,
    DEFAULT_HTTP_PORT,
    DEFAULT_HTTPS_PORT,
    DEFAULT_SERVER_HOST,
    DEFAULT_SERVER_PORT,
    LOGGING_WAIT_TIME,
    RH_LOGFILE_PATH,
)
from runhouse.globals import configs, obj_store, rns_client
from runhouse.rns.utils.api import resolve_absolute_path
from runhouse.rns.utils.names import _generate_default_name
from runhouse.servers.http.auth import hash_token, verify_cluster_access
from runhouse.servers.http.certs import TLSCertConfig
from runhouse.servers.http.http_utils import (
    DeleteObjectParams,
    get_token_from_request,
    handle_exception_response,
    load_current_cluster_rns_address,
    Message,
    OutputType,
    pickle_b64,
    PutObjectParams,
    PutResourceParams,
    RenameObjectParams,
    Response,
    ServerSettings,
)
from runhouse.servers.nginx.config import NginxConfig
from runhouse.servers.obj_store import (
    ClusterServletSetupOption,
    ObjStore,
    RaySetupOption,
)

logger = logging.getLogger(__name__)

app = FastAPI()


def validate_cluster_access(func):
    """If using Den auth, validate the user's Runhouse token and access to the cluster before continuing."""

    @wraps(func)
    async def wrapper(*args, **kwargs):
        HTTPServer.register_activity()

        request: Request = kwargs.get("request")
        den_auth_enabled: bool = HTTPServer.get_den_auth()
        is_coro = inspect.iscoroutinefunction(func)

        func_call: bool = func.__name__ in ["call_module_method", "call", "get_call"]
        token = get_token_from_request(request)

        if func_call and token:
            obj_store.add_user_to_auth_cache(token, refresh_cache=False)

        # The logged-in user always has full access to the cluster. This is especially important if they flip on
        # Den Auth without saving the cluster.
        # If this is a func call, we'll handle the auth in the object store.
        if not den_auth_enabled or func_call or (token and configs.token == token):
            if is_coro:
                return await func(*args, **kwargs)

            return func(*args, **kwargs)

        if token is None:
            raise HTTPException(
                status_code=404,
                detail="No token found in request auth headers. Expected in "
                f"format: {json.dumps({'Authorization': 'Bearer <token>'})}",
            )

        cluster_uri = load_current_cluster_rns_address()
        if cluster_uri is None:
            logger.error(
                f"Failed to load cluster RNS address. Make sure cluster config YAML has been saved "
                f"on the cluster in path: {CLUSTER_CONFIG_PATH}"
            )
            raise HTTPException(
                status_code=403,
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
    SKY_YAML = str(Path("~/.sky/sky_ray.yml").expanduser())
    memory_exporter = None

    def __init__(
        self,
        conda_env=None,
        enable_local_span_collection=None,
        from_test: bool = False,
        *args,
        **kwargs,
    ):
        runtime_env = {"conda": conda_env} if conda_env else {}

        # If enable_local_span_collection flag is passed, setup the span exporter and related functionality
        if enable_local_span_collection:
            from opentelemetry import trace
            from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
            from opentelemetry.instrumentation.requests import RequestsInstrumentor
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import SimpleSpanProcessor
            from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
                InMemorySpanExporter,
            )

            trace.set_tracer_provider(
                TracerProvider(
                    resource=Resource.create(
                        {"service.name": "runhouse-in-memory-service"}
                    )
                )
            )
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

        # Ray and ClusterServlet should already be
        # initialized by the start script (see below)
        # But if the HTTPServer was started standalone in a test,
        # We still want to make sure the cluster servlet is initialized
        if from_test:
            obj_store.initialize("base", setup_ray=RaySetupOption.TEST_PROCESS)

        # TODO disabling due to latency, figure out what to do with this
        # try:
        #     # Collect metadata for the cluster immediately on init
        #     self._collect_cluster_stats()
        # except Exception as e:
        #     logger.error(f"Failed to collect cluster stats: {str(e)}")

        try:
            # Collect telemetry stats for the cluster
            self._collect_telemetry_stats()
        except Exception as e:
            logger.error(f"Failed to collect cluster telemetry stats: {str(e)}")

        # We initialize a base env servlet where some things may run.
        # TODO: We aren't sure _exactly_ where this is or isn't used.
        # There are a few spots where we do `env_name or "base"`, and
        # this allows that base env to be pre-initialized.
        _ = ObjStore.get_env_servlet(
            env_name="base",
            create=True,
            runtime_env=runtime_env,
        )

        HTTPServer.register_activity()

    @classmethod
    def get_den_auth(cls):
        return obj_store.get_cluster_config().get("den_auth", False)

    @classmethod
    def enable_den_auth(cls, flush=True):
        obj_store.set_cluster_config_value("den_auth", True)
        if flush:
            obj_store.clear_auth_cache()

    @classmethod
    def disable_den_auth(cls):
        obj_store.set_cluster_config_value("den_auth", False)

    @staticmethod
    def register_activity():
        try:
            from sky.skylet.autostop_lib import set_last_active_time_to_now

            set_last_active_time_to_now()
        except ImportError:
            pass

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
                env = env or obj_store.get_env_servlet_name_for_key(lookup_env_for_name)
            servlet = ObjStore.get_env_servlet(env or "base", create=create)
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
    @app.post("/settings")
    @validate_cluster_access
    def update_settings(request: Request, message: ServerSettings) -> Response:
        if message.den_auth:
            HTTPServer.enable_den_auth(flush=message.flush_auth_cache)
        elif message.den_auth is not None and not message.den_auth:
            HTTPServer.disable_den_auth()

        return Response(output_type=OutputType.SUCCESS)

    @staticmethod
    @app.post("/resource")
    @validate_cluster_access
    def put_resource(request: Request, params: PutResourceParams):
        try:
            env_name = params.env_name or "base"
            return obj_store.put_resource(
                serialized_data=params.serialized_data,
                serialization=params.serialization,
                env_name=env_name,
            )
        except Exception as e:
            return handle_exception_response(e, traceback.format_exc())

    @staticmethod
    @app.post("/{module}/{method}")
    @validate_cluster_access
    def call_module_method(
        request: Request, module, method=None, message: dict = Body(default=None)
    ):
        token = get_token_from_request(request)
        den_auth_enabled = HTTPServer.get_den_auth()
        token_hash = hash_token(token) if den_auth_enabled and token else None
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
            env = message.env or obj_store.get_env_servlet_name_for_key(module)
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
                    [module, method, message, token_hash, den_auth_enabled],
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
    def _get_results_and_logs_generator(
        key, env, stream_logs, remote=False, pop=False, serialization=None
    ):
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
                    ret_val = ray.get(obj_ref, timeout=LOGGING_WAIT_TIME)
                    # Last result in a stream will have type RESULT to indicate the end
                    if ret_val is None:
                        # Still waiting for results in queue
                        obj_ref = None
                        # time.sleep(LOGGING_WAIT_TIME)
                        raise ray.exceptions.GetTimeoutError
                    if not ret_val.output_type == OutputType.RESULT_STREAM:
                        waiting_for_results = False
                    ret_val = ret_val.data if serialization == "json" else ret_val
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
                    lines_resp = (
                        Response(
                            data=ret_lines,
                            output_type=OutputType.STDOUT,
                        )
                        if not serialization == "json"
                        else ret_lines
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

    @staticmethod
    @app.post("/object")
    @validate_cluster_access
    def put_object(request: Request, params: PutObjectParams):
        try:
            obj_store.put(
                key=params.key,
                value=params.serialized_data,
                env=params.env_name,
                serialization=params.serialization,
                create_env_if_not_exists=True,
            )
            return Response(output_type=OutputType.SUCCESS)
        except Exception as e:
            return handle_exception_response(e, traceback.format_exc())

    @staticmethod
    @app.post("/rename")
    @validate_cluster_access
    def rename_object(request: Request, params: RenameObjectParams):
        try:
            obj_store.rename(
                old_key=params.key,
                new_key=params.new_key,
            )
            return Response(output_type=OutputType.SUCCESS)
        except Exception as e:
            return handle_exception_response(e, traceback.format_exc())

    @staticmethod
    @app.post("/delete_object")
    @validate_cluster_access
    def delete_obj(request: Request, params: DeleteObjectParams):
        try:
            if len(params.keys) == 0:
                cleared = obj_store.keys()
                obj_store.clear()
            else:
                cleared = []
                for key in params.keys:
                    obj_store.delete(key)
                    cleared.append(key)

            # Expicitly tell the client not to attempt to deserialize the output
            return Response(
                data=cleared,
                output_type=OutputType.RESULT_SERIALIZED,
                serialization=None,
            )
        except Exception as e:
            return handle_exception_response(e, traceback.format_exc())

    @staticmethod
    @app.get("/keys")
    @validate_cluster_access
    def get_keys(request: Request, env_name: Optional[str] = None):
        try:
            if not env_name:
                output = obj_store.keys()
            else:
                output = ObjStore.keys_for_env_servlet_name(env_name)

            # Expicitly tell the client not to attempt to deserialize the output
            return Response(
                data=output,
                output_type=OutputType.RESULT_SERIALIZED,
                serialization=None,
            )
        except Exception as e:
            return handle_exception_response(e, traceback.format_exc())

    @staticmethod
    @app.get("/{module}/{method}")
    @validate_cluster_access
    def get_call(request: Request, module, method=None, serialization="json"):
        token = get_token_from_request(request)
        den_auth_enabled = HTTPServer.get_den_auth()
        token_hash = hash_token(token) if den_auth_enabled and token else None
        # Stream the logs and result (e.g. if it's a generator)
        HTTPServer.register_activity()
        try:
            kwargs = dict(request.query_params)
            kwargs.pop("serialization", None)
            method = None if method == "None" else method
            message = Message(stream_logs=True, data=kwargs)
            env = obj_store.get_env_servlet_name_for_key(module)
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
                    [
                        module,
                        method,
                        message,
                        token_hash,
                        den_auth_enabled,
                        serialization,
                    ],
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
                if not obj_store.contains(message.key):
                    return Response(output_type=OutputType.NOT_FOUND, data=message.key)

            if message.run_async:
                return (
                    Response(
                        data=pickle_b64(message.key),
                        output_type=OutputType.RESULT,
                    )
                    if not serialization == "json"
                    else message.key
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
                error=pickle_b64(e) if not serialization == "json" else str(e),
                traceback=pickle_b64(traceback.format_exc())
                if not serialization == "json"
                else str(traceback.format_exc()),
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
        den_auth_enabled = HTTPServer.get_den_auth()
        if query_params:
            kwargs.update(query_params)
        token = get_token_from_request(request)
        token_hash = hash_token(token) if den_auth_enabled and token else None
        resp = HTTPServer.call_in_env_servlet(
            "call",
            [module, method, args, kwargs, serialization, token_hash, den_auth_enabled],
            create=True,
            lookup_env_for_name=module,
        )

        return JSONResponse(content=resp)

    @staticmethod
    @app.get("/status")
    @validate_cluster_access
    def get_status(request: Request):
        return obj_store.get_status()

    @staticmethod
    def _collect_cluster_stats():
        """Collect cluster metadata and send to Grafana Loki"""
        if configs.get("disable_data_collection") is True:
            return

        cluster_data = HTTPServer._cluster_status_report()
        sky_data = HTTPServer._cluster_sky_report()

        HTTPServer._log_cluster_data(
            {**cluster_data, **sky_data},
            labels={"username": configs.username, "environment": "prod"},
        )

    @staticmethod
    @app.middleware("http")
    async def _add_username_to_span(request: Request, call_next):
        from opentelemetry import trace

        span = trace.get_current_span()
        username = configs.get("username")

        # Set the username as a span attribute
        span.set_attribute("username", username)
        return await call_next(request)

    @staticmethod
    def _collect_telemetry_stats():
        """Collect telemetry stats and send them to the Runhouse hosted OpenTelemetry collector"""
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.instrumentation.requests import RequestsInstrumentor
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        telemetry_collector_address = configs.get("telemetry_collector_address")

        logger.info(f"Preparing to send telemetry to {telemetry_collector_address}")

        # Set the tracer provider and the exporter
        import runhouse

        service_version = runhouse.__version__
        if telemetry_collector_address == "https://api-dev.run.house:14318":
            service_name = "runhouse-service-dev"
            deployment_env = "dev"
        else:
            service_name = "runhouse-service-prod"
            deployment_env = "prod"
        trace.set_tracer_provider(
            TracerProvider(
                resource=Resource.create(
                    {
                        "service.namespace": "Runhouse_OSS",
                        "service.name": service_name,
                        "service.version": service_version,
                        "deployment.environment": deployment_env,
                    }
                )
            )
        )
        otlp_exporter = OTLPSpanExporter(
            endpoint=telemetry_collector_address + "/v1/traces",
        )

        # Add the exporter to the tracer provider
        trace.get_tracer_provider().add_span_processor(
            BatchSpanProcessor(otlp_exporter)
        )

        logger.info(
            f"Successfully added telemetry exporter {telemetry_collector_address}"
        )

        # Instrument the app object
        FastAPIInstrumentor.instrument_app(app)

        # Instrument the requests library
        RequestsInstrumentor().instrument()

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

    ssl_keyfile, ssl_certfile = None, None

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
        help="Port to run daemon on on. If provided and nginx is not enabled, "
        f"will attempt to run the daemon on this port, defaults to {DEFAULT_SERVER_PORT}",
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

    parse_args = parser.parse_args()

    conda_name = parse_args.conda_env
    use_https = parse_args.use_https
    restart_proxy = parse_args.restart_proxy
    use_nginx = parse_args.use_nginx

    # The object store and the cluster servlet within it need to be
    # initiailzed in order to call `obj_store.get_cluster_config()`, which
    # uses the object store to load the cluster config from Ray.
    # When setting up the server, we always want to create a new ClusterServlet.
    # We only want to forcibly start a Ray cluster if asked.
    # We connect this to the "base" env, which we'll initialize later,
    # so writes to the obj_store within the server get proxied to the "base" env.
    obj_store.initialize(
        "base",
        setup_cluster_servlet=ClusterServletSetupOption.FORCE_CREATE,
    )

    cluster_config = obj_store.get_cluster_config()
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

    # Server port
    if parse_args.port != cluster_config.get("server_port"):
        logger.warning(
            f"CLI provided server port: {parse_args.port} is different from the server port specified in "
            f"cluster_config.json: {cluster_config.get('server_port')}. Prioritizing CLI provided port."
        )

    port_arg = parse_args.port or cluster_config.get("server_port")

    if port_arg is not None:
        cluster_config["server_port"] = port_arg

    # Den auth enabled
    if parse_args.use_den_auth != cluster_config.get("den_auth", False):
        logger.warning(
            f"CLI provided den_auth: {parse_args.use_den_auth} is different from the den_auth specified in "
            f"cluster_config.json: {cluster_config.get('den_auth')}. Prioritizing CLI provided den_auth."
        )

    den_auth = parse_args.use_den_auth or cluster_config.get("den_auth", False)
    cluster_config["den_auth"] = den_auth

    # Telemetry enabled
    if parse_args.use_local_telemetry != cluster_config.get(
        "use_local_telemetry", False
    ):
        logger.warning(
            f"CLI provided use_local_telemetry: {parse_args.use_local_telemetry} is different from the "
            f"use_local_telemetry specified in cluster_config.json: "
            f"{cluster_config.get('use_local_telemetry')}. Prioritizing CLI provided use_local_telemetry."
        )

    use_local_telemetry = parse_args.use_local_telemetry or cluster_config.get(
        "use_local_telemetry", False
    )
    cluster_config["use_local_telemetry"] = use_local_telemetry

    # Keyfile
    if parse_args.ssl_keyfile != cluster_config.get("ssl_keyfile"):
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
    if parse_args.ssl_certfile != cluster_config.get("ssl_certfile"):
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
    if parse_args.host != cluster_config.get("server_host"):
        logger.warning(
            f"CLI provided server_host: {parse_args.host} is different from the server_host specified in "
            f"cluster_config.json: {cluster_config.get('server_host')}. Prioritizing CLI provided server_host."
        )

    host = parse_args.host or cluster_config.get("server_host") or DEFAULT_SERVER_HOST
    cluster_config["server_host"] = host

    # Address in the case we're a TLS server
    if parse_args.certs_address != cluster_config.get("ips", [None])[0]:
        logger.warning(
            f"CLI provided certs_address: {parse_args.certs_address} is different from the certs_address specified in "
            f"cluster_config.json: {cluster_config.get('ips', [None])[0]}. Prioritizing CLI provided certs_address."
        )

    address = parse_args.certs_address or cluster_config.get("ips", [None])[0]
    if address is not None:
        cluster_config["ips"] = [address]
    else:
        cluster_config["ips"] = ["0.0.0.0"]

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

    obj_store.set_cluster_config(cluster_config)
    logger.info("Updated cluster config with parsed argument values.")

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
        cert_config = TLSCertConfig()
        ssl_keyfile = resolve_absolute_path(parsed_ssl_keyfile or cert_config.key_path)
        ssl_certfile = resolve_absolute_path(
            parsed_ssl_certfile or cert_config.cert_path
        )

        if not Path(ssl_keyfile).exists() and not Path(ssl_certfile).exists():
            # If the user has specified a server port and we're not using nginx, then they
            # want to run a TLS server on an arbitrary port. In order to do this,
            # they need to pass their own certs.
            if port_arg is not None and not use_nginx:
                # if using a custom HTTPS port must provide private key file and certs explicitly
                raise FileNotFoundError(
                    f"Could not find SSL private key and cert files on the cluster, which are required when specifying "
                    f"a custom port ({port_arg}). Please specify the paths using the --ssl-certfile and "
                    f"--ssl-keyfile flags."
                )

            cert_config.generate_certs(address=address)
            logger.info(
                f"Generated new self-signed cert and private key files on the cluster in "
                f"paths: {cert_config.cert_path} and {cert_config.key_path}"
            )

    # If the daemon port was not specified, it should be the default RH port
    daemon_port = port_arg
    if not daemon_port or daemon_port in [
        DEFAULT_HTTP_PORT,
        DEFAULT_HTTPS_PORT,
    ]:
        # Since one of HTTP_PORT or HTTPS_PORT was specified, nginx is set up to forward requests
        # from the daemon to that port
        daemon_port = DEFAULT_SERVER_PORT

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
            rh_server_port=daemon_port,
            ssl_key_path=ssl_keyfile,
            ssl_cert_path=ssl_certfile,
            use_https=use_https,
            force_reinstall=restart_proxy,
        )
        nc.configure()

        if use_https and (parsed_ssl_keyfile or parsed_ssl_certfile):
            # reload nginx in case updated certs were provided
            nc.reload()

        nginx_port = DEFAULT_HTTPS_PORT if use_https else DEFAULT_HTTP_PORT
        logger.info(f"Nginx is proxying requests from {nginx_port} to {daemon_port}.")

    logger.info(
        f"Launching Runhouse API server with den_auth={den_auth} and "
        + f"use_local_telemetry={use_local_telemetry} "
        + f"on host={host} and use_https={use_https} and port_arg={daemon_port}"
    )

    # Only launch uvicorn with certs if HTTPS is enabled and not using Nginx
    uvicorn_cert = ssl_certfile if not use_nginx and use_https else None
    uvicorn_key = ssl_keyfile if not use_nginx and use_https else None

    uvicorn.run(
        app,
        host=host,
        port=daemon_port,
        ssl_certfile=uvicorn_cert,
        ssl_keyfile=uvicorn_key,
    )
