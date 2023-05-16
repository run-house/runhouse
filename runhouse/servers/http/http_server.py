import codecs
import json
import logging
import traceback
from pathlib import Path
import subprocess

import ray
from ray import serve
import ray.cloudpickle as pickle
import requests
from sky.skylet.autostop_lib import set_last_active_time_to_now
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.encoders import jsonable_encoder

import runhouse.servers.grpc.unary_pb2 as pb2

from runhouse.rh_config import configs, obj_store
from runhouse.rns.packages.package import Package
from runhouse.rns.run_module_utils import call_fn_by_type
from runhouse.rns.top_level_rns_fns import (
    clear_pinned_memory,
    pinned_keys,
    remove_pinned_object,
)
from runhouse.servers.http.http_utils import Message, Response, pickle_b64, b64_unpickle, OutputType

logger = logging.getLogger(__name__)

app = FastAPI()

@serve.deployment()
@serve.ingress(app)
class HTTPServer:
    DEFAULT_PORT = 50052
    MAX_MESSAGE_LENGTH = 1 * 1024 * 1024 * 1024  # 1 GB
    LOGGING_WAIT_TIME = 1.0
    SKY_YAML = str(Path("~/.sky/sky_ray.yml").expanduser())

    def __init__(self, *args, **kwargs):
        ray.init(address="auto", ignore_reinit_error=True)

        # Collect metadata for the cluster immediately on init
        self._collect_cluster_stats()

        self.register_activity()

    def register_activity(self):
        set_last_active_time_to_now()

    @app.post("/check")
    def check_server(self, message: Message):
        self.register_activity()
        cluster_config = message.data
        try:
            if cluster_config:
                logger.info(f"Message received from client to check server: {cluster_config}")
                rh_dir = Path("~/.rh").expanduser()
                rh_dir.mkdir(exist_ok=True)
                (rh_dir / "cluster_config.yaml").write_text(cluster_config)
                # json.dump(cluster_config, open(rh_dir / "cluster_config.yaml", "w"), indent=4)

            # Check if Ray is deadlocked
            # Get `ray status` from command line
            status = subprocess.check_output(["ray", "status"]).decode("utf-8")
            return Response(data=pickle_b64(status), output_type=OutputType.RESULT)
        except Exception as e:
            logger.exception(e)
            self.register_activity()
            return Response(err=pickle_b64(e),
                            traceback=pickle_b64(traceback.format_exc()),
                            output_type=OutputType.EXCEPTION)

    @app.post("/install")
    def install(self, message: Message):
        self.register_activity()
        try:
            packages, env = b64_unpickle(message.data)
            logger.info(f"Message received from client to install packages: {packages}")
            for package in packages:
                if isinstance(package, str):
                    pkg = Package.from_string(package)

                elif hasattr(package, "install"):
                    pkg = package
                else:
                    raise ValueError(f"package {package} not recognized")

                logger.info(f"Installing package: {str(pkg)}")
                pkg.install(env)

            self.register_activity()
            return Response(output_type=OutputType.SUCCESS)
        except Exception as e:
            logger.exception(e)
            self.register_activity()
            return Response(err=pickle_b64(e),
                            traceback=pickle_b64(traceback.format_exc()),
                            output_type=OutputType.EXCEPTION)

    @app.post("/run")
    def RunModule(self, message: Message):
        self.register_activity()
        # get the function result from the incoming request
        [
            relative_path,
            module_name,
            fn_name,
            fn_type,
            resources,
            conda_env,
            args,
            kwargs,
        ] = b64_unpickle(message.data)

        try:
            result = call_fn_by_type(
                fn_type=fn_type,
                fn_name=fn_name,
                relative_path=relative_path,
                module_name=module_name,
                resources=resources,
                conda_env=conda_env,
                args=args,
                kwargs=kwargs,
                serialize_res=True,
            )
            # We need to pin the obj_ref in the server's Python context rather than inside the call_fn context,
            # because the call_fn context is a separate process and the pinned object will be lost when Ray
            # garbage collects the call_fn process.
            if fn_type == "remote":
                (run_key, obj_ref) = result
                obj_store.put_obj_ref(key=run_key, obj_ref=obj_ref)
                result = pickle.dumps(run_key)

            self.register_activity()
            if isinstance(result, list):
                return Response(data=[codecs.encode(i, "base64").decode() for i in result],
                                output_type=OutputType.RESULT_LIST)
            else:
                return Response(data=codecs.encode(result, "base64").decode(),
                                output_type=OutputType.RESULT)
        except Exception as e:
            logger.exception(e)
            self.register_activity()
            return Response(err=pickle_b64(e),
                            traceback=pickle_b64(traceback.format_exc()),
                            output_type=OutputType.EXCEPTION)

    @app.get("/get_object")
    def GetObject(self, message: Message):
        self.register_activity()
        key, stream_logs = b64_unpickle(message.data)
        logger.info(f"Message received from client to get object: {key}")

        # ret_obj = obj_store.get(key, timeout=self.LOGGING_WAIT_TIME)
        # return json.dumps(jsonable_encoder(Response(
        #     message=ret_obj,
        #     output_type=OutputType.RESULT)))
        return StreamingResponse(self._get_object_and_logs_generator(key, stream_logs=stream_logs),
                                 media_type='application/json')

    def _get_object_and_logs_generator(self, key, stream_logs=False):
        logfiles = None
        open_files = None
        ret_obj = None
        err = None
        tb = None
        returned = False
        while not returned:
            try:
                ret_obj = obj_store.get(key, timeout=self.LOGGING_WAIT_TIME)
                logger.info(f"Got object of type {type(ret_obj)} back from object store")
                returned = True
                # Don't return yet, go through the loop once more to get any remaining log lines
            except ray.exceptions.GetTimeoutError:
                pass
            except ray.exceptions.TaskCancelledError as e:
                logger.info(f"Attempted to get task {key} that was cancelled.")
                returned = True
                ret_obj = None
                err = e
                tb = traceback.format_exc()
            except Exception as e:
                yield json.dumps(jsonable_encoder(Response(
                    error=pickle_b64(e),
                    traceback=pickle_b64(traceback.format_exc()),
                    output_type=OutputType.EXCEPTION)))

            if stream_logs:
                if not logfiles:
                    logfiles = obj_store.get_logfiles(key)
                    open_files = [open(i, "r") for i in logfiles]
                    logger.info(f"Streaming logs for {key} from {logfiles}")

                # Grab all the lines written to all the log files since the last time we checked
                ret_lines = []
                for i, f in enumerate(open_files):
                    file_lines = f.readlines()
                    if file_lines:
                        # TODO [DG] handle .out vs .err, and multiple workers
                        # if len(logfiles) > 1:
                        #     ret_lines.append(f"Process {i}:")
                        ret_lines += file_lines
                if ret_lines:
                    yield json.dumps(jsonable_encoder(Response(
                        data=ret_lines,
                        output_type=OutputType.STDOUT,
                    )))

        if stream_logs:
            # We got the object back from the object store, so we're done (but we went through the loop once
            # more to get any remaining log lines)
            [f.close() for f in open_files]
        if ret_obj:
            yield json.dumps(jsonable_encoder(Response(
                data=codecs.encode(ret_obj, "base64").decode(),
                output_type=OutputType.RESULT)))
        else:
            yield json.dumps(jsonable_encoder(Response(
                error=pickle_b64(err),
                traceback=pickle_b64(tb),
                output_type=OutputType.EXCEPTION)))

    def PutObject(self, request, context):
        self.register_activity()
        key, obj = pickle.loads(request.message)
        logger.info(f"Message received from client to put object: {key}")
        try:
            obj_store.put(key, obj)
            ret_obj = [key, None, None]
        except Exception as e:
            logger.error(f"Error putting object {key} in object store: {e}")
            ret_obj = [None, e, traceback.format_exc()]
        return pb2.MessageResponse(message=pickle.dumps(ret_obj), received=True)

    def ClearPins(self, request, context):
        self.register_activity()
        pins_to_clear = pickle.loads(request.message)
        logger.info(
            f"Message received from client to clear pins: {pins_to_clear or 'all'}"
        )
        cleared = []
        if pins_to_clear:
            for pin in pins_to_clear:
                remove_pinned_object(pin)
                cleared.append(pin)
        else:
            cleared = list(pinned_keys())
            clear_pinned_memory()

        self.register_activity()
        return pb2.MessageResponse(message=pickle.dumps(cleared), received=True)

    def CancelRun(self, request, context):
        self.register_activity()
        run_keys, force, all = pickle.loads(request.message)
        if all:
            # Cancel all runs
            run_keys = obj_store.keys()
        elif not hasattr(run_keys, "len"):
            run_keys = [run_keys]

        for obj_ref in obj_store.get_obj_refs_list(run_keys):
            obj_store.cancel(obj_ref)

        if all:
            obj_store.clear()

        return pb2.MessageResponse(
            message=pickle.dumps("Cancelled"),
            received=True,
            output_type=OutputType.RESULT,
        )

    def ListKeys(self, request, context):
        self.register_activity()
        keys: list = obj_store.keys()
        return pb2.MessageResponse(
            message=pickle.dumps(keys), received=True, output_type=OutputType.RESULT
        )

    def AddSecrets(self, request, context):
        from runhouse import Secrets

        self.register_activity()
        secrets_to_add: dict = pickle.loads(request.message)
        failed_providers = (
            {}
        )  # Track which providers fail and send them back to the user
        for provider_name, provider_secrets in secrets_to_add.items():
            p = Secrets.builtin_provider_class_from_name(provider_name)
            if p is None:
                error_msg = f"{provider_name} is not a Runhouse builtin provider"
                failed_providers[provider_name] = error_msg
                continue

            # NOTE: For now we are always saving in the provider's default location on the cluster
            credentials_path = p.default_credentials_path()
            try:
                p.save_secrets(provider_secrets, overwrite=True)
            except Exception as e:
                failed_providers[provider_name] = str(e)
                continue

            # update config on the cluster with the default creds path for each provider
            configs.set_nested("secrets", {provider_name: credentials_path})
            logger.info(f"Added secrets for {provider_name} to: {credentials_path}")

        return pb2.MessageResponse(
            message=pickle.dumps(failed_providers),
            received=True,
            output_type=OutputType.RESULT,
        )

    def _collect_cluster_stats(self):
        """Collect cluster metadata and send to Grafana Loki"""
        if configs.get("disable_data_collection") is True:
            return

        cluster_data = self._cluster_status_report()
        sky_data = self._cluster_sky_report()

        self._log_cluster_data(
            {**cluster_data, **sky_data},
            labels={"username": configs.get("username"), "environment": "prod"},
        )

    def _cluster_status_report(self):
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

    def _cluster_sky_report(self):
        try:
            from runhouse import Secrets

            sky_ray_data = Secrets.read_yaml_file(self.SKY_YAML)
        except FileNotFoundError:
            # For on prem clusters we won't have sky data
            return {}

        provider = sky_ray_data["provider"]
        node_config = sky_ray_data["available_node_types"].get("ray.head.default", {})

        return {
            "cluster_name": sky_ray_data.get("cluster_name"),
            "region": provider.get("region"),
            "provider": provider.get("module"),
            "instance_type": node_config.get("node_config", {}).get("InstanceType"),
        }

    def _log_cluster_data(self, data: dict, labels: dict):
        from runhouse.rns.api_utils.utils import log_timestamp

        payload = {
            "streams": [
                {"stream": labels, "values": [[str(log_timestamp()), json.dumps(data)]]}
            ]
        }

        payload = json.dumps(payload)
        resp = requests.post(
            f"{configs.get('api_server_url')}/admin/logs", data=json.dumps(payload)
        )

        if resp.status_code == 405:
            # api server not configured to receive grafana logs
            return

        if resp.status_code != 200:
            logger.error(
                f"({resp.status_code}) Failed to send logs to Grafana Loki: {resp.text}"
            )

server = HTTPServer.bind()
