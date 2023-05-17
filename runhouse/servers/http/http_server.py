import codecs
import json
import logging
import subprocess
import traceback
from pathlib import Path

import ray
import ray.cloudpickle as pickle
import requests
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from fastapi.responses import StreamingResponse
from ray import serve
from sky.skylet.autostop_lib import set_last_active_time_to_now

from runhouse.rh_config import configs, obj_store
from runhouse.rns.packages.package import Package
from runhouse.rns.run_module_utils import call_fn_by_type
from runhouse.rns.top_level_rns_fns import (
    clear_pinned_memory,
    pinned_keys,
    remove_pinned_object,
)
from runhouse.servers.http.http_utils import (
    b64_unpickle,
    Message,
    Args,
    OutputType,
    pickle_b64,
    Response,
)

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
            return Response(data=pickle_b64(status), output_type=OutputType.RESULT)
        except Exception as e:
            logger.exception(e)
            self.register_activity()
            return Response(
                error=pickle_b64(e),
                traceback=pickle_b64(traceback.format_exc()),
                output_type=OutputType.EXCEPTION,
            )

    @app.post("/env")
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
            return Response(
                error=pickle_b64(e),
                traceback=pickle_b64(traceback.format_exc()),
                output_type=OutputType.EXCEPTION,
            )

    @app.post("/run")
    def run_module(self, message: Message):
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
            args = obj_store.get_obj_refs_list(args)
            kwargs = obj_store.get_obj_refs_dict(kwargs)
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
            # We need to pin the run_key in the server's Python context rather than inside the call_fn context,
            # because the call_fn context is a separate process and the pinned object will be lost when Ray
            # garbage collects the call_fn process.
            if fn_type == "remote":
                (run_key, obj_ref) = result
                obj_store.put_obj_ref(key=run_key, obj_ref=obj_ref)
                result = pickle.dumps(run_key)

            self.register_activity()
            if isinstance(result, list):
                return Response(
                    data=[codecs.encode(i, "base64").decode() for i in result],
                    output_type=OutputType.RESULT_LIST,
                )
            else:
                return Response(
                    data=codecs.encode(result, "base64").decode(),
                    output_type=OutputType.RESULT,
                )
        except Exception as e:
            logger.exception(e)
            self.register_activity()
            return Response(
                error=pickle_b64(e),
                traceback=pickle_b64(traceback.format_exc()),
                output_type=OutputType.EXCEPTION,
            )

    @app.get("/object")
    def get_object(self, message: Message):
        self.register_activity()
        key, stream_logs = b64_unpickle(message.data)
        logger.info(f"Message received from client to get object: {key}")

        return StreamingResponse(
            self._get_object_and_logs_generator(key, stream_logs=stream_logs),
            media_type="application/json",
        )

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
                logger.info(
                    f"Got object of type {type(ret_obj)} back from object store"
                )
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
                yield json.dumps(
                    jsonable_encoder(
                        Response(
                            error=pickle_b64(e),
                            traceback=pickle_b64(traceback.format_exc()),
                            output_type=OutputType.EXCEPTION,
                        )
                    )
                )

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
                    yield json.dumps(
                        jsonable_encoder(
                            Response(
                                data=ret_lines,
                                output_type=OutputType.STDOUT,
                            )
                        )
                    )

        if stream_logs:
            # We got the object back from the object store, so we're done (but we went through the loop once
            # more to get any remaining log lines)
            [f.close() for f in open_files]
        if ret_obj:
            if isinstance(ret_obj, bytes):
                ret_serialized = codecs.encode(ret_obj, "base64").decode()
            else:
                ret_serialized = pickle_b64(ret_obj)
            yield json.dumps(
                jsonable_encoder(
                    Response(
                        data=ret_serialized,
                        output_type=OutputType.RESULT,
                    )
                )
            )
        else:
            yield json.dumps(
                jsonable_encoder(
                    Response(
                        error=pickle_b64(err),
                        traceback=pickle_b64(tb),
                        output_type=OutputType.EXCEPTION,
                    )
                )
            )

    @app.put("/object")
    def put_object(self, message: Message):
        self.register_activity()
        # We may not want to deserialize the object here in case the object requires dependencies
        # (to be used inside an env) which aren't present in the BaseEnv.
        key, obj = b64_unpickle(message.data)
        logger.info(f"Message received from client to get object: {key}")
        try:
            obj_store.put(key, obj)
            return Response(output_type=OutputType.SUCCESS)
        except Exception as e:
            logger.exception(e)
            self.register_activity()
            return Response(
                error=pickle_b64(e),
                traceback=pickle_b64(traceback.format_exc()),
                output_type=OutputType.EXCEPTION,
            )

    @app.delete("/object")
    def delete_obj(self, message: Message):
        self.register_activity()
        pins_to_clear = b64_unpickle(message.data)
        logger.info(
            f"Message received from client to clear pins: {pins_to_clear or 'all'}"
        )
        try:
            cleared = []
            if pins_to_clear:
                for pin in pins_to_clear:
                    remove_pinned_object(pin)
                    cleared.append(pin)
            else:
                cleared = list(pinned_keys())
                clear_pinned_memory()
                return Response(data=pickle_b64(cleared), output_type=OutputType.RESULT)
        except Exception as e:
            logger.exception(e)
            self.register_activity()
            return Response(
                error=pickle_b64(e),
                traceback=pickle_b64(traceback.format_exc()),
                output_type=OutputType.EXCEPTION,
            )

    @app.post("/cancel")
    def cancel_run(self, message: Message):
        # Having this be a POST instead of a DELETE on the "run" endpoint is strange, but we're not actually
        # deleting the run, just cancelling it. Maybe we should merge this into get_object to allow streaming logs.
        self.register_activity()
        run_keys, force = b64_unpickle(message.data)
        logger.info(f"Message received from client to cancel runs: {run_keys}")
        cancel_runs = run_keys == "all"
        try:
            if cancel_runs:
                # Cancel cancel_runs runs
                run_keys = obj_store.keys()
            elif not hasattr(run_keys, "len"):
                run_keys = [run_keys]

            for obj_ref in obj_store.get_obj_refs_list(run_keys, resolve=False):
                obj_store.cancel(obj_ref)

            if cancel_runs:
                obj_store.clear()
            return Response(output_type=OutputType.SUCCESS)
        except Exception as e:
            logger.exception(e)
            self.register_activity()
            return Response(
                error=pickle_b64(e),
                traceback=pickle_b64(traceback.format_exc()),
                output_type=OutputType.EXCEPTION,
            )

    @app.get("/keys")
    def get_keys(self):
        self.register_activity()
        keys: list = obj_store.keys()
        return Response(data=pickle_b64(keys), output_type=OutputType.RESULT)

    @app.post("/secrets")
    def add_secrets(self, message: Message):
        from runhouse import Secrets

        self.register_activity()
        secrets_to_add: dict = b64_unpickle(message.data)
        failed_providers = (
            {}
        )  # Track which providers fail and send them back to the user
        try:
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
            return Response(
                data=pickle_b64(failed_providers), output_type=OutputType.RESULT
            )
        except Exception as e:
            logger.exception(e)
            self.register_activity()
            return Response(
                error=pickle_b64(e),
                traceback=pickle_b64(traceback.format_exc()),
                output_type=OutputType.EXCEPTION,
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

    @app.post("/call/{fn_name}")
    def run_module(self, fn_name: str, args: Args):
        self.register_activity()
        from runhouse import function
        fn = function(name=fn_name, dryrun=True)
        return fn(*(args.args or []), **(args.kwargs or {}))


server = HTTPServer.bind()
