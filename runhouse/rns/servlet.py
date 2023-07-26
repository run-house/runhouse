import codecs
import inspect
import logging
import queue
import traceback
from pathlib import Path

import ray
import ray.cloudpickle as pickle
from sky.skylet.autostop_lib import set_last_active_time_to_now

from runhouse.rh_config import configs, obj_store

from runhouse.rns.blobs import blob
from runhouse.rns.packages.package import Package
from runhouse.rns.queues import Queue
from runhouse.rns.resource import Resource
from runhouse.rns.run import run
from runhouse.rns.run_module_utils import call_fn_by_type
from runhouse.rns.utils.names import _generate_default_name
from runhouse.servers.http.http_utils import (
    b64_unpickle,
    Message,
    OutputType,
    pickle_b64,
    Response,
)

logger = logging.getLogger(__name__)


class EnvServlet:
    DEFAULT_PORT = 50052
    MAX_MESSAGE_LENGTH = 1 * 1024 * 1024 * 1024  # 1 GB
    LOGGING_WAIT_TIME = 1.0
    SKY_YAML = str(Path("~/.sky/sky_ray.yml").expanduser())
    LOGS_DIR = ".rh/logs"
    RH_LOGFILE_PATH = Path.home() / LOGS_DIR

    def __init__(self, env_name, *args, **kwargs):
        self.env_name = env_name

        obj_store.set_name(self.env_name)

        self.results_streams = {}

    @staticmethod
    def register_activity():
        set_last_active_time_to_now()

    def install(self, message: Message):
        self.register_activity()
        try:
            packages, env = b64_unpickle(message.data)
            logger.info(f"Message received from client to install packages: {packages}")
            for package in packages:
                if isinstance(package, str):
                    pkg = Package.from_string(package)

                elif hasattr(package, "_install"):
                    pkg = package
                else:
                    raise ValueError(f"package {package} not recognized")

                logger.info(f"Installing package: {str(pkg)}")
                pkg._install(env)

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

    def put_resource(self, message: Message):
        self.register_activity()
        try:
            resource_config, state, dryrun = b64_unpickle(message.data)
            # Resolve any sub-resources which are string references to resources already sent to this cluster
            resource_config = obj_store.get_obj_refs_dict(resource_config)
            logger.info(
                f"Message received from client to construct resource: {resource_config}"
            )

            resource = Resource.from_config(config=resource_config, dryrun=dryrun)

            for attr, val in state.items():
                setattr(resource, attr, val)

            if not resource.name and message.key:
                resource.name = message.key
            name = resource.name or _generate_default_name(
                prefix=resource.RESOURCE_TYPE
            )

            if hasattr(resource, "remote_init"):
                logger.info(
                    f"Initializing module {resource.name} in env servlet {self.env_name}"
                )
                resource.remote_init()

            obj_store.put(name, resource)
            self.register_activity()
            # Return the name in case we had to set it
            return Response(output_type=OutputType.RESULT, data=pickle_b64(name))
        except Exception as e:
            logger.exception(e)
            self.register_activity()
            return Response(
                error=pickle_b64(e),
                traceback=pickle_b64(traceback.format_exc()),
                output_type=OutputType.EXCEPTION,
            )

    def call_module_method(self, module_name, method_name, message: Message):
        self.register_activity()
        callable_method = None
        run_obj = None

        try:
            self.results_streams[message.key] = Queue()
            args, kwargs = b64_unpickle(message.data) if message.data else ([], {})
            logger.info(
                f"Message received from client to call method {method_name} on module {module_name}"
            )
            module = obj_store.get(module_name, None)
            if not module:
                raise ValueError(f"Resource {module_name} not found")

            # If method_name is None, return the module itself as this is a "get" request
            try:
                method = getattr(module, method_name) if method_name else module
            except AttributeError:
                logger.info(module.__dict__)
                raise ValueError(
                    f"Method {method_name} not found on module {module_name}"
                )

            # Don't call the method if it's a property or a "get" request (returning the module itself)
            if hasattr(method, "__call__") and method_name:
                # If method is callable, call it and return the result
                logger.info(
                    f"{self.env_name} servlet: Calling method {method_name} on module {module_name}"
                )
                callable_method = True
            else:
                # Method is a property, return the value
                logger.info(
                    f"Env {self.env_name} servlet: Getting property {method_name} on module {module_name}"
                )
                callable_method = False

            if not callable_method and kwargs and "new_value" in kwargs:
                # If new_value was passed, that means we're setting a property
                setattr(module, method_name, kwargs["new_value"])
                resp = Response(output_type=OutputType.SUCCESS)
                self.results_streams[message.key].put(resp)
                return

            if callable_method:
                # Only create a run for callable methods, not properties
                run_obj = run(name=message.key, load=False)
                run_obj.__enter__()

            # If method is a property, `method = getattr(module, method_name, None)` above already
            # got our result
            result = method(*args, **kwargs) if callable_method else method
            if inspect.isgenerator(result):
                # Stream back the results of the generator
                logger.info(
                    f"Streaming back results of generator {module_name}.{method_name}"
                )
                results_to_save = []
                resp = None
                for val in result:
                    self.register_activity()
                    # Doing this at the top of the loop so we can catch the final result and change the OutputType
                    if resp is not None:
                        self.results_streams[message.key].put(resp)
                    results_to_save = results_to_save + [val] if message.save else []
                    resp = Response(
                        output_type=OutputType.RESULT_STREAM, data=pickle_b64(val)
                    )
                # Change type of final result to RESULT to indicate end of stream
                resp.output_type = OutputType.RESULT
                self.results_streams[message.key].put(resp)
                if callable_method:
                    run_obj.__exit__(None, None, None)

                if message.save or message.remote:
                    # TODO save queue inside loop instead?
                    b = blob(data=results_to_save)
                    b.provenance = run_obj
                    b.save(message.key)
            else:
                if callable_method:
                    run_obj.__exit__(None, None, None)

                if message.save or message.remote:
                    if isinstance(result, Resource):
                        result.provenance = run_obj
                        result.save(message.key)
                    else:
                        b = blob(data=result)
                        b.provenance = run_obj
                        b.save(message.key)
                self.register_activity()
                resp = Response(output_type=OutputType.RESULT, data=pickle_b64(result))
                self.results_streams[message.key].put(resp)
        except Exception as e:
            logger.exception(e)
            self.register_activity()
            resp = Response(
                error=pickle_b64(e),
                traceback=pickle_b64(traceback.format_exc()),
                output_type=OutputType.EXCEPTION,
            )
            self.results_streams[message.key].put(resp)

    def _get_result_from_stream(self, key, block=True, timeout=None):
        # TODO return queue.Empty instead of None
        self.register_activity()
        if key not in self.results_streams:
            return None
        try:
            return self.results_streams[key].get(block=block, timeout=timeout)
        except queue.Empty:
            return None

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
            env_vars,
            run_name,
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
                env_vars=env_vars,
                run_name=run_name,
                args=args,
                kwargs=kwargs,
                serialize_res=True,
            )
            # We need to pin the run_key in the server's Python context rather than inside the call_fn context,
            # because the call_fn context is a separate process and the pinned object will be lost when Ray
            # garbage collects the call_fn process.
            from runhouse import Run

            (res, obj_ref, run_key) = result

            if obj_ref is not None:
                obj_store.put_obj_ref(key=run_key, obj_ref=obj_ref)

            result = pickle.dumps(res) if isinstance(res, Run) else res

            self.register_activity()
            if isinstance(result, ray.exceptions.RayTaskError):
                # If Ray throws an error when executing the function as part of a Run,
                # it will be reflected in the result since we catch the exception and do not immediately raise it
                logger.exception(result)
                return Response(
                    error=pickle_b64(result),
                    traceback=pickle_b64(traceback.format_exc()),
                    output_type=OutputType.EXCEPTION,
                )
            elif isinstance(result, list):
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

    def get(self, key, timeout=None, _intra_cluster=False):
        self.register_activity()
        try:
            ret_obj = obj_store.get(
                key, timeout=timeout, check_other_envs=not _intra_cluster
            )
            logger.info(
                f"Servlet {self.env_name} got object of type "
                f"{type(ret_obj)} back from object store for key {key}"
            )
            if _intra_cluster:
                return ret_obj
            # Case 1: ...
            if isinstance(ret_obj, tuple):
                (res, obj_ref, run_name) = ret_obj
                ret_obj = res
            # Case 2: ...
            if isinstance(ret_obj, bytes):
                ret_serialized = codecs.encode(ret_obj, "base64").decode()
            # Case 3: ...
            else:
                ret_serialized = pickle_b64(ret_obj)
            return Response(
                data=ret_serialized,
                output_type=OutputType.RESULT,
            )
        except ray.exceptions.GetTimeoutError:
            return None
        except (
            ray.exceptions.TaskCancelledError,
            ray.exceptions.RayTaskError,
        ) as e:
            logger.info(f"Attempted to get task {key} that was cancelled.")
            return Response(
                error=pickle_b64(e),
                traceback=pickle_b64(traceback.format_exc()),
                output_type=OutputType.EXCEPTION,
            )
        except Exception as e:
            return Response(
                error=pickle_b64(e),
                traceback=pickle_b64(traceback.format_exc()),
                output_type=OutputType.EXCEPTION,
            )

    def get_logfiles(self, key):
        return obj_store.get_logfiles(key)

    def put_object(self, key, value, _intra_cluster=False):
        self.register_activity()
        # We may not want to deserialize the object here in case the object requires dependencies
        # (to be used inside an env) which aren't present in the BaseEnv.
        if _intra_cluster:
            obj = value
        else:
            obj = b64_unpickle(value)
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

    def rename_object(self, message: Message):
        self.register_activity()
        # We may not want to deserialize the object here in case the object requires dependencies
        # (to be used inside an env) which aren't present in the BaseEnv.
        old_key, new_key = b64_unpickle(message.data)
        logger.info(
            f"Message received from client to rename object {old_key} to {new_key}"
        )
        try:
            obj_store.rename(old_key, new_key)
            return Response(output_type=OutputType.SUCCESS)
        except Exception as e:
            logger.exception(e)
            self.register_activity()
            return Response(
                error=pickle_b64(e),
                traceback=pickle_b64(traceback.format_exc()),
                output_type=OutputType.EXCEPTION,
            )

    def delete_obj(self, message: Message):
        self.register_activity()
        keys = b64_unpickle(message.data)
        logger.info(f"Message received from client to delete keys: {keys or 'all'}")
        try:
            cleared = []
            if keys:
                for pin in keys:
                    obj_store.delete(pin)
                    cleared.append(pin)
            else:
                cleared = list(obj_store.keys())
                obj_store.clear()
            return Response(data=pickle_b64(cleared), output_type=OutputType.RESULT)
        except Exception as e:
            logger.exception(e)
            self.register_activity()
            return Response(
                error=pickle_b64(e),
                traceback=pickle_b64(traceback.format_exc()),
                output_type=OutputType.EXCEPTION,
            )

    def get_run_object(self, message: Message):
        from runhouse import folder, Run, run
        from runhouse.rns.utils.api import resolve_absolute_path

        self.register_activity()
        logger.info(
            f"Message received from client to get run object: {b64_unpickle(message.data)}"
        )
        run_name, folder_path = b64_unpickle(message.data)

        # Create folder object which points to the Run's folder on the system
        folder_path = folder_path or Run._base_cluster_folder_path(run_name)
        folder_path_on_system = resolve_absolute_path(folder_path)
        system_folder = folder(path=folder_path_on_system, dryrun=True)

        try:
            result = None
            try:
                run_config = Run._load_run_config(folder=system_folder)
                if run_config:
                    # Re-load the Run object from the Run config data (and RNS data where relevant)
                    result = run(name=run_name, path=folder_path_on_system)
            except FileNotFoundError:
                logger.info(
                    f"No config for Run {run_name} found in path: {folder_path_on_system}"
                )

            return Response(
                data=pickle_b64(result),
                output_type=OutputType.RESULT,
            )

        except Exception as e:
            return Response(
                error=pickle_b64(e),
                traceback=pickle_b64(traceback.format_exc()),
                output_type=OutputType.EXCEPTION,
            )

    def cancel_run(self, message: Message):
        # Having this be a POST instead of a DELETE on the "run" endpoint is strange, but we're not actually
        # deleting the run, just cancelling it. Maybe we should merge this into get_object to allow streaming logs.
        self.register_activity()
        run_key, force = b64_unpickle(message.data)
        logger.info(f"Message received from client to cancel runs: {run_key}")
        try:
            if run_key == "all":
                [obj_store.cancel(key, force=force) for key in obj_store.keys()]
            else:
                obj_store.cancel(run_key, force=force)

            return Response(output_type=OutputType.SUCCESS)
        except Exception as e:
            logger.exception(e)
            self.register_activity()
            return Response(
                error=pickle_b64(e),
                traceback=pickle_b64(traceback.format_exc()),
                output_type=OutputType.EXCEPTION,
            )

    def get_keys(self):
        self.register_activity()
        keys: list = list(obj_store.keys())
        return Response(data=pickle_b64(keys), output_type=OutputType.RESULT)

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

    def call_fn(fn_name, args):
        self.register_activity()
        from runhouse import function

        fn = function(name=fn_name, dryrun=True)
        result = fn(*(args.args or []), **(args.kwargs or {}))

        (fn_res, obj_ref, run_key) = result
        if isinstance(fn_res, bytes):
            fn_res = pickle.loads(fn_res)

        return fn_res
