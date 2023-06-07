import importlib
import logging
import os
import subprocess
import sys
from functools import wraps
from pathlib import Path
from typing import Any

import ray

from ray import cloudpickle as pickle

from runhouse import rh_config
from runhouse.rh_config import obj_store
from runhouse.rns.api_utils.utils import resolve_absolute_path

logger = logging.getLogger(__name__)


def call_fn_by_type(
    fn,
    fn_type,
    fn_name,
    relative_path,
    module_name,
    resources,
    conda_env=None,
    run_name=None,
    args=None,
    kwargs=None,
    serialize_res=True,
):
    from runhouse import Run

    run_key = run_name if run_name else Run._create_new_run_name(fn_name)
    logger.info(f"Run key: {run_key}")

    # TODO other possible fn_types: 'batch', 'streaming'
    if fn_type == "get_or_run":
        # If Run already exists with this run key return the Run object
        run_obj = _existing_run_from_file(run_key)
        if run_obj is not None:
            logger.info(f"Found existing Run {run_obj.name}")
            return pickle.dumps(run_obj)
        else:
            # No existing run found, continue on with async execution by setting the fn_type to "remote_run",
            # which will trigger the execution async and return the Run object
            logger.info(f"No existing Run found in file system for {run_key}")
            # Note: we distinguish this from "remote" bc we only want to return a Run object in this scenario
            fn_type = "remote_run"

    if fn_type == "get":
        obj_ref = args[0]
        res = pickle.dumps(obj_store.get(obj_ref))
    else:
        ray.init(ignore_reinit_error=True)
        num_gpus = ray.cluster_resources().get("GPU", 0)
        num_cuda_devices = resources.get("num_gpus") or num_gpus

        # We need to add the module_path to the PYTHONPATH because ray runs remotes in a new process
        # We need to set max_calls to make sure ray doesn't cache the remote function and ignore changes to the module
        # See: https://docs.ray.io/en/releases-2.2.0/ray-core/package-ref.html#ray-remote
        module_path = (
            str((Path.home() / relative_path).resolve()) if relative_path else None
        )
        runtime_env = {"env_vars": {"PYTHONPATH": module_path or ""}}
        if conda_env:
            runtime_env["conda"] = conda_env

        fn_pointers = (module_path, module_name, fn_name)

        logging_wrapped_fn = enable_logging_fn_wrapper(get_fn_from_pointers, run_key)

        ray_fn = ray.remote(
            num_cpus=resources.get("num_cpus") or 0.0001,
            num_gpus=resources.get("num_gpus") or 0.0001 if num_gpus > 0 else None,
            max_calls=len(args) if fn_type in ["map", "starmap"] else 1,
            runtime_env=runtime_env,
        )(logging_wrapped_fn)

        if fn_type == "map":
            obj_ref = [
                ray_fn.remote(
                    fn_pointers, serialize_res, num_cuda_devices, arg, **kwargs
                )
                for arg in args
            ]
        elif fn_type == "starmap":
            obj_ref = [
                ray_fn.remote(
                    fn_pointers, serialize_res, num_cuda_devices, *arg, **kwargs
                )
                for arg in args
            ]
        elif fn_type in ("queue", "remote", "remote_run", "call", "get_or_call"):
            obj_ref = ray_fn.remote(
                fn_pointers, serialize_res, num_cuda_devices, *args, **kwargs
            )
        elif fn_type == "repeat":
            [num_repeats, args] = args
            obj_ref = [
                ray_fn.remote(
                    fn_pointers, serialize_res, num_cuda_devices, *args, **kwargs
                )
                for _ in range(num_repeats)
            ]
        else:
            raise ValueError(f"fn_type {fn_type} not recognized")

        if fn_type in ("remote", "remote_run"):
            run_obj = _get_or_run_async(run_key, obj_ref, fn, args, kwargs)
            if fn_type == "remote":
                res = (run_obj, obj_ref)
            else:
                # If remote_run, only return the run object since this is being triggered via `get_or_run()`
                res = pickle.dumps(run_obj)
        elif fn_type == "get_or_call":
            # Get the result if it already exists, otherwise create a new synchronous run and return its result
            res = _get_or_call_synchronously(run_key, obj_ref, fn, args, kwargs)
        elif fn_type in ("call", "nested"):
            if run_name:
                # Create a new synchronous run
                res = _run_fn_synchronously(run_key, obj_ref, fn, args, kwargs)
            else:
                res = ray.get(obj_ref)
        else:
            res = ray.get(obj_ref)

    return res


def deserialize_args_and_kwargs(args, kwargs):
    if args:
        args = pickle.loads(args)
    if kwargs:
        kwargs = pickle.loads(kwargs)
    return args, kwargs


def get_fn_by_name(module_name, fn_name, relative_path=None):
    if relative_path:
        module_path = str((Path.home() / relative_path).resolve())
        sys.path.append(module_path)
        logger.info(f"Appending {module_path} to sys.path")

    if module_name in rh_config.obj_store.imported_modules:
        importlib.invalidate_caches()
        rh_config.obj_store.imported_modules[module_name] = importlib.reload(
            rh_config.obj_store.imported_modules[module_name]
        )
        logger.info(f"Reloaded module {module_name}")
    else:
        logger.info(f"Importing module {module_name}")
        rh_config.obj_store.imported_modules[module_name] = importlib.import_module(
            module_name
        )
    fn = getattr(rh_config.obj_store.imported_modules[module_name], fn_name)
    return fn


def get_fn_from_pointers(fn_pointers, serialize_res, num_gpus, *args, **kwargs):
    (module_path, module_name, fn_name) = fn_pointers
    if module_name == "notebook":
        fn = fn_name  # already unpickled
    else:
        if module_path:
            sys.path.append(module_path)
            logger.info(f"Appending {module_path} to sys.path")

        if module_name in obj_store.imported_modules:
            importlib.invalidate_caches()
            obj_store.imported_modules[module_name] = importlib.reload(
                obj_store.imported_modules[module_name]
            )
            logger.info(f"Reloaded module {module_name}")
        else:
            logger.info(f"Importing module {module_name}")
            obj_store.imported_modules[module_name] = importlib.import_module(
                module_name
            )
        fn = getattr(obj_store.imported_modules[module_name], fn_name)

    cuda_visible_devices = list(range(int(num_gpus)))
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, cuda_visible_devices))

    result = fn(*args, **kwargs)
    if serialize_res:
        return pickle.dumps(result)
    return result


def _stdout_files_for_fn():
    worker_id = ray._private.worker.global_worker.worker_id.hex()

    ray_logs_path = RAY_LOGFILE_PATH
    stdout_files = list(ray_logs_path.glob(f"worker-{worker_id}-*"))

    return stdout_files


def _execute_remote_async_fn(async_run_obj, obj_ref):
    run_name = async_run_obj.name
    run_folder_path = async_run_obj.folder.path
    logger.info(
        f"Executing remote function for {run_name}, saving to path: {run_folder_path}"
    )

    with async_run_obj:
        result, run_status = _get_result_from_ray(obj_ref)

    # Save the result to the object store
    obj_store.put(key=run_name, value=result)
    logger.info(f"Saved result of type {type(result)} for {run_name} to object store")

    # Result will already be serialized, so no need to serialize again before saving down
    completed_run = _register_completed_run(path=run_folder_path, run_status=run_status)

    logger.info(
        f"Registered completed run: {completed_run.name} with status: {run_status}"
    )
    return completed_run


def _run_fn_async(async_run_obj, obj_ref):
    import threading

    t = threading.Thread(target=_execute_remote_async_fn, args=(async_run_obj, obj_ref))
    t.start()


def _run_fn_synchronously(run_name, obj_ref, fn, args, kwargs):
    """Run a function on the cluster, wait for its result, then return the Run object along with the result of the
    function's execution."""
    new_run = _create_new_run(run_name, fn, args, kwargs)

    logger.info(
        f"Starting execution of function for {run_name}, waiting for completion"
    )

    with new_run:
        # Open context manager to track stderr and stdout + other artifacts created by the function for this run
        # With async runs we call ray.remote() which creates the symlink to the worker files for us,
        # but since we are running this synchronously we need to create those files ourselves (via the context manager)
        result, run_status = _get_result_from_ray(obj_ref)

    # Save the result to the object store
    obj_store.put(key=run_name, value=result)
    logger.info(f"Saved result of type {type(result)} for {run_name} to object store")

    completed_run = _register_completed_run(
        path=new_run.folder.path, run_status=run_status
    )

    logger.info(f"Registered run completion in path: {completed_run.folder.path}")

    return result


def _get_or_call_synchronously(run_name, obj_ref, fn, args, kwargs) -> Any:
    if run_name == "latest":
        # TODO [JL]
        raise NotImplementedError("Latest not currently supported")

    existing_run = _existing_run_from_file(run_name)
    if existing_run is not None:
        logger.info(
            f"Loaded existing Run {existing_run.name} from the cluster's file system"
        )
        result = obj_store.get(run_name)
        if result is not None:
            # Return result saved in object store - result is not serialized so pickle
            # it before sending back over the wire
            return pickle.dumps(result)

        path_to_results = existing_run._fn_result_path()

        try:
            # Try loading from the Run's folder
            result = existing_run._load_blob_from_path(path=path_to_results).fetch()
            if result is not None:
                return result
        except:
            pass

    # No Run exists for this name, create a new one and run synchronously
    logger.info(
        f"No Run found on cluster for name {run_name}, creating a new one and running synchronously"
    )

    return _run_fn_synchronously(run_name, obj_ref, fn, args, kwargs)


def _get_or_run_async(run_key, obj_ref, fn, args, kwargs):
    existing_run = _existing_run_from_file(run_key)
    if existing_run is not None:
        logger.info(
            f"Loaded existing Run {existing_run.name} from the cluster's file system"
        )
        return existing_run

    # No Run exists for this name, create a new one and run asynchronously
    logger.info(f"No Run found for name {run_key}, creating a new one async")

    new_async_run = _create_new_run(run_key, fn, args, kwargs)

    _run_fn_async(new_async_run, obj_ref)

    # Return a Run object for the async run - regardless of whether the function execution has completed
    existing_run = _existing_run_from_file(run_key)
    logger.info(f"Loaded existing async Run {existing_run.name}")

    return existing_run


def _create_new_run(run_name, fn, args, kwargs):
    from runhouse import Run, run
    from runhouse.rns.obj_store import THIS_CLUSTER

    # Path to config file inside the dedicated folder for this particular run
    run_config_file: str = (
        f"{Run._base_cluster_folder_path(run_name)}/{Run.RUN_CONFIG_FILE}"
    )

    config_path = resolve_absolute_path(run_config_file)

    inputs = {"args": args, "kwargs": kwargs}

    if not THIS_CLUSTER:
        raise ValueError("Failed to get current cluster from config")

    current_cluster: str = rh_config.rns_client.resolve_rns_path(THIS_CLUSTER)

    # Create a new Run object, and save down its config data to the log folder on the cluster
    new_run = run(
        name=run_name,
        fn=fn,
        overwrite=True,
        fn_name=fn.__name__,
        system=current_cluster,
    )

    # Save down config for new Run
    new_run._register_new_fn_run()

    # Save down pickled inputs to the function for the Run
    new_run.write(
        data=pickle.dumps(inputs),
        path=new_run._fn_inputs_path(),
    )

    logger.info(f"Finished writing inputs for {run_name} to path: {config_path}")

    return new_run


def _get_result_from_ray(obj_ref):
    from runhouse.rns.run import RunStatus

    try:
        result = ray.get(obj_ref)
        return result, RunStatus.COMPLETED

    except Exception as e:
        return str(e), RunStatus.ERROR


def _register_completed_run(path, run_status):
    from runhouse import Run

    logger.info(f"Registering completed run in path: {path}")
    run_obj = Run.from_path(path)
    logger.info(f"Run obj in register completed run: {run_obj.config_for_rns}")

    # Update the config data for the completed run
    run_obj._register_fn_run_completion(run_status)

    return run_obj


def _existing_run_from_file(run_key):
    from runhouse import Run

    if run_key == "latest":
        # TODO [JL]
        raise NotImplementedError("Latest not currently supported")
    else:
        folder_path = Run._base_cluster_folder_path(run_key)

    folder_path_on_system = resolve_absolute_path(folder_path)

    existing_run = Run.from_path(path=folder_path_on_system)

    return existing_run


def fn_from_module_path(relative_path, fn_name, module_name):
    module_path = (
        str((Path.home() / relative_path).resolve()) if relative_path else None
    )
    logger.info(f"Module path on unary server: {module_path}")

    if module_name == "notebook":
        fn = fn_name  # Already unpickled above
    else:
        fn = get_fn_by_name(module_name, fn_name, module_path)

    return fn


def create_command_based_run(run_name, commands, cmd_prefix, python_cmd):
    from runhouse import Run, run
    from runhouse.rns.obj_store import THIS_CLUSTER
    from runhouse.rns.run import RunStatus

    folder_path = Run._base_cluster_folder_path(run_name)
    folder_path_on_system = resolve_absolute_path(folder_path)

    run_obj = run(
        name=run_name, cmds=commands, overwrite=True, path=folder_path_on_system
    )

    run_obj._register_new_fn_run()

    final_stdout = []
    final_stderr = []

    for command in commands:
        command = f"{cmd_prefix} {command}" if cmd_prefix else command

        shell = True
        if not python_cmd:
            # CLI command
            command = command.split()
            shell = False

        result = subprocess.run(command, shell=shell, capture_output=True, text=True)

        stdout = result.stdout
        stderr = result.stderr

        if stdout:
            final_stdout.append(stdout)

        if stderr:
            final_stderr.append(stderr)

    final_stdout = "\n".join(final_stdout)
    final_stderr = "\n".join(final_stderr)

    run_obj.end_time = run_obj._current_timestamp()
    run_obj.status = RunStatus.COMPLETED

    # Write the stdout and stderr of the Run to its dedicated log folder on the cluster
    run_obj.write(data=final_stdout.encode(), path=run_obj._stdout_path)
    run_obj.write(data=final_stderr.encode(), path=run_obj._stderr_path)

    logger.info(
        f"Finished saving stdout and stderr for the run to path: {run_obj.folder.path}"
    )

    # TODO [JL] better way of setting the system for this run to the current cluster (and not local)
    config_data = run_obj.config_for_rns

    current_cluster = rh_config.rns_client.resolve_rns_path(THIS_CLUSTER)
    logger.info(f"Current cluster: {current_cluster}")

    config_data["system"] = current_cluster

    run_obj._write_config(config=config_data)

    return final_stdout


RAY_LOGFILE_PATH = Path("/tmp/ray/session_latest/logs")


def enable_logging_fn_wrapper(fn, run_name):
    @wraps(fn)
    def wrapped_fn(*inner_args, **inner_kwargs):
        """Sets the logfiles for a function to be the logfiles for the current ray worker.
        This is used to stream logs from the worker to the client."""
        logdir = Path.home() / ".rh/logs" / run_name
        logdir.mkdir(exist_ok=True, parents=True)
        # Create simlinks to the ray log files in the run directory
        stdout_files: list = _stdout_files_for_fn()
        for f in stdout_files:
            symlink = logdir / f.name
            if not symlink.exists():
                symlink.symlink_to(f)
        return fn(*inner_args, **inner_kwargs)

    return wrapped_fn
