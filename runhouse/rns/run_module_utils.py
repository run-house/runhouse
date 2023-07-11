import importlib
import logging
import os
import sys
import time
from functools import wraps
from pathlib import Path
from typing import Any, Union

import ray

from ray import cloudpickle as pickle

from runhouse import rh_config
from runhouse.rns.api_utils.utils import resolve_absolute_path
from runhouse.rns.utils.env import _env_vars_from_file

logger = logging.getLogger(__name__)


def call_fn_by_type(
    fn_type,
    fn_name,
    relative_path,
    module_name,
    resources,
    conda_env=None,
    env_vars=None,
    run_name=None,
    args=None,
    kwargs=None,
    serialize_res=True,
):
    from runhouse import Run, RunStatus

    # **NOTE**: No need to pass in the fn callable here! We should be able to get it from the module_name and fn_name

    run_name = run_name if run_name else Run._create_new_run_name(fn_name)

    if fn_type.startswith("get_or"):
        # If Run already exists with this run key return the Run object
        run_obj = _load_existing_run_on_cluster(run_name)
        if run_obj is not None:
            logger.info(f"Found existing Run {run_obj.name}")
            if fn_type == "get_or_run":
                logger.info(
                    f"Found existing Run with name: {run_name}, returning the Run object"
                )
                return run_obj, None, run_name
            elif fn_type == "get_or_call":
                results_path = run_obj._fn_result_path()
                run_status = run_obj.status
                logger.info(f"Found existing Run {run_name} with status: {run_status}")
                try:
                    fn_result = run_obj._load_blob_from_path(path=results_path).fetch()
                    logger.info("Loaded result for Run")
                except FileNotFoundError:
                    # If no result is saved (ex: Run failed) return the stderr or stdout depending on its status
                    logger.info(f"No result found in path: {results_path}")
                    run_output = (
                        run_obj.stderr()
                        if run_status == RunStatus.ERROR
                        else run_obj.stdout()
                    )
                    # Note: API expects serialized result, so pickle before returning
                    fn_result = pickle.dumps(run_output)
                return fn_result, None, run_name
            else:
                raise ValueError(
                    f"Invalid fn_type {fn_type} for an existing Run with name {run_name}"
                )
        else:
            logger.info(f"No existing Run found in file system for {run_name}")
            if fn_type == "get_or_run":
                # No existing run found, continue on with async execution
                fn_type = "remote"
                logger.info(
                    f"No Run found for name {run_name}, creating a new one async"
                )
            elif fn_type == "get_or_call":
                # No existing run found, continue on with synchronous execution
                fn_type = "call"
                logger.info(
                    f"No Run found for name {run_name}, creating a new one and running synchronously"
                )
            else:
                raise ValueError(
                    f"Invalid fn_type {fn_type} for a Run that does not exist with name {run_name}"
                )

    num_gpus = ray.cluster_resources().get("GPU", 0)
    num_cuda_devices = resources.get("num_gpus") or num_gpus

    # We need to add the module_path to the PYTHONPATH because ray runs remotes in a new process
    # We need to set max_calls to make sure ray doesn't cache the remote function and ignore changes to the module
    # See: https://docs.ray.io/en/releases-2.2.0/ray-core/package-ref.html#ray-remote
    module_path = (
        str((Path.home() / relative_path).resolve()) if relative_path else None
    )

    if isinstance(env_vars, str):
        env_vars = _env_vars_from_file(env_vars)
    elif env_vars is None:
        env_vars = {}

    env_vars["PYTHONPATH"] = module_path or ""
    runtime_env = {"env_vars": env_vars}
    if conda_env:
        runtime_env["conda"] = conda_env

    fn_pointers = (module_path, module_name, fn_name)

    logging_wrapped_fn = enable_logging_fn_wrapper(get_fn_from_pointers, run_name)

    ray_fn = ray.remote(
        num_cpus=resources.get("num_cpus") or 0.0001,
        num_gpus=resources.get("num_gpus") or 0.0001 if num_gpus > 0 else None,
        max_calls=len(args) if fn_type in ["starmap"] else 1,
        runtime_env=runtime_env,
    )(logging_wrapped_fn)

    # TODO other possible fn_types: 'batch', 'streaming'
    if fn_type == "starmap":
        obj_ref = [
            ray_fn.remote(fn_pointers, serialize_res, num_cuda_devices, *arg, **kwargs)
            for arg in args
        ]
    elif fn_type in ("queue", "remote", "call", "nested"):
        run_obj = _create_new_run(run_name, fn_name, args, kwargs)
        obj_ref = ray_fn.remote(
            fn_pointers, serialize_res, num_cuda_devices, *args, **kwargs
        )
    elif fn_type == "repeat":
        [num_repeats, args] = args
        obj_ref = [
            ray_fn.remote(fn_pointers, serialize_res, num_cuda_devices, *args, **kwargs)
            for _ in range(num_repeats)
        ]
    else:
        raise ValueError(f"fn_type {fn_type} not recognized")

    if fn_type in ("queue", "remote", "call", "nested"):
        fn_result = _populate_run_with_result(run_obj, fn_type, obj_ref)
        logger.info(f"Result from function execution: {type(fn_result)}")
        res = fn_result
    else:
        res = ray.get(obj_ref)

    # Note: Results by default are serialized
    return res, obj_ref, run_name


def get_fn_from_pointers(fn_pointers, serialize_res, num_gpus, *args, **kwargs):
    (module_path, module_name, fn_name) = fn_pointers

    if module_name == "notebook":
        fn = fn_name  # already unpickled
    else:
        if module_path:
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


def _populate_run_with_result(run_obj, fn_type, obj_ref) -> ["Run", Any]:
    """Execute a function on the cluster. If the ``fn_type`` is ``remote``, then execute asynchronously on another
    thread before returning the Run object. Otherwise execute the function synchronously and return its
    result once completed."""
    run_name = run_obj.name
    logger.info(f"Starting execution of type {fn_type} for Run with name: {run_name}")

    if fn_type == "remote":
        import threading

        logger.info("Running remote function asynchronously on a new thread")

        t = threading.Thread(target=_get_result_from_ray, args=(run_obj, obj_ref))
        t.start()

        # Reload the Run object
        async_run = _load_existing_run_on_cluster(run_name)
        return async_run
    else:
        fn_result, run_status = _get_result_from_ray(run_obj, obj_ref)
        logger.info(
            f"Completed Run for fn type {fn_type} in path: {run_obj.folder.path}"
        )

        # Result should be already serialized
        return fn_result


def _create_new_run(run_name, fn_name, args, kwargs):
    """Create a new Run object and save down relevant config data to its dedicated folder on the cluster."""
    from runhouse.rns.run import Run

    inputs = {"args": args, "kwargs": kwargs}
    logger.info(f"Inputs for Run: {inputs}")

    system = _get_current_system()
    path_to_run = Run._base_cluster_folder_path(run_name)

    logger.info(f"Path to new run: {path_to_run}")
    # Create a new Run object
    new_run = Run(
        name=run_name,
        fn_name=fn_name,
        system=system,
        overwrite=True,
        path=path_to_run,
    )

    logger.info(f"Created new run object from factory: {new_run.run_config}")

    # Save down config for new Run
    new_run._register_new_run()

    # Save down pickled inputs to the function for the Run
    new_run.write(data=pickle.dumps(inputs), path=new_run._fn_inputs_path())

    logger.info(
        f"Finished writing inputs for {run_name} to path: {new_run.folder.path}"
    )

    return new_run


def _get_current_system():
    from runhouse import Folder
    from runhouse.rns.utils.hardware import _current_cluster, _get_cluster_from

    system = _get_cluster_from(
        _current_cluster(key="config") or Folder.DEFAULT_FS, dryrun=True
    )
    return system


def _get_result_from_ray(run_obj, obj_ref):
    from runhouse import RunStatus

    try:
        # Result will not be serialized by default
        result = ray.get(obj_ref)
        logger.info(
            f"Successfully got result of type {type(result)} from Ray object store"
        )

        run_status = RunStatus.COMPLETED

        if not isinstance(result, bytes):
            # Serialize before returning
            result = pickle.dumps(result)

        # Only store result if Run completed successfully
        _save_run_result(result, run_obj)

    except ray.exceptions.RayTaskError as e:
        logger.info(f"Failed to get result from Ray object store: {e}")
        run_status = RunStatus.ERROR
        result = e

        # write traceback to new error file - this will be loaded as the result of the Run
        stderr_path = run_obj._stderr_path
        logger.info(f"Writing error to stderr path: {stderr_path}")
        time.sleep(1)
        run_obj.write(data=str(e).encode(), path=stderr_path)

    finally:
        run_obj._register_fn_run_completion(run_status)

        logger.info(
            f"Registered completed Run {run_obj.name} with status: {run_status}"
        )

        return result, run_status


def _load_existing_run_on_cluster(run_name) -> Union["Run", None]:
    """Load a Run for a given name from the cluster's file system. If the Run is not found returns None."""
    from runhouse import Run, RunType
    from runhouse.rns.folders import folder as folder_factory

    if run_name == "latest":
        # TODO [JL]
        raise NotImplementedError("Latest not currently supported")
    else:
        folder_path = Run._base_cluster_folder_path(run_name)

    folder_path_on_system = resolve_absolute_path(folder_path)
    logger.info(
        f"folder path on system for loading run from file: {folder_path_on_system}"
    )

    system = _get_current_system()
    system_folder = folder_factory(
        path=folder_path_on_system, system=system, dryrun=True
    )

    # Try loading Run from the file system
    run_config = Run._load_run_config(system_folder)
    if not run_config:
        return None

    try:
        # Load the RNS data for this Run - needed to reload the Run object
        rns_config = Run.from_name(run_name, dryrun=True)
        logger.info("Loaded Run metadata from RNS")
    except:
        # No RNS metadata saved for this Run
        logger.info("No RNS metadata found for this Run")
        rns_config = {
            "resource_type": Run.RESOURCE_TYPE,
            "resource_subtype": Run.RESOURCE_TYPE,
            "path": folder_path_on_system,
            "system": system,
            "run_type": RunType.FUNCTION_RUN,
        }

    config = {**run_config, **rns_config}

    return Run.from_config(config, dryrun=True)


def _save_run_result(result: bytes, run_obj):
    """Save the serialized result of the function to the Run's dedicated folder on the cluster"""
    results_path = run_obj._fn_result_path()
    logger.info(f"Writing function result to Run's folder in path: {results_path}")
    run_obj.write(result, path=results_path)


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
