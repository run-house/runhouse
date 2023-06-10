import importlib
import logging
import os
import subprocess
import sys
from functools import wraps
from pathlib import Path
from typing import Any, Union

import ray

from ray import cloudpickle as pickle

from runhouse import rh_config
from runhouse.rns.api_utils.utils import resolve_absolute_path

logger = logging.getLogger(__name__)


def call_fn_by_type(
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
                logger.info(
                    f"Found existing Run with name: {run_name}, returning its result"
                )
                fn_result = run_obj._load_blob_from_path(
                    path=run_obj._fn_result_path()
                ).fetch()
                logger.info(f"Function result: {type(fn_result)}")
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

    logging_wrapped_fn = enable_logging_fn_wrapper(get_fn_from_pointers, run_name)

    ray_fn = ray.remote(
        num_cpus=resources.get("num_cpus") or 0.0001,
        num_gpus=resources.get("num_gpus") or 0.0001 if num_gpus > 0 else None,
        max_calls=len(args) if fn_type in ["map", "starmap"] else 1,
        runtime_env=runtime_env,
    )(logging_wrapped_fn)

    # TODO other possible fn_types: 'batch', 'streaming'
    if fn_type == "map":
        obj_ref = [
            ray_fn.remote(fn_pointers, serialize_res, num_cuda_devices, arg, **kwargs)
            for arg in args
        ]
    elif fn_type == "starmap":
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
        fn_result = _execute_fn_run(run_obj, run_name, fn_type, fn_name)
        logger.info(f"Result from function execution: {type(fn_result)}")
        res = fn_result
    else:
        res = ray.get(obj_ref)

    # Note: Results by default are serialized
    return res, obj_ref, run_name


def deserialize_args_and_kwargs(args, kwargs):
    if args:
        args = pickle.loads(args)
    if kwargs:
        kwargs = pickle.loads(kwargs)
    return args, kwargs


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


def _execute_fn_with_ray(run_obj, fn_name):
    from runhouse import RunStatus

    result, run_status = _get_result_from_ray(run_obj.name)

    run_obj._register_run_completion(run_status)

    if run_status == RunStatus.COMPLETED:
        # Only store result if Run completed successfully
        _serialize_and_store_result(result, run_obj)

    logger.info(
        f"Registered completed async run {run_obj.name} for function {fn_name} with status: {run_status}"
    )

    return run_obj


def _execute_fn_run(run_obj, run_name, fn_type, fn_name) -> ["Run", Any]:
    """Execute a function on the cluster. If the ``fn_type`` is ``remote``, then execute asynchronously on another
    thread before returning the Run object. Otherwise execute the function synchronously and return its
    result once completed."""
    from runhouse import Run, RunStatus

    logger.info(f"Starting execution of type {fn_type} for Run with name: {run_name}")

    if fn_type == "remote":
        import threading

        logger.info("Running remote function asynchronously on a new thread")

        t = threading.Thread(target=_execute_fn_with_ray, args=(run_obj, fn_name))
        t.start()

        # Reload the Run object
        async_run = _load_existing_run_on_cluster(run_name)
        return async_run

    else:
        fn_result, run_status = _get_result_from_ray(run_name)

        path = run_obj.folder.path
        logger.info(
            f"Registering completed run with status {run_status} in path: {path}"
        )

        run_obj = Run.from_path(path)

        # Update the config data for the completed run
        run_obj._register_run_completion(run_status)

        if run_status == RunStatus.COMPLETED:
            # Only store result if Run completed successfully
            _serialize_and_store_result(fn_result, run_obj)

        logger.info(f"Completed Run for fn type {fn_type} in path: {path}")

        # Result from Ray object store will not be serialized by default
        return pickle.dumps(fn_result)


def _create_new_run(run_name, fn_name, args, kwargs):
    """Create a new Run object and save down relevant config data to its dedicated folder on the cluster."""
    from runhouse import Run
    from runhouse.rns.obj_store import THIS_CLUSTER

    # Path to config file inside the dedicated folder for this particular run
    run_config_file: str = (
        f"{Run._base_cluster_folder_path(run_name)}/{Run.RUN_CONFIG_FILE}"
    )

    config_path = resolve_absolute_path(run_config_file)

    inputs = {"args": args, "kwargs": kwargs}
    logger.info(f"Inputs for Run: {inputs}")

    if not THIS_CLUSTER:
        raise ValueError("Failed to get current cluster from config")

    current_cluster: str = rh_config.rns_client.resolve_rns_path(THIS_CLUSTER)

    # Create a new Run object
    new_run = Run(
        name=run_name, fn_name=fn_name, system=current_cluster, overwrite=True
    )

    # Save down config for new Run
    new_run._register_new_run()

    # Save down pickled inputs to the function for the Run
    new_run.write(
        data=pickle.dumps(inputs),
        path=new_run._fn_inputs_path(),
    )

    logger.info(f"Finished writing inputs for {run_name} to path: {config_path}")

    return new_run


def _get_result_from_ray(run_name):
    from runhouse.rns.run import RunStatus

    try:
        result = rh_config.obj_store.get(key=run_name)
        return result, RunStatus.COMPLETED
    except Exception as e:
        return str(e), RunStatus.ERROR


def _load_existing_run_on_cluster(run_name) -> Union["Run", None]:
    """Load a Run for a given name from the cluster's file system. If the Run is not found returns None."""
    from runhouse import Run

    if run_name == "latest":
        # TODO [JL]
        raise NotImplementedError("Latest not currently supported")
    else:
        folder_path = Run._base_cluster_folder_path(run_name)

    folder_path_on_system = resolve_absolute_path(folder_path)
    logger.info(
        f"folder path on system for loading run from file: {folder_path_on_system}"
    )

    existing_run = Run.from_path(path=folder_path_on_system)

    return existing_run


def _serialize_and_store_result(result, run_obj):
    """Save the result of the function to the Run's dedicated folder on the cluster"""
    # NOTE: relying on the ray cluster object store for storing the object ref is a finicky, so for now we
    # serialize and store the result on the file system
    results_path = run_obj._fn_result_path()
    logger.info(f"Writing function result to Run's folder in path: {results_path}")
    run_obj.write(pickle.dumps(result), path=results_path)


def create_command_based_run(run_name, commands, cmd_prefix, python_cmd) -> list:
    """Create a Run for a CLI or python command(s).

    Returns:
        A list of tuples containing: (returncode, stdout, stderr)."""

    from runhouse import Run
    from runhouse.rns.obj_store import THIS_CLUSTER
    from runhouse.rns.run import RunStatus

    folder_path = Run._base_cluster_folder_path(run_name)
    folder_path_on_system = resolve_absolute_path(folder_path)

    run_obj = Run(
        name=run_name, cmds=commands, overwrite=True, path=folder_path_on_system
    )

    run_obj._register_new_run()

    final_stdout = []
    final_stderr = []
    return_codes = []

    run_status = RunStatus.COMPLETED

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

        return_code = result.returncode
        if return_code != 0:
            run_status = RunStatus.ERROR

        commands_res = (return_code, stdout, stderr)
        return_codes.append(commands_res)

    end_time = Run._current_timestamp()

    final_stdout = "\n".join(final_stdout)
    final_stderr = "\n".join(final_stderr)

    config_data = run_obj.config_for_rns

    # Update the "system" stored in the config for this Run from "file" to the current cluster
    config_data["system"] = rh_config.rns_client.resolve_rns_path(THIS_CLUSTER)
    config_data["end_time"] = end_time
    config_data["status"] = run_status

    # Save the updated config for the Run
    run_obj._write_config(config=config_data)

    # Write the stdout and stderr of the Run to its dedicated folder
    run_obj.write(data=final_stdout.encode(), path=run_obj._stdout_path)
    run_obj.write(data=final_stderr.encode(), path=run_obj._stderr_path)

    logger.info(
        f"Finished saving stdout and stderr for the Run to path: {run_obj.folder.path}"
    )

    return return_codes


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
