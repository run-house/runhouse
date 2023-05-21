import importlib
import logging
import os
import sys
from functools import wraps
from pathlib import Path
from typing import Any

import ray

from ray import cloudpickle as pickle

from runhouse import rh_config
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
):
    from runhouse import Run

    run_key = run_name if run_name else Run.base_folder_name(fn_name)
    logger.info(f"Run key: {run_key}")

    # TODO other possible fn_types: 'batch', 'streaming'
    if fn_type == "get_or_run":
        try:
            # Check if Run already exists for name, immediately return the Run object if so.
            # Otherwise execute a new run async
            existing_run = _existing_run_from_file(run_key)
            logger.info(f"Found existing Run {existing_run.name}")
            return pickle.dumps(existing_run)
        except:
            # No existing run found, continue on with async execution by setting the fn_type to "remote"
            logger.info(f"No existing run found for {run_key}")
            fn_type = "remote"

    if fn_type == "get":
        obj_ref = args[0]
        res = pickle.dumps(rh_config.obj_store.get(obj_ref))
    else:
        args = rh_config.obj_store.get_obj_refs_list(args)
        kwargs = rh_config.obj_store.get_obj_refs_dict(kwargs)

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
                ray_fn.remote(fn_pointers, fn_type, num_cuda_devices, arg, **kwargs)
                for arg in args
            ]
        elif fn_type == "starmap":
            obj_ref = [
                ray_fn.remote(fn_pointers, fn_type, num_cuda_devices, *arg, **kwargs)
                for arg in args
            ]
        elif fn_type in ("queue", "remote", "call", "get_or_call", "nested"):
            obj_ref = ray_fn.remote(
                fn_pointers, fn_type, num_cuda_devices, *args, **kwargs
            )
        elif fn_type == "repeat":
            [num_repeats, args] = args
            obj_ref = [
                ray_fn.remote(fn_pointers, fn_type, num_cuda_devices, *args, **kwargs)
                for _ in range(num_repeats)
            ]
        else:
            raise ValueError(f"fn_type {fn_type} not recognized")

        if fn_type == "remote":
            # Create a new thread and start running the function async in the background - when finished the result
            # will be saved to the Run's dedicated folder on the cluster
            rh_config.obj_store.put_obj_ref(key=run_key, obj_ref=obj_ref)

            async_run = _get_or_run_async(run_key, obj_ref, fn, args, kwargs)
            # Return a run object
            res = pickle.dumps(async_run)
        elif fn_type == "get_or_call":
            res = _get_or_call_synchronously(run_key, obj_ref, fn, args, kwargs)
            res = pickle.dumps(res)
        elif fn_type in ("call", "nested"):
            if run_name:
                # Create a synchronous run
                res = _run_fn_synchronously(run_key, obj_ref, fn, args, kwargs)
                res = pickle.dumps(res)
            else:
                res = ray.get(obj_ref)
        else:
            res = pickle.dumps(ray.get(obj_ref))

    return res


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


def get_fn_from_pointers(fn_pointers, fn_type, num_gpus, *args, **kwargs):
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
    if fn_type == "call":
        return pickle.dumps(result)
    return result


def _stdout_files_for_fn():
    worker_id = ray._private.worker.global_worker.worker_id.hex()

    ray_logs_path = RAY_LOGFILE_PATH
    stdout_files = list(ray_logs_path.glob(f"worker-{worker_id}-*"))

    return stdout_files


def _execute_remote_async_fn(async_run_obj, obj_ref):
    logger.info(f"Executing remote function for {async_run_obj.name}")

    with async_run_obj:
        result = ray.get(obj_ref)

    completed_run = _register_completed_run(
        run_obj=async_run_obj,
        result=result,
    )

    logger.info(f"Registered completed run: {completed_run.name}")
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
        res = ray.get(obj_ref)

    # Need to decode the result for a synchronous run since we are returning it directly back to the user
    result = pickle.loads(res)

    completed_run = _register_completed_run(
        run_obj=new_run,
        result=result,
    )

    logger.info(f"Registered run completion in path: {completed_run.path}")

    return result


def _get_or_call_synchronously(run_name, obj_ref, fn, args, kwargs) -> Any:
    if run_name == "latest":
        # TODO [JL]
        raise NotImplementedError("Latest not currently supported")

    try:
        existing_run = _existing_run_from_file(run_name)
        logger.info(
            f"Loaded existing Run {existing_run.name} from the cluster's file system"
        )
        return existing_run.result()

    except:
        # No Run exists for this name, create a new one and run synchronously
        logger.info(
            f"No Run found for name {run_name}, creating a new one and running synchronously"
        )
        return _run_fn_synchronously(run_name, obj_ref, fn, args, kwargs)


def _get_or_run_async(run_key, obj_ref, fn, args, kwargs):
    try:
        existing_run = _existing_run_from_file(run_key)
        logger.info(
            f"Loaded existing Run {existing_run.name} from the cluster's file system"
        )
        return existing_run

    except:
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
        f"{Run.base_cluster_folder_path(run_name)}/{Run.RUN_CONFIG_FILE}"
    )

    config_path = resolve_absolute_path(run_config_file)

    inputs = {"args": args, "kwargs": kwargs}

    # Create a new Run object, and save down its config data to the log folder on the cluster
    current_cluster: str = rh_config.rns_client.resolve_rns_path(THIS_CLUSTER)

    new_run = run(
        name=run_name,
        fn=fn,
        overwrite=True,
        fn_name=fn.__name__,
        system=current_cluster,
    )

    # Write the inputs to the function for this run to folder
    new_run.write(
        data=pickle.dumps(inputs),
        path=new_run.fn_inputs_path(),
    )

    logger.info(f"Finished writing inputs for {run_name} to path: {config_path}")

    new_run.register_new_fn_run()

    return new_run


def _register_completed_run(run_obj, result):
    # Load the Run object we previously created
    logger.info(f"Registering completed run for {run_obj.name} in path: {run_obj.path}")

    # Update the config data for the completed run
    run_obj.register_fn_run_completion()

    # Write the result of the Run to its dedicated log folder on the cluster
    run_obj.write(
        data=pickle.dumps(result),
        path=run_obj.fn_result_path(),
    )

    logger.info(f"Saved pickled result to folder in path: {run_obj.fn_result_path()}")

    return run_obj


def _existing_run_from_file(run_key):
    from runhouse import Run

    if run_key == "latest":
        # TODO [JL]
        raise NotImplementedError("Latest not currently supported")
    else:
        folder_path = Run.base_cluster_folder_path(run_key)

    if folder_path is None:
        return None

    folder_path_on_system = resolve_absolute_path(folder_path)

    existing_run = Run.from_file(run_name=run_key, folder_path=folder_path_on_system)

    return existing_run


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
