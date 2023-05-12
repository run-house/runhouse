import importlib
import json
import logging
import os
import sys
from functools import wraps
from pathlib import Path

import ray

from ray import cloudpickle as pickle

from runhouse import rh_config

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

    run_key = run_name or Run.base_folder_name(fn_name)
    logger.info(f"Run key: {run_key} for run name: {run_name}")

    # TODO other possible fn_types: 'batch', 'streaming'
    if fn_type == "get":
        obj_ref = args[0]
        res = pickle.dumps(rh_config.obj_store.get(obj_ref))
    elif fn_type == "get_or_run":
        # look up result in object store, if the run is not started execute and synchronously return the result
        run_status: dict = _get_or_run(run_key, fn, args, kwargs)
        res = pickle.dumps(run_status)
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
        elif fn_type in ("queue", "remote", "call", "nested"):
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
            rh_config.obj_store.put_obj_ref(key=run_key, obj_ref=obj_ref)
            res = pickle.dumps(run_key)
            # create a new thread and start running the function in the background - when finished it will
            # be stored in the ray object store for faster retrieval
            _run_fn_async(run_key, fn, obj_ref, args, kwargs)
        elif fn_type in ("call", "nested"):
            if run_name is not None:
                # Create a synchronous run
                res = _run_fn_synchronously(run_key, fn, args, kwargs)
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


def _execute_remote_async_fn(run_key, fn, obj_ref, args, kwargs):
    logger.info(f"Executing remote function for {run_key}")

    current_cluster = current_cluster_obj()
    logger.info(f"Current cluster: {current_cluster.name}")

    new_run, config_path_on_cluster = _create_new_run(
        run_key, fn, current_cluster, args, kwargs
    )
    logger.info(f"Created new run {new_run.name} in path: {config_path_on_cluster}")

    logger.info("Using ray to execute function async...")

    result = ray.get(obj_ref)

    logger.info(f"Finished running function async for {run_key}")

    _register_completed_run(
        run_key=run_key,
        config_path=config_path_on_cluster,
        result=result,
    )


def _run_fn_async(run_key, fn, obj_ref, args, kwargs):
    import threading

    t = threading.Thread(
        target=_execute_remote_async_fn, args=(run_key, fn, obj_ref, args, kwargs)
    )
    t.start()
    logger.info(f"Started new thread for running {run_key}")


def _run_fn_synchronously(run_key, fn, args, kwargs):
    current_cluster = current_cluster_obj()
    new_run, config_path_on_cluster = _create_new_run(
        run_key, fn, current_cluster, args, kwargs
    )
    logger.info(f"Starting execution of function for {run_key}, waiting for completion")

    with new_run:
        # Open context manager to track stderr and stdout + other artifacts created by the function for this run
        # With async runs we call ray.remote() which creates the symlink to the worker files for us,
        # but since we are running this synchronously we need to create those files ourselves (via the context manager)
        res = fn(*args, **kwargs)

    logger.info(f"Finished running function synchronously for {run_key}")

    _register_completed_run(
        run_key=run_key,
        config_path=config_path_on_cluster,
        result=res,
    )
    return res


def current_cluster_obj():
    from runhouse import cluster
    from runhouse.rns.obj_store import THIS_CLUSTER

    return cluster(f"@/{THIS_CLUSTER}")


def _get_or_run(run_key, fn, args, kwargs) -> dict:
    from runhouse import Run

    system = current_cluster_obj()
    logger.info(f"Current System: {system.name}")

    config_path_on_system = Run.default_config_path(run_key, system)
    config_path = os.path.abspath(os.path.expanduser(config_path_on_system))
    logger.info(f"Config path on system for {run_key}: {config_path}")

    try:
        # Load config data for this Run saved on the cluster
        with open(config_path, "r") as f:
            run_config = json.load(f)
    except FileNotFoundError:
        logger.error(f"Failed to find existing config for run in path {config_path}")
        return {"status": Run.NOT_STARTED_STATUS, "result": None}

    # Re-load the Run object
    existing_run = Run(**run_config, dryrun=True)
    logger.info(f"Loaded existing Run {run_key} from {system.name}")

    config = existing_run.config_for_rns
    run_status = config.get("status")

    if run_status == Run.COMPLETED_STATUS:
        logger.info(f"Run {run_key} is already completed, returning result")
        return {"status": run_status, "result": existing_run.result()}

    elif run_status == Run.NOT_STARTED_STATUS:
        logger.info(f"Run {run_key} has not started, starting now")
        result = fn(*args, **kwargs)
        logger.info(f"Finished running {run_key}")
        return {"status": run_status, "result": result}

    elif run_status == Run.ERROR_STATUS:
        logger.info(f"Run {run_key} has failed, returning stderr")
        return {"status": run_status, "result": existing_run.stderr()}

    elif existing_run.status == Run.RUNNING_STATUS:
        logger.info(f"Run {run_key} is still running")
        return {"status": run_status, "result": run_key}

    else:
        logger.warning(f"Run {run_key} in unexpected state: {run_status}")
        return {"status": run_status, "result": None}


def _create_new_run(run_key, fn, current_cluster, args, kwargs):
    from runhouse import Run, run

    logger.info(f"Creating Run with name: {run_key}")

    # Path to config file inside the dedicated folder for this particular run
    run_config_file: str = Run.default_config_path(name=run_key, system=current_cluster)
    config_path = os.path.abspath(os.path.expanduser(run_config_file))

    inputs = {"args": args, "kwargs": kwargs}

    # Create a new Run object, and save down its config data to the log folder on the cluster
    new_run = run(
        name=run_key, fn=fn, system=current_cluster, load=False, overwrite=True
    )

    logger.info(f"Writing inputs to path: {new_run.fn_inputs_path()}")
    # Write the inputs to the function for this run to folder
    new_run.write(
        data=pickle.dumps(inputs),
        path=new_run.fn_inputs_path(),
    )

    logger.info(f"Finished writing inputs for {run_key} to path: {config_path}")

    new_run.register_new_fn_run()

    return new_run, config_path


def _register_completed_run(run_key, config_path, result):
    from runhouse import Run

    # Load the Run object we previously created
    logger.info(f"Registering completed run for {run_key} in path: {config_path}")

    # Load the config for this Run from the local file system on the cluster
    existing_run = Run.from_file(
        name=run_key, path=f"~/{rh_config.obj_store.LOGS_DIR}/{run_key}"
    )

    logger.info(
        f"Loaded existing Run {existing_run.name} from the cluster's file system"
    )

    if not existing_run.exists_in_system(path=existing_run._stdout_path):
        existing_run._folder.put({f"{run_key}.out": "".encode()})
        logger.info(f"Created empty stdout file in path: {existing_run._stdout_path}")

    if not existing_run.exists_in_system(path=existing_run._stderr_path):
        existing_run._folder.put({f"{run_key}.err": "".encode()})
        logger.info(f"Created empty stderr file in path: {existing_run._stderr_path}")

    # Update the config data for the completed run
    logger.info(f"Registering completed run for {run_key}")
    existing_run.register_fn_run_completion()

    # Write the result of the Run to its dedicated log folder on the cluster
    existing_run.write(
        data=pickle.dumps(result),
        path=existing_run.fn_result_path(),
    )

    logger.info(
        f"Saved pickled result to folder in path: {existing_run.fn_result_path()}"
    )


RAY_LOGFILE_PATH = Path("/tmp/ray/session_latest/logs")


def enable_logging_fn_wrapper(fn, run_key):
    @wraps(fn)
    def wrapped_fn(*inner_args, **inner_kwargs):
        """Sets the logfiles for a function to be the logfiles for the current ray worker.
        This is used to stream logs from the worker to the client."""
        worker_id = ray._private.worker.global_worker.worker_id.hex()
        logdir = Path.home() / ".rh/logs" / run_key
        logdir.mkdir(exist_ok=True, parents=True)
        ray_logs_path = RAY_LOGFILE_PATH
        stdout_files = list(ray_logs_path.glob(f"worker-{worker_id}-*"))
        # Create simlinks to the ray log files in the run directory
        logger.info(f"Writing logs on cluster to {logdir}")
        for f in stdout_files:
            symlink = logdir / f.name
            if not symlink.exists():
                symlink.symlink_to(f)
        return fn(*inner_args, **inner_kwargs)

    return wrapped_fn
