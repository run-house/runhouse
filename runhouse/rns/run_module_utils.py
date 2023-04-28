import importlib
import logging
import sys
import traceback
from datetime import datetime
from functools import wraps
from pathlib import Path

import ray
import ray.cloudpickle as pickle

from runhouse import rh_config


logger = logging.getLogger(__name__)


def call_fn_by_type(
    fn_type,
    fn_name,
    relative_path,
    module_name,
    resources,
    conda_env=None,
    args=None,
    kwargs=None,
):
    run_key = f"{fn_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # TODO other possible fn_types: 'batch', 'streaming'
    if fn_type == "get":
        obj_ref = args[0]
        res = rh_config.obj_store.get(obj_ref)
    else:
        args = rh_config.obj_store.get_obj_refs_list(args)
        kwargs = rh_config.obj_store.get_obj_refs_dict(kwargs)

        ray.init(ignore_reinit_error=True)
        has_gpus = ray.cluster_resources().get("GPU", 0) > 0

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
            num_gpus=resources.get("num_gpus") or 0.0001 if has_gpus else None,
            max_calls=len(args) if fn_type in ["map", "starmap"] else 1,
            runtime_env=runtime_env,
        )(logging_wrapped_fn)
        obj_ref = ray_fn.remote(fn_pointers, fn_type, *args, **kwargs)

        if fn_type == "remote":
            rh_config.obj_store.put_obj_ref(key=run_key, obj_ref=obj_ref)
            res = pickle.dumps([run_key, None, None])
        else:
            res = ray.get(obj_ref)
    return res


def get_fn_from_pointers(fn_pointers, fn_type, *args, **kwargs):
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

    try:
        if fn_type == "map":
            result = [fn(arg, **kwargs) for arg in args]
        elif fn_type == "starmap":
            result = [fn(*arg, **kwargs) for arg in args]
        elif fn_type in ("queue", "remote", "call"):
            result = fn(*args, **kwargs)
        elif fn_type == "repeat":
            [num_repeats, args] = args
            result = [fn(*args, **kwargs) for _ in range(num_repeats)]
        else:
            raise ValueError(f"fn_type {fn_type} not recognized")

        if fn_type == "remote":
            return result
        else:
            # [res, None, None] is a silly hack for packaging result alongside exception and traceback
            return pickle.dumps([result, None, None])
    except Exception as e:
        logger.exception(e)
        return pickle.dumps([None, e, traceback.format_exc()])


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
