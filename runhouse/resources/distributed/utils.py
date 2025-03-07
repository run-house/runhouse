import os
import sys
from pathlib import Path


def _setup_subprocess_env(pointers, conn, ray_opts={}):
    """Common setup logic for ray subprocess functions."""

    def write_stdout(msg):
        conn.send((msg, "stdout"))

    def write_stderr(msg):
        conn.send((msg, "stderr"))

    sys.stdout.write = write_stdout
    sys.stderr.write = write_stderr

    abs_module_path = str(Path(pointers[0]).expanduser().resolve())

    ray_opts["runtime_env"] = ray_opts.get("runtime_env", {})
    ray_opts["runtime_env"]["env_vars"] = ray_opts["runtime_env"].get("env_vars", {})
    ray_opts["runtime_env"]["env_vars"]["RH_LOG_LEVEL"] = os.environ.get(
        "RH_LOG_LEVEL", "INFO"
    )

    if "PYTHONPATH" in ray_opts["runtime_env"]["env_vars"]:
        pp = ray_opts["runtime_env"]["env_vars"]["PYTHONPATH"]
        ray_opts["runtime_env"]["env_vars"]["PYTHONPATH"] = f"{abs_module_path}:{pp}"
    else:
        ray_opts["runtime_env"]["env_vars"]["PYTHONPATH"] = abs_module_path

    return ray_opts


def _cleanup_subprocess(conn):
    conn.send((EOFError, None))
    conn.close()


def subprocess_ray_fn_call_helper(pointers, args, kwargs, conn, ray_opts={}):
    ray_opts = _setup_subprocess_env(pointers, conn, ray_opts)

    import ray

    ray.init(address="auto", **ray_opts)

    from runhouse.resources.module import Module

    (module_path, module_name, class_name) = pointers
    orig_fn = Module._get_obj_from_pointers(
        module_path, module_name, class_name, reload=False
    )
    try:
        res = orig_fn(*args, **kwargs)
        return res
    finally:
        ray.shutdown()
        _cleanup_subprocess(
            conn
        )  # Send an EOFError over the pipe because for some reason .close is hanging


def subprocess_raydp_fn_call_helper(
    pointers, args, kwargs, conn, ray_opts={}, spark_opts={}
):
    ray_opts = _setup_subprocess_env(pointers, conn, ray_opts)

    import ray

    try:
        import raydp
    except ImportError as e:
        raise ImportError(
            "RayDP not installed. Please add RayDP to your Runhouse image or install it to use the Spark distributed module"
        ) from e

    ray.init(address="auto", **ray_opts)

    from runhouse.resources.module import Module

    (module_path, module_name, class_name) = pointers
    orig_fn = Module._get_obj_from_pointers(
        module_path, module_name, class_name, reload=False
    )

    spark = raydp.init_spark(
        app_name=spark_opts.get("app_name", "runhouse_spark"),
        num_executors=spark_opts.get("num_executors", 2),
        executor_cores=spark_opts.get("executor_cores", 2),
        executor_memory=spark_opts.get("executor_memory", "2GB"),
    )

    import inspect

    if "spark" not in inspect.signature(orig_fn).parameters:
        raise TypeError(
            "The function you are trying to distribute with Spark must have an argument named 'spark.' Runhouse will provide the Spark client to your function."
        )

    try:
        try:
            res = orig_fn(spark=spark, *args, **kwargs)
        except Exception as e:  # Serialization issue with exceptions
            print("Error calling Spark function", e)
            import traceback

            raise RuntimeError(
                f"{str(e)}\nTraceback:\n{traceback.format_exc()}"
            ) from None
        return res

    finally:
        ray.shutdown()
        _cleanup_subprocess(conn)
