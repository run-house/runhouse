import builtins
import os
import sys
import traceback
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


def extract_error_from_ray_result(result_obj):
    error = result_obj.error
    error_msg = error.__str__()
    exception = error_msg.split("\n")[-1].strip().split(": ")
    exception_type = exception[0]
    exception_msg = exception[1] if len(exception) > 1 else ""

    exception_class = Exception

    # Try to find more the exact exception that is should be raised
    if hasattr(builtins, exception_type):
        exception_class = getattr(builtins, exception_type)
    else:
        # Try to find the exception class in common modules
        for module_name in ["exceptions", "os", "io", "socket", "ray.exceptions"]:
            try:
                module = sys.modules.get(module_name) or __import__(module_name)
                if hasattr(module, exception_type):
                    exception_class = getattr(module, exception_type)
                    break
            # ImportError, AttributeError are part of the builtin methods.
            except (ImportError, AttributeError):
                continue

    # Create the exception instance with the original message
    exception_instance = exception_class(exception_msg)

    # Optionally add the original traceback as a note (Python 3.11+)
    if hasattr(exception_instance, "__notes__"):
        exception_instance.__notes__ = traceback.format_exception(error)
    raise exception_instance


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
        res = extract_error_from_ray_result(res) if hasattr(res, "error") else res
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
