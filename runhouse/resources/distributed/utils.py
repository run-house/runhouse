import os
import sys
from pathlib import Path


def subprocess_ray_fn_call_helper(pointers, args, kwargs, conn, ray_opts={}):
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

    import ray

    ray.init(address="auto", **ray_opts)

    from runhouse.resources.module import Module

    (module_path, module_name, class_name) = pointers
    orig_fn = Module._get_obj_from_pointers(
        module_path, module_name, class_name, reload=False
    )
    res = orig_fn(*args, **kwargs)

    ray.shutdown()

    # Send an EOFError over the pipe because for some reason .close is hanging
    conn.send((EOFError, None))
    conn.close()

    return res
