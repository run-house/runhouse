import logging
import os
import traceback
from functools import wraps
from typing import Any, Dict, Optional

from runhouse.globals import obj_store

from runhouse.servers.http.http_utils import (
    deserialize_data,
    handle_exception_response,
    OutputType,
    Response,
    serialize_data,
)
from runhouse.servers.obj_store import ClusterServletSetupOption

from runhouse.utils import arun_in_thread, get_node_ip

logger = logging.getLogger(__name__)


def error_handling_decorator(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        ctx = kwargs.pop("ctx", None)
        ctx_token = None
        if ctx:
            ctx_token = obj_store.set_ctx(**ctx)

        serialization = kwargs.get("serialization", None)
        if "data" in kwargs:
            serialized_data = kwargs.get("data", None)
            deserialized_data = deserialize_data(serialized_data, serialization)
            kwargs["data"] = deserialized_data

        # If serialization is None, then we have not called this from the server,
        # so we should return the result of the function directly, or raise
        # the exception if there is one, instead of returning a Response object.
        try:
            output = await func(*args, **kwargs)
            if serialization is None or serialization == "none":
                return output
            if output is not None:
                serialized_data = serialize_data(output, serialization)
                return Response(
                    output_type=OutputType.RESULT_SERIALIZED,
                    data=serialized_data,
                    serialization=serialization,
                )
            else:
                return Response(
                    output_type=OutputType.SUCCESS,
                )
        except Exception as e:
            return handle_exception_response(e, traceback.format_exc(), serialization)
        finally:
            if ctx_token:
                obj_store.unset_ctx(ctx_token)

    return wrapper


class EnvServlet:
    async def __init__(self, env_name: str, *args, **kwargs):
        self.env_name = env_name

        await obj_store.ainitialize(
            self.env_name,
            has_local_storage=True,
            setup_cluster_servlet=ClusterServletSetupOption.GET_OR_FAIL,
        )

        # Ray defaults to setting OMP_NUM_THREADS to 1, which unexpectedly limit parallelism in user programs.
        # We delete it by default, but if we find that the user explicitly set it to another value, we respect that.
        # This is really only a factor if the user set the value inside the VM or container, or inside the base_env
        # which a cluster was initialized with. If they set it inside the env constructor and the env was sent to the
        # cluster normally with .to, it will be set after this point.
        # TODO this had no effect when we did it below where we set CUDA_VISIBLE_DEVICES, so we may need to move that
        #  here and mirror the same behavior (setting it based on the detected gpus in the whole cluster may not work
        #  for multinode, but popping it may also break things, it needs to be tested).
        num_threads = os.environ.get("OMP_NUM_THREADS")
        if num_threads is not None and num_threads != "1":
            os.environ["OMP_NUM_THREADS"] = num_threads
        else:
            os.environ["OMP_NUM_THREADS"] = ""

        self.output_types = {}
        self.thread_ids = {}

    ##############################################################
    # Methods to disable or enable den auth
    ##############################################################
    async def aset_cluster_config(self, cluster_config: Dict[str, Any]):
        obj_store.cluster_config = cluster_config

    async def aset_cluster_config_value(self, key: str, value: Any):
        obj_store.cluster_config[key] = value

    ##############################################################
    # Methods decorated with a standardized error decorating handler
    # These catch exceptions and wrap the output in a Response object.
    # They also handle arbitrary serialization and deserialization.
    # NOTE: These need to take in "data" and "serialization" as arguments
    # even if unused, because they are used by the decorator
    ##############################################################
    @error_handling_decorator
    async def aput_resource_local(
        self,
        data: Any,  # This first comes in a serialized format which the decorator re-populates after deserializing
        serialization: Optional[str] = None,
    ):
        resource_config, state, dryrun = tuple(data)
        return await obj_store.aput_resource_local(resource_config, state, dryrun)

    @error_handling_decorator
    async def aput_local(
        self, key: Any, data: Any, serialization: Optional[str] = None
    ):
        return await obj_store.aput_local(key, data)

    @error_handling_decorator
    async def acall_local(
        self,
        key: Any,
        method_name: str = None,
        data: Any = None,
        serialization: Optional[str] = None,
        run_name: Optional[str] = None,
        stream_logs: bool = False,
        remote: bool = False,
        ctx: Optional[dict] = None,
    ):
        if data is not None:
            args, kwargs = data.get("args", []), data.get("kwargs", {})
        else:
            args, kwargs = [], {}

        return await obj_store.acall_local(
            key,
            method_name,
            run_name=run_name,
            stream_logs=stream_logs,
            remote=remote,
            *args,
            **kwargs,
        )

    @error_handling_decorator
    async def aget_local(
        self,
        key: Any,
        default: Optional[Any] = None,
        serialization: Optional[str] = None,
        remote: bool = False,
    ):
        return obj_store.get_local(key, default=default, remote=remote)

    ##############################################################
    # IPC methods for interacting with local object store only
    # These do not catch exceptions, and do not wrap the output
    # in a Response object.
    ##############################################################
    async def akeys_local(self):
        return obj_store.keys_local()

    async def arename_local(self, key: Any, new_key: Any):
        return await obj_store.arename_local(key, new_key)

    async def acontains_local(self, key: Any):
        return obj_store.contains_local(key)

    async def apop_local(self, key: Any, *args):
        return await obj_store.apop_local(key, *args)

    async def adelete_local(self, key: Any):
        return await obj_store.adelete_local(key)

    async def aclear_local(self):
        return await obj_store.aclear_local()

    def _get_env_cpu_usage(self):

        import psutil

        from runhouse.utils import get_pid

        cluster_config = obj_store.cluster_config

        total_memory = psutil.virtual_memory().total
        node_ip = get_node_ip()
        env_servlet_pid = get_pid()

        if not cluster_config.get("resource_subtype") == "Cluster":
            stable_internal_external_ips = cluster_config.get(
                "stable_internal_external_ips"
            )
            for ips_set in stable_internal_external_ips:
                internal_ip, external_ip = ips_set[0], ips_set[1]
                if internal_ip == node_ip:
                    # head ip equals to cluster address equals to cluster.ips[0]
                    if ips_set[1] == cluster_config.get("ips")[0]:
                        node_name = f"head ({external_ip})"
                    else:
                        node_name = f"worker_{stable_internal_external_ips.index(ips_set)} ({external_ip}"
        else:
            # a case it is a BYO cluster, assume that first ip in the ips list is the head.
            ips = cluster_config.get("ips")
            if len(ips) == 1 or node_ip == ips[0]:
                node_name = f"head ({node_ip})"
            else:
                node_name = f"worker_{ips.index(node_ip)} ({node_ip})"

        try:
            env_servlet_process = psutil.Process(pid=env_servlet_pid)
            memory_size_bytes = env_servlet_process.memory_full_info().uss
            cpu_usage_percent = env_servlet_process.cpu_percent(interval=1)
            env_memory_usage = {
                "memory_size_bytes": memory_size_bytes,
                "cpu_usage_percent": cpu_usage_percent,
                "memory_percent_from_cluster": (memory_size_bytes / total_memory) * 100,
                "total_cluster_memory": total_memory,
                "env_memory_info": psutil.virtual_memory(),
            }
        except psutil.NoSuchProcess:
            env_memory_usage = {}

        return (env_memory_usage, node_name, total_memory, env_servlet_pid, node_ip)

    def _get_env_gpu_usage(self, env_servlet_pid: int):
        import subprocess

        from runhouse.resources.hardware.utils import detect_cuda_version_or_cpu

        # check it the cluster uses GPU or not
        if detect_cuda_version_or_cpu() == "cpu":
            return {}

        try:
            gpu_general_info = (
                subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=utilization.gpu,memory.total,count",
                        "--format=csv,noheader,nounits",
                    ],
                    stdout=subprocess.PIPE,
                )
                .stdout.decode("utf-8")
                .strip()
                .split(", ")
            )
            gpu_util_percent = float(gpu_general_info[0])
            total_gpu_memory = int(gpu_general_info[1]) * (1024**2)  # in bytes
            num_of_gpus = int(gpu_general_info[2])
            used_gpu_memory = 0  # in bytes

            env_gpu_usage = (
                subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-compute-apps=pid,gpu_uuid,used_memory",
                        "--format=csv,nounits",
                    ],
                    stdout=subprocess.PIPE,
                )
                .stdout.decode("utf-8")
                .strip()
                .split("\n")
            )
            for i in range(1, len(env_gpu_usage)):
                single_env_gpu_info = env_gpu_usage[i].strip().split(", ")
                if int(single_env_gpu_info[0]) == env_servlet_pid:
                    used_gpu_memory = used_gpu_memory + int(single_env_gpu_info[-1]) * (
                        1024**2
                    )
            if used_gpu_memory > 0:
                env_gpu_usage = {
                    "used_gpu_memory": used_gpu_memory,
                    "gpu_util_percent": gpu_util_percent / num_of_gpus,
                    "total_gpu_memory": total_gpu_memory,
                }
            else:
                env_gpu_usage = {}
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to get GPU usage for {self.env_name}: {e}")
            env_gpu_usage = {}

        return env_gpu_usage

    def _status_local_helper(self):
        objects_in_env_servlet = obj_store.keys_with_type()

        (
            env_memory_usage,
            node_name,
            total_memory,
            env_servlet_pid,
            node_ip,
        ) = self._get_env_cpu_usage()

        # Try loading GPU data (if relevant)
        env_gpu_usage = self._get_env_gpu_usage(int(env_servlet_pid))

        env_servlet_utilization_data = {
            "env_gpu_usage": env_gpu_usage,
            "node_ip": node_ip,
            "node_name": node_name,
            "pid": env_servlet_pid,
            "env_memory_usage": env_memory_usage,
        }

        return objects_in_env_servlet, env_servlet_utilization_data

    async def astatus_local(self):
        return await arun_in_thread(self._status_local_helper)
