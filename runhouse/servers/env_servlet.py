import os
import traceback
from functools import wraps
from typing import Any, Dict, Iterable, Optional

from runhouse.constants import DEFAULT_LOG_LEVEL
from runhouse.globals import obj_store
from runhouse.logger import logger

from runhouse.servers.http.http_utils import (
    deserialize_data,
    handle_exception_response,
    OutputType,
    Response,
    serialize_data,
)
from runhouse.servers.obj_store import ClusterServletSetupOption

from runhouse.utils import arun_in_thread, get_node_ip


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
                if kwargs.get("remote"):
                    return Response(
                        output_type=OutputType.CONFIG,
                        data=output,
                    )
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

        logs_level = kwargs.get("logs_level", DEFAULT_LOG_LEVEL)
        logger.setLevel(logs_level)
        # self.logger = logger

        await obj_store.ainitialize(
            self.env_name,
            has_local_storage=True,
            setup_cluster_servlet=ClusterServletSetupOption.GET_OR_FAIL,
            logs_level=logs_level,
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

    def _get_cpu_usage(self, system_cpu_utilization: Iterable):
        """
        # TODO [SB]: add documentation
        :param system_cpu_utilization:
        :return:
        """
        cpus = []
        percent = []
        current_val = next(system_cpu_utilization, None)
        while current_val:
            percent.append(current_val.value)
            cpu = current_val.attributes.get("cpu")
            if cpu not in cpus:
                cpus.append(cpu)
            current_val = next(system_cpu_utilization, None)
        return round(sum(percent) / len(cpus), 2)

    def _get_system_memory_usage(
        self, system_memory_usage: Iterable, system_memory_utilization: Iterable
    ):
        """
        # TODO [SB]: add documentation
        :param system_memory_usage: in bytes
        :param system_memory_utilization: percent, float
        :return: dict with the following keys: total_memory, used_memory, free_memory, percent
        """
        memory_usage = {
            "total_memory": 0,  # int
            "used_memory": 0,  # int
            "free_memory": 0,  # int
            "percent": 0,  # float
        }

        system_memory_usage_val = next(system_memory_usage, None)
        while system_memory_usage_val:
            value = system_memory_usage_val.value
            value_type = system_memory_usage_val.attributes.get("state")
            if value_type == "used":
                memory_usage["used_memory"] += value
            elif value_type == "free":
                memory_usage["free_memory"] += value
            elif value_type == "total":
                memory_usage["total_memory"] += value
            system_memory_usage_val = next(system_memory_usage, None)

        system_memory_utilization_val = next(system_memory_utilization, None)
        while system_memory_utilization_val:
            value = system_memory_utilization_val.value
            value_type = system_memory_utilization_val.attributes.get("state")
            if value_type == "used":
                memory_usage["percent"] = round(value, 4)
            system_memory_utilization_val = next(system_memory_utilization, None)

        return memory_usage

    def _get_env_memory_usage(self, env_memory_usage: Iterable):
        """
        # TODO [SB]: add documentation
        :param env_memory_usage:
        :return: dict with the following keys: total_memory, used_memory, free_memory, percent
        """
        # rss is the Resident Set Size, which is the actual physical memory the process is using
        # vms is the Virtual Memory Size which is the virtual memory that process is using
        # returning rss, because it is more accurate.
        used_memory = 0

        system_memory_usage_val = next(env_memory_usage, None)
        while system_memory_usage_val:
            value = system_memory_usage_val.value
            value_type = system_memory_usage_val.attributes.get("state")
            if value_type == "rss":
                used_memory = value
                break

        return used_memory

    def _get_env_cpu_usage(self, cluster_config: dict = None):

        from runhouse.utils import get_pid

        cluster_config = cluster_config or obj_store.cluster_config

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
                        node_name = f"worker_{stable_internal_external_ips.index(ips_set)} ({external_ip})"
        else:
            # a case it is a BYO cluster, assume that first ip in the ips list is the head.
            ips = cluster_config.get("ips")
            if len(ips) == 1 or node_ip == ips[0]:
                node_name = f"head ({node_ip})"
            else:
                node_name = f"worker_{ips.index(node_ip)} ({node_ip})"

        try:
            from opentelemetry.instrumentation.system_metrics import (
                SystemMetricsInstrumentor,
            )

            servlet_name = f"{self.env_name}_env_servlet"
            CPU_FIELDS = (
                "idle user system irq softirq nice iowait steal interrupt dpc".split()
            )
            configuration = {
                "system.cpu.utilization": CPU_FIELDS,
                "system.memory.usage": ["used", "free", "cached", "total"],
                "system.memory.utilization": ["used", "free", "cached"],
                "process.runtime.memory": ["rss", "vms"],
                "process.runtime.cpu.utilization": None,
            }
            inst = SystemMetricsInstrumentor(
                config=configuration, labels={"servlet_name": servlet_name}
            )
            inst.instrument()

            cpu_usage = self._get_cpu_usage(inst._get_runtime_cpu_utilization(None))
            memory_system_usage = self._get_system_memory_usage(
                inst._get_system_memory_usage(None),
                inst._get_system_memory_utilization(None),
            )
            total_memory = memory_system_usage.get("total_memory")
            env_system_usage = self._get_env_memory_usage(
                inst._get_runtime_memory(None)
            )

            env_memory_usage = {
                "used_memory": env_system_usage,
                "utilization_percent": cpu_usage,
                "total_memory": total_memory,
            }
        except Exception:
            env_memory_usage = {}
            total_memory = 0

        return (env_memory_usage, node_name, total_memory, env_servlet_pid, node_ip)

    def _get_env_gpu_usage(self, env_servlet_pid: int):
        import subprocess

        try:
            gpu_general_info = (
                subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=utilization.gpu,memory.total,count,utilization.memory",
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
            memory_utilization_percent = int(
                gpu_general_info[3]
            )  # in %, meaning 0 <= val <= 100, out of total gpu memory
            allocated_gpu_memory = 0  # in bytes

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
                    allocated_gpu_memory = allocated_gpu_memory + int(
                        single_env_gpu_info[-1]
                    ) * (1024**2)
            used_memory = round(memory_utilization_percent / 100, 2) * total_gpu_memory
            if allocated_gpu_memory > 0:
                env_gpu_usage = {
                    "allocated_memory": allocated_gpu_memory,
                    "total_memory": total_gpu_memory,
                    "used_memory": used_memory,  # in bytes
                    "utilization_percent": gpu_util_percent / num_of_gpus,
                    "memory_percent_allocated": round(
                        used_memory / allocated_gpu_memory, 4
                    )
                    * 100,
                }
            else:
                env_gpu_usage = {}
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to get GPU usage for {self.env_name}: {e}")
            env_gpu_usage = {}

        return env_gpu_usage

    def _status_local_helper(self):
        objects_in_env_servlet = obj_store.keys_with_info()
        cluster_config = obj_store.cluster_config

        (
            env_memory_usage,
            node_name,
            total_memory,
            env_servlet_pid,
            node_ip,
        ) = self._get_env_cpu_usage(cluster_config)

        # Try loading GPU data (if relevant)
        env_gpu_usage = (
            self._get_env_gpu_usage(int(env_servlet_pid))
            if cluster_config.get("has_cuda", False)
            else {}
        )

        env_servlet_utilization_data = {
            "env_gpu_usage": env_gpu_usage,
            "node_ip": node_ip,
            "node_name": node_name,
            "pid": env_servlet_pid,
            "env_cpu_usage": env_memory_usage,
        }

        return objects_in_env_servlet, env_servlet_utilization_data

    async def astatus_local(self):
        return await arun_in_thread(self._status_local_helper)
