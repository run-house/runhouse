import copy
import os
import sys
import threading
import time
import traceback
from functools import wraps
from typing import Any, Dict, Optional

from runhouse import configs

from runhouse.constants import (
    DEFAULT_STATUS_CHECK_INTERVAL,
    GPU_COLLECTION_INTERVAL,
    MAX_GPU_INFO_LEN,
    REDUCED_GPU_INFO_LEN,
)

from runhouse.globals import obj_store
from runhouse.logger import get_logger

from runhouse.resources.hardware.utils import is_gpu_cluster

from runhouse.servers.http.http_utils import (
    CreateProcessParams,
    deserialize_data,
    handle_exception_response,
    OutputType,
    Response,
    serialize_data,
)
from runhouse.servers.obj_store import ClusterServletSetupOption

from runhouse.utils import (
    arun_in_thread,
    get_node_ip,
    get_pid,
    parse_gpu_usage,
    ServletType,
)

logger = get_logger(__name__)


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


class Servlet:
    async def __init__(
        self, create_process_params: CreateProcessParams, *args, **kwargs
    ):
        import psutil

        self.env_name = create_process_params.name

        await obj_store.ainitialize(
            self.env_name,
            has_local_storage=True,
            setup_cluster_servlet=ClusterServletSetupOption.GET_OR_FAIL,
            create_process_params=create_process_params,
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

        self.pid = get_pid()
        self.process = psutil.Process(pid=self.pid)

        self.gpu_metrics = None  # will be updated only if this is a gpu cluster.
        self.lock = (
            threading.Lock()
        )  # will be used when self.gpu_metrics will be updated by different threads.

        self.is_gpu_cluster = is_gpu_cluster()
        if self.is_gpu_cluster:
            logger.debug("Creating _periodic_gpu_check thread.")
            collect_gpu_thread = threading.Thread(
                target=self._collect_process_gpu_usage, daemon=True
            )
            collect_gpu_thread.start()

    ##############################################################
    # Simple system interactions for this worker process
    ##############################################################
    async def aset_env_vars(self, env_vars: Dict[str, str]):
        obj_store.set_process_env_vars_local(env_vars)

    async def aprepend_to_sys_path(self, path: str):
        sys.path.insert(0, path)

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

    async def arun_with_logs_local(
        self, cmd: str, require_outputs: bool = True, run_name: Optional[str] = None
    ):
        return await obj_store.arun_with_logs_local(
            cmd=cmd, require_outputs=require_outputs, run_name=run_name
        )

    async def alogs_local(
        self,
        run_name: str,
    ):
        async for ret_lines in obj_store.alogs_local(run_name=run_name):
            yield ret_lines

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
    def keys_local(self):
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

    def _get_process_cpu_usage(self, cluster_config: dict = None):
        import psutil

        total_memory = psutil.virtual_memory().total
        internal_node_ip = get_node_ip()

        if not cluster_config.get("resource_subtype") == "Cluster":
            compute_properties = cluster_config.get("compute_properties", {})
            if compute_properties.get("internal_ips"):
                internal_ips = compute_properties.get("internal_ips", [])
                ips = compute_properties.get("ips", [])
            else:
                stable_internal_external_ips = cluster_config.get(
                    "stable_internal_external_ips", []
                )
                internal_ips = [int_ip for int_ip, _ in stable_internal_external_ips]
                ips = [ip for _, ip in stable_internal_external_ips]
        else:
            # if it is a BYO cluster, assume that first ip in the ips list is the head.
            internal_ips = ips = cluster_config.get("ips", [])

        node_index = (
            0 if len(internal_ips) == 1 else internal_ips.index(internal_node_ip)
        )
        node_name = f"worker_{node_index}"

        try:

            memory_size_bytes = self.process.memory_full_info().uss
            used_memory_percent = round(
                self.process.memory_percent(memtype="uss"), 2
            )  # value is between 0 and 100
            cpu_usage_percent = self.process.cpu_percent(
                interval=0
            )  # value is between 0 and 100
            process_memory_usage = {
                "used_memory": memory_size_bytes,
                "utilization_percent": cpu_usage_percent,
                "total_memory": total_memory,
                "used_memory_percent": used_memory_percent,
            }
        except psutil.NoSuchProcess:
            process_memory_usage = {}

        return (
            process_memory_usage,
            node_name,
            total_memory,
            self.pid,
            ips[
                node_index
            ],  # getting the "external" ip, so it will match the ip we see in the workers field in the cluster status output
            node_index,
        )

    def _get_process_gpu_usage(self):

        collected_gpus_info = copy.deepcopy(self.gpu_metrics)

        if not collected_gpus_info or not collected_gpus_info.get(0):
            return None

        return parse_gpu_usage(
            collected_gpu_info=collected_gpus_info, servlet_type=ServletType.process
        )

    def _collect_process_gpu_usage(self):
        """periodically collects env gpu usage"""
        import pynvml

        pynvml.nvmlInit()  # init nvidia ml info collection

        while True:
            try:
                gpu_count = pynvml.nvmlDeviceGetCount()
                with self.lock:
                    if not self.gpu_metrics:
                        self.gpu_metrics: Dict[int, list[Dict[str, int]]] = {
                            device: [] for device in range(gpu_count)
                        }

                    for gpu_index in range(gpu_count):
                        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
                        processes = pynvml.nvmlDeviceGetComputeRunningProcesses_v3(
                            handle
                        )
                        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                        if processes:
                            for p in processes:
                                if p.pid == self.pid:
                                    used_memory = p.usedGpuMemory  # in bytes
                                    total_memory = memory_info.total  # in bytes
                                    current_gpu_metrics: list[
                                        Dict[str, int]
                                    ] = self.gpu_metrics[gpu_index]
                                    # to reduce cluster memory usage (we are saving the gpu_usage info on the cluster),
                                    # we save only the most updated gpu usage. If for some reason the size of updated_gpu_info is
                                    # too big, we remove the older gpu usage info.
                                    # This is relevant when using cluster.status() directly and not relying on status being sent to den.
                                    if len(current_gpu_metrics) + 1 > MAX_GPU_INFO_LEN:
                                        current_gpu_metrics = current_gpu_metrics[
                                            REDUCED_GPU_INFO_LEN:
                                        ]
                                    current_gpu_metrics.append(
                                        {
                                            "used_memory": used_memory,
                                            "total_memory": total_memory,
                                        }
                                    )
                                    self.gpu_metrics[gpu_index] = current_gpu_metrics
            except Exception as e:
                logger.error(str(e))
                pynvml.nvmlShutdown()
                break
            finally:
                # collects gpu usage every 5 seconds.
                time.sleep(GPU_COLLECTION_INTERVAL)

    def _status_local_helper(self):
        objects_in_servlet = obj_store.keys_with_info()
        cluster_config = obj_store.cluster_config

        (
            process_memory_usage,
            node_name,
            total_memory,
            servlet_pid,
            node_ip,
            node_index,
        ) = self._get_process_cpu_usage(cluster_config)

        cluster_config = obj_store.cluster_config
        interval_size = cluster_config.get(
            "status_check_interval", DEFAULT_STATUS_CHECK_INTERVAL
        )

        # TODO: [sb]: once introduced, we could use ClusterServlet _cluster_periodic_thread_alive() to replace the
        #  'should_send_status_and_logs_to_den' logic below.
        # Only if one of these is true, do we actually need to get the status from each Servlet
        should_send_status_and_logs_to_den: bool = (
            configs.token is not None and interval_size != -1
        )

        servlet_utilization_data = {
            "node_ip": node_ip,
            "node_name": node_name,
            "node_index": node_index,
            "process_cpu_usage": process_memory_usage,
            "pid": servlet_pid,
        }

        if self.is_gpu_cluster:
            # Try loading GPU data
            process_gpu_usage = self._get_process_gpu_usage()
            if process_gpu_usage:
                if should_send_status_and_logs_to_den:
                    # reset the gpu_info only if the current env_gpu collection will be sent to den. Otherwise, keep collecting it.
                    with self.lock:
                        self.gpu_metrics = None

                servlet_utilization_data["process_gpu_usage"] = process_gpu_usage

        return objects_in_servlet, servlet_utilization_data

    async def astatus_local(self):
        return await arun_in_thread(self._status_local_helper)

    async def aremove_servlet_from_cache(self, servlet_name):
        logger.info(f"Removing {servlet_name} from {self.env_name}")
        await obj_store.adelete_servlet_from_cache(servlet_name)
