import asyncio
import copy
import os
import threading
from typing import Any, Dict, Optional

import psutil

from runhouse.constants import (
    DEFAULT_LOG_LEVEL,
    GPU_COLLECTION_INTERVAL,
    INCREASED_GPU_COLLECTION_INTERVAL,
    MAX_GPU_INFO_LEN,
    REDUCED_GPU_INFO_LEN,
)

from runhouse.globals import obj_store
from runhouse.logger import get_logger
from runhouse.resources.hardware import load_cluster_config_from_file
from runhouse.resources.hardware.ray_utils import kill_actors
from runhouse.resources.hardware.utils import is_gpu_cluster
from runhouse.servers.obj_store import ClusterServletSetupOption
from runhouse.utils import arun_in_thread

logger = get_logger(__name__)


class NodeServlet:
    async def __init__(self, node_name: str, **kwargs):
        # Still need the object store to communicate with ClusterServlet and other actors
        await obj_store.ainitialize(
            servlet_name=None,
            has_local_storage=None,
            setup_cluster_servlet=ClusterServletSetupOption.GET_OR_FAIL,
        )

        self.node_name = node_name

        cluster_config = load_cluster_config_from_file()
        self._cluster_config: Dict[str, Any] = cluster_config
        self._cluster_name = cluster_config.get("name")

        self.gpu_metrics = None  # will be updated only if this is a gpu cluster.
        # will be used when self.gpu_metrics will be updated by different threads.
        self.lock = threading.Lock()

        self._cluster_config["is_gpu"] = is_gpu_cluster()

        # creating periodic check loops
        if self._cluster_config.get("is_gpu"):
            logger.debug("Creating _periodic_gpu_check thread.")
            collect_gpu_thread = threading.Thread(
                target=self._periodic_gpu_check, daemon=True
            )
            collect_gpu_thread.start()

    async def arun_with_logs_local(
        self, cmd: str, require_outputs: bool = True, run_name: Optional[str] = None
    ):
        return await obj_store.arun_with_logs_local(
            cmd=cmd, require_outputs=require_outputs, run_name=run_name
        )

    async def alogs_local(self, run_name: str):
        async for ret_lines in obj_store.alogs_local(run_name=run_name, bash_run=True):
            yield ret_lines

    def _get_cpu_usage(self):

        relevant_memory_info = {
            "available": "free_memory",
            "percent": "used_memory_percent",
            "total": "total_memory",
            "used": "used_memory",
        }

        cpu_usage = psutil.cpu_percent(interval=0)
        cpu_memory_usage = psutil.virtual_memory()._asdict()
        cpu_memory_usage = {
            v: cpu_memory_usage[k] for k, v in relevant_memory_info.items()
        }
        cpu_memory_usage["utilization_percent"] = cpu_usage
        return cpu_memory_usage

    async def aget_cpu_usage(self):
        return await arun_in_thread(self._get_cpu_usage)

    def get_gpu_metrics(self, send_to_den: bool = False):
        with self.lock:
            current_gpu_metrics = copy.deepcopy(self.gpu_metrics)
            if send_to_den:
                # reset the gpu metrics, so we'll next time we send status to den, we'll send the updated gpu metrics
                self.gpu_metrics = None
        return current_gpu_metrics

    async def _aperiodic_gpu_check(self):
        """periodically collects cluster gpu usage"""
        import pynvml

        logger.debug("Started gpu usage collection")

        pynvml.nvmlInit()  # init nvidia ml info collection

        while True:
            try:
                gpu_count = pynvml.nvmlDeviceGetCount()
                with self.lock:
                    if not self.gpu_metrics:
                        self.gpu_metrics = {device: [] for device in range(gpu_count)}

                    for gpu_index in range(gpu_count):
                        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
                        util_info = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

                        total_memory = memory_info.total  # in bytes
                        used_memory = memory_info.used  # in bytes
                        free_memory = memory_info.free  # in bytes
                        utilization_percent = float(util_info.gpu)

                        # to reduce cluster memory usage (we are saving the gpu_usage info on the cluster),
                        # we save only the most updated gpu usage. If for some reason the size of updated_gpu_info is
                        # too big, we remove the older gpu usage info.
                        # This is relevant when using cluster.status() directly and not relying on status being sent to den.
                        updated_gpu_info = self.gpu_metrics[gpu_index]
                        if len(updated_gpu_info) + 1 > MAX_GPU_INFO_LEN:
                            updated_gpu_info = updated_gpu_info[REDUCED_GPU_INFO_LEN:]
                        updated_gpu_info.append(
                            {
                                "total_memory": total_memory,
                                "used_memory": used_memory,
                                "free_memory": free_memory,
                                "utilization_percent": utilization_percent,
                            }
                        )
                        self.gpu_metrics[gpu_index] = updated_gpu_info

            except Exception as e:
                logger.error(
                    f"{self._cluster_name}'s GPU metrics collection for node {self.node_name} failed: {str(e)}"
                )

                pynvml.nvmlShutdown()

                # killing the cluster servlet only if log level is debug
                log_level = os.getenv("RH_LOG_LEVEL") or DEFAULT_LOG_LEVEL
                if log_level.lower() == "debug":
                    kill_actors(gracefully=False, actor_name=self.node_name)
                else:
                    # increase sleep interval
                    await asyncio.sleep(INCREASED_GPU_COLLECTION_INTERVAL)

            finally:
                # collects gpu usage every 5 seconds.
                await asyncio.sleep(GPU_COLLECTION_INTERVAL)

    def _periodic_gpu_check(self):
        # This is only ever called once in its own thread, so we can do asyncio.run here instead of `sync_function`.
        asyncio.run(self._aperiodic_gpu_check())
