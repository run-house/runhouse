from typing import Optional

import psutil

from runhouse.globals import obj_store
from runhouse.servers.obj_store import ClusterServletSetupOption
from runhouse.utils import arun_in_thread


class NodeServlet:
    async def __init__(self):
        # Still need the object store to communicate with ClusterServlet and other actors
        await obj_store.ainitialize(
            servlet_name=None,
            has_local_storage=None,
            setup_cluster_servlet=ClusterServletSetupOption.GET_OR_FAIL,
        )

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
