from runhouse.globals import obj_store
from runhouse.servers.obj_store import ClusterServletSetupOption
from runhouse.utils import run_with_logs


class NodeServlet:
    async def __init__(self):
        # Still need the object store to communicate with ClusterServlet and other actors
        await obj_store.ainitialize(
            servlet_name=None,
            has_local_storage=None,
            setup_cluster_servlet=ClusterServletSetupOption.GET_OR_FAIL,
        )

    async def arun_with_logs(self, cmd: str):
        return run_with_logs(cmd, stream_logs=True, require_outputs=True)[:2]
