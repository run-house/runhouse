from runhouse.resources.distributed.supervisor import Supervisor
from runhouse.resources.functions.function import Function

from runhouse.resources.module import Module


class DaskDistributed(Supervisor):
    def __init__(self, name, module: Module = None, **kwargs):
        super().__init__(name=name, **kwargs)
        self._module = module

    @staticmethod
    def setup_cluster(cluster, num_replicas=None, replicas_per_node=None):
        # Run the dask distributed setup on the head node first
        cluster.run("dask scheduler --port 8786", node=cluster.ips[0])
        # Note: We need to do this on the head node too, because this creates all the worker processes
        tot_workers = num_replicas or (replicas_per_node * len(cluster.ips))
        nworkers_list = []
        if tot_workers:
            nworkers_list = [num_replicas // len(cluster.ips)] * len(cluster.ips)
            remainder = num_replicas % len(cluster.ips)
            for i in range(remainder):
                nworkers_list[i] += 1
        for node in cluster.ips:
            nworkers = nworkers_list.pop(0, "auto")
            cluster.run(
                f"dask worker tcp://{cluster.internal_ips[0]:8786} --nworkers {nworkers}",
                node=node,
            )

    @staticmethod
    def teardown_cluster(cluster):
        cluster.run("pkill -f dask", node="all")

    @classmethod
    def restart_dask(cls, cluster):
        cls.teardown_cluster(cluster)
        cls.setup_cluster(cluster)

    def signature(self, rich=False):
        return self.local._module.signature(rich=rich)

    def forward(self, item, *args, **kwargs):
        method = getattr(self._module, item)
        return method(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        if isinstance(self._module, Function):
            return self.call(*args, **kwargs)
        else:
            raise NotImplementedError(
                "DaskDistributed.__call__ can only be called on Function replicas."
            )
