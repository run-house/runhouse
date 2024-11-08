from concurrent.futures.thread import ThreadPoolExecutor
from typing import List, Optional

from runhouse.resources.distributed.supervisor import Supervisor

from runhouse.resources.envs.env import Env

from runhouse.resources.hardware import Cluster, OnDemandCluster

from runhouse.resources.module import Module


class PyTorchDistributed(Supervisor):
    def __init__(self, name, replicas: List[Module] = None, port=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self._replicas = replicas or []
        self._port = port

    def _compute_signature(self, rich=False):
        return self.local._replicas[0].signature(rich=rich)

    def _find_available_port_on_head_rank(self):
        find_available_port_cmd = "python -c \"import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()\""
        env_name = (
            self._replicas[0].env.name
            if isinstance(self._replicas[0].env, Env)
            else self._replicas[0].env
            if isinstance(self._replicas[0].env, str)
            else None
        )  # Todo make this run on the head node
        status_code, stdout, _ = self._replicas[0].system.run(
            find_available_port_cmd, env=env_name, require_outputs=True
        )[0]
        if status_code != 0:
            raise RuntimeError(f"Failed to find available port on head rank: {stdout}")
        return stdout

    def forward(self, item, timeout: Optional[int] = None, *args, **kwargs):
        port = self._port or self._find_available_port_on_head_rank()

        def run_on_replica(replica, rank):
            # Per https://pytorch.org/docs/stable/distributed.html#environment-variable-initialization
            master_addr = (
                self.system.internal_ips[0]
                if isinstance(self.system, OnDemandCluster)
                else self.system.address
                if isinstance(self.system, Cluster)
                else "localhost"
            )
            dist_config = {
                "MASTER_ADDR": master_addr,
                "MASTER_PORT": port,
                "RANK": str(rank),
                "WORLD_SIZE": str(len(self._replicas)),
            }
            if isinstance(replica.env, Env):
                env_name = replica.env.name
            elif isinstance(replica.env, str):
                env_name = replica.env
            else:
                raise ValueError("env must be an Env or a string")
            replica.system.set_process_env_vars(env_name, dist_config)
            method = getattr(replica, item)
            return method(*args, **kwargs)

        with ThreadPoolExecutor(max_workers=len(self._replicas)) as executor:
            res = executor.map(
                run_on_replica, self._replicas, range(len(self._replicas))
            )
            res = list(res)

        return res

    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)
