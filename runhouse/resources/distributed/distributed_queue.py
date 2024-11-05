import time
from typing import List, Optional

from runhouse.resources.distributed.supervisor import Supervisor
from runhouse.resources.functions.function import Function

from runhouse.resources.module import Module


class DistributedQueue(Supervisor):
    def __init__(
        self, name, replicas: List[Module] = None, max_concurrency=1, **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self._replicas = replicas or []
        self._max_concurrency = max_concurrency
        self._available_replicas = list(
            range(len(self._replicas) * self._max_concurrency)
        )

    def signature(self, rich=False):
        return self.local._replicas[0].signature(rich=rich)

    def forward(self, item, timeout: Optional[int] = None, *args, **kwargs):
        time_waited = 0
        while not self._available_replicas:
            if timeout == 0:
                raise TimeoutError("No available replicas.")
            if timeout is not None and time_waited >= timeout:
                raise TimeoutError("Timed out waiting for a replica to be available.")
            time.sleep(0.25)
            time_waited += 0.25
        worker_idx = self._available_replicas.pop(0)
        worker = self._replicas[worker_idx // self._max_concurrency]
        method = getattr(worker, item)
        res = method(*args, **kwargs)
        self._available_replicas.append(worker_idx)
        return res

    def __call__(self, *args, **kwargs):
        if isinstance(self._replicas[0], Function):
            return self.call(*args, **kwargs)
        else:
            raise NotImplementedError(
                "DistributedQueue.__call__ can only be called on Function replicas."
            )
