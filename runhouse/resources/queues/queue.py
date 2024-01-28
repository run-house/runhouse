import queue
from typing import Any, List, Optional, Union

from runhouse import Cluster, Env
from runhouse.resources.module import Module


class Queue(Module):
    RESOURCE_TYPE = "queue"
    DEFAULT_CACHE_FOLDER = ".cache/runhouse/queues"

    """Simple dict wrapper to act as a queue. Wrapping this in an actor allows us to access
    it across Ray processes and nodes, and even keep some things pinned to Python memory."""

    def __init__(
        self,
        name: Optional[str] = None,
        system: Union[Cluster, str] = None,
        env: Optional[Env] = None,
        max_size: int = 0,
        persist: bool = False,  # TODO
        dryrun: bool = False,
        **kwargs,
    ):
        """
        Runhouse Queue object

        .. note::
                To build a Queue, please use the factory method :func:`queue`.
        """
        super().__init__(name=name, system=system, env=env, dryrun=dryrun, **kwargs)
        if not self._system or self._system.on_this_cluster():
            self.data = queue.Queue(maxsize=max_size)
            self.persist = persist
            self._subscribers = []

    def put(self, item: Any, block=True, timeout=None):
        self.data.put(item, block=block, timeout=timeout)
        for fn, out_queue in self._subscribers:
            res = fn(item)
            if out_queue:
                out_queue.put(res)

    def put_nowait(self, item: Any):
        self.data.put_nowait(item)

    def put_batch(self, items: List[Any], block=True, timeout=None):
        for item in items:
            self.data.put(item, block=block, timeout=timeout)

    def get(self, block=True, timeout=None):
        return self.data.get(block=block, timeout=timeout)

    def get_nowait(self):
        return self.data.get_nowait()

    def get_batch(self, batch_size: int, block=True, timeout=None):
        items = []
        for _ in range(batch_size):
            items.append(self.data.get(block=block, timeout=timeout))
        return items

    def __iter__(self):
        try:
            while True:
                yield self.get()
        except queue.Empty:
            return

    def qsize(self):
        return self.data.qsize()

    def empty(self):
        return self.data.empty()

    def full(self):
        return self.data.full()

    def task_done(self):
        return self.data.task_done()

    def join(self):
        return self.data.join()

    def subscribe(self, function, out_queue=None):
        self._subscribers.append((function, out_queue))
