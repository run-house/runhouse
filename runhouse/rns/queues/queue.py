import queue
from typing import Any, List, Optional, Union

from runhouse import Cluster
from runhouse.rns.resource import Resource


class Queue(Resource):
    RESOURCE_TYPE = "queue"
    DEFAULT_CACHE_FOLDER = ".cache/runhouse/queues"

    """Simple dict wrapper to act as key-value/object storage. Wrapping this in an actor allows us to
    access it across Ray processes and nodes, and even keep some things pinned to Python memory."""

    def __init__(
        self,
        name: Optional[str] = None,
        system: Union[None, str, Cluster] = None,
        max_size: int = 0,
        dryrun: bool = False,
        **kwargs,
    ):
        """
        Runhouse Blob object

        .. note::
                To build a Blob, please use the factory method :func:`blob`.
        """
        super().__init__(name=name, dryrun=dryrun)
        self._system = system
        self.data = queue.Queue(maxsize=max_size)

    def put(self, item: Any, block=True, timeout=None):
        self.data.put(item, block=block, timeout=timeout)

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
