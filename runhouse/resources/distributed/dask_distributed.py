from runhouse.constants import DEFAULT_DASK_PORT
from runhouse.resources.distributed.supervisor import Supervisor

from runhouse.resources.module import Module


class DaskDistributed(Supervisor):
    def __init__(
        self,
        name,
        module: Module = None,
        port: int = DEFAULT_DASK_PORT,
        client_timeout="3s",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self._module = module
        self._dask_port = port
        self._dask_client = None
        self._client_timeout = client_timeout

    def _compute_signature(self, rich=False):
        return self.local._module.signature(rich=rich)

    def forward(self, item, *args, **kwargs):
        if not self._dask_client:
            self._dask_client = self.system.connect_dask(
                port=self._dask_port, client_timeout=self._client_timeout
            )
        method = getattr(self._module, item)
        return method(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)
