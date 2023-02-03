from typing import List, Optional

from .. import SkyCluster
from ..top_level_rns_fns import save

from .table import Table


class RayTable(Table):
    DEFAULT_FOLDER_PATH = "/runhouse/ray-tables"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def from_config(config: dict, dryrun=True):
        """Load config values into the object."""
        if isinstance(config["fs"], dict):
            config["fs"] = SkyCluster.from_config(config["fs"], dryrun=dryrun)
        return RayTable(**config)

    def save(
        self,
        name: Optional[str] = None,
        snapshot: bool = False,
        overwrite: bool = True,
        **snapshot_kwargs
    ):
        if self._cached_data is not None:
            self.data.write_parquet(self._folder.fsspec_url)

        save(self, name=name, snapshot=snapshot, overwrite=overwrite, **snapshot_kwargs)

    def fetch(self, **kwargs):
        import ray

        self._cached_data = ray.data.read_parquet(
            self._folder.fsspec_url, **self.data_config
        )
        return self._cached_data
