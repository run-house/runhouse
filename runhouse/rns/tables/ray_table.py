import logging
from typing import Optional

from .. import SkyCluster
from ..top_level_rns_fns import save

from .table import Table

logger = logging.getLogger(__name__)


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
        **snapshot_kwargs,
    ):
        if self._cached_data is not None:
            self.write_ray_dataset(self.data)
            logger.info(f"Saved {str(self)} to: {self.fsspec_url}")

        save(self, name=name, snapshot=snapshot, overwrite=overwrite, **snapshot_kwargs)

        return self

    def fetch(self, **kwargs):
        import ray

        self._cached_data = ray.data.read_parquet(
            self.fsspec_url, filesystem=self._folder.fsspec_fs
        )
        return self._cached_data
