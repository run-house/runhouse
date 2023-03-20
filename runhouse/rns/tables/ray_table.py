import logging

from .. import OnDemandCluster

from .table import Table

logger = logging.getLogger(__name__)


class RayTable(Table):
    DEFAULT_FOLDER_PATH = "/runhouse/ray-tables"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def from_config(config: dict, dryrun=True):
        """Load config values into the object."""
        if isinstance(config["system"], dict):
            config["system"] = OnDemandCluster.from_config(
                config["system"], dryrun=dryrun
            )
        return RayTable(**config, dryrun=dryrun)

    def write(self):
        if self._cached_data is not None:
            self.write_ray_dataset(self.data)
            logger.info(f"Saved {str(self)} to: {self.fsspec_url}")

        return self

    def fetch(self, **kwargs):
        import ray

        self._cached_data = ray.data.read_parquet(
            self.fsspec_url, filesystem=self._folder.fsspec_fs
        )
        return self._cached_data
