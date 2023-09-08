import logging

from .table import Table

logger = logging.getLogger(__name__)


class RayTable(Table):
    DEFAULT_FOLDER_PATH = "/runhouse/ray-tables"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def write(self):
        if self._cached_data is not None:
            self._write_ray_dataset(self.data)
            logger.info(f"Saved {str(self)} to: {self.fsspec_url}")

        return self

    def fetch(self, **kwargs):
        import ray

        self._cached_data = ray.data.read_parquet(
            self.fsspec_url, filesystem=self._folder.fsspec_fs
        )
        return self._cached_data
