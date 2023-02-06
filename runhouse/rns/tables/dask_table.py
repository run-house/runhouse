import logging
from typing import Optional

from .. import SkyCluster
from ..top_level_rns_fns import save

from .table import Table

logger = logging.getLogger(__name__)


class DaskTable(Table):
    DEFAULT_FOLDER_PATH = "/runhouse/dask-tables"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def from_config(config: dict, dryrun=True):
        """Load config values into the object."""
        if isinstance(config["fs"], dict):
            config["fs"] = SkyCluster.from_config(config["fs"], dryrun=dryrun)
        return DaskTable(**config)

    def save(
        self,
        name: Optional[str] = None,
        snapshot: bool = False,
        overwrite: bool = True,
        write_index: bool = False,
        **snapshot_kwargs,
    ):
        # https://docs.dask.org/en/stable/how-to/connect-to-remote-data.html
        if self._cached_data is not None:
            # https://stackoverflow.com/questions/72891631/how-to-remove-null-dask-index-from-parquet-file
            self.data.to_parquet(
                self.fsspec_url,
                write_index=write_index,
                storage_options=self.data_config,
            )
            logger.info(f"Saved {str(self)} to: {self.fsspec_url}")

        save(self, name=name, snapshot=snapshot, overwrite=overwrite, **snapshot_kwargs)

        return self

    def fetch(self, **kwargs):
        import dask.dataframe as dd

        # https://docs.dask.org/en/stable/generated/dask.dataframe.read_parquet.html
        self._cached_data = dd.read_parquet(
            self.fsspec_url, filesystem=self._folder.fsspec_fs
        )

        return self._cached_data
