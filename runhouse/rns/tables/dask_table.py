from typing import Optional, List

from .table import Table
from .. import Cluster
from ..top_level_rns_fns import save


class DaskTable(Table):
    DEFAULT_FOLDER_PATH = '/runhouse/dask-tables'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def from_config(config: dict, dryrun=True):
        """ Load config values into the object. """
        if isinstance(config['fs'], dict):
            config['fs'] = Cluster.from_config(config['fs'], dryrun=dryrun)
        return DaskTable(**config)

    def save(self,
             name: Optional[str] = None,
             snapshot: bool = False,
             overwrite: bool = True,
             **snapshot_kwargs):
        # https://docs.dask.org/en/stable/how-to/connect-to-remote-data.html
        if self._cached_data is not None:
            self.data.to_parquet(self._folder.fsspec_url)

        save(self,
             name=name,
             snapshot=snapshot,
             overwrite=overwrite,
             **snapshot_kwargs)

    def fetch(self, **kwargs):
        import dask.dataframe as dd
        # https://docs.dask.org/en/stable/generated/dask.dataframe.read_parquet.html
        self._cached_data = dd.read_parquet(self._folder.fsspec_url, storage_options=self.data_config)
        return self._cached_data
