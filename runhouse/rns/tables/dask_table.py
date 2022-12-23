from pathlib import Path

import fsspec

from typing import Optional, List

from .table import Table
from ..top_level_rns_fns import save


class DaskTable(Table):
    DEFAULT_FOLDER_PATH = '/runhouse/dask-tables'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def from_config(config: dict, **kwargs):
        """ Load config values into the object. """
        return DaskTable(**config)

    def save(self,
             name: Optional[str] = None,
             snapshot: bool = False,
             save_to: Optional[List[str]] = None,
             overwrite: bool = False,
             **snapshot_kwargs):
        # https://docs.dask.org/en/stable/how-to/connect-to-remote-data.html
        if self._cached_data is None or overwrite:
            self.data.to_parquet(self._folder.fsspec_url)

        save(self,
             save_to=save_to if save_to is not None else self.save_to,
             snapshot=snapshot,
             overwrite=overwrite,
             **snapshot_kwargs)

    def fetch(self, **kwargs):
        self.import_package('dask')
        import dask.dataframe as dd
        # https://docs.dask.org/en/stable/generated/dask.dataframe.read_parquet.html
        self._cached_data = dd.read_parquet(self._folder.fsspec_url, storage_options=self.data_config)
        return self._cached_data
