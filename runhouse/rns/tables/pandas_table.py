from pathlib import Path
from typing import Optional, List

from .table import Table
from ..top_level_rns_fns import save
from ... import rns_client


class PandasTable(Table):
    DEFAULT_FOLDER_PATH = '/runhouse/pandas-tables'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def root_path(self):
        """Add suffix to the URL when using a local filesystem to prevent IsADirectoryError.
        PyArrow will create this suffix for us, but with Pandas we need to do it ourselves."""
        fsspec_url = self._folder.fsspec_url
        if self.fs != rns_client.DEFAULT_FS:
            return fsspec_url

        return fsspec_url if Path(fsspec_url).suffix else f'{fsspec_url}/{Path(fsspec_url).stem}.parquet'

    @staticmethod
    def from_config(config: dict, **kwargs):
        """ Load config values into the object. """
        return PandasTable(**config)

    def save(self,
             name: Optional[str] = None,
             snapshot: bool = False,
             save_to: Optional[List[str]] = None,
             overwrite: bool = False,
             **snapshot_kwargs):
        if self._cached_data is None or overwrite:
            self.data.to_parquet(self.root_path,
                                 partition_cols=self.partition_cols,
                                 storage_options=self.data_config)

        save(self,
             name=name,
             save_to=save_to if save_to is not None else self.save_to,
             snapshot=snapshot,
             overwrite=overwrite,
             **snapshot_kwargs)

    def fetch(self, **kwargs):
        self.import_package('pandas')

        import pandas as pd
        # https://pandas.pydata.org/docs/reference/api/pandas.read_parquet.html
        self._cached_data = pd.read_parquet(self.root_path, storage_options=self.data_config)
        return self._cached_data
