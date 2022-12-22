from pathlib import Path
from typing import Optional, List

from .table import Table
from ..top_level_rns_fns import save


class PandasTable(Table):
    DEFAULT_FOLDER_PATH = '/runhouse/pandas-tables'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def url(self):
        return self._folder.url

    @url.setter
    def url(self, new_url):
        self._folder.url = str(Path(new_url).parent)

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
            self.data.to_parquet(self.fsspec_url)

        save(self,
             save_to=save_to if save_to is not None else self.save_to,
             snapshot=snapshot,
             overwrite=overwrite,
             **snapshot_kwargs)

    def fetch(self, **kwargs):
        self.import_package('pandas')
        import pandas as pd
        # https://pandas.pydata.org/docs/reference/api/pandas.read_parquet.html
        self._cached_data = pd.read_parquet(self.fsspec_url, storage_options=self.data_config)
        return self._cached_data

