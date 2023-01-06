import uuid
from typing import Optional, List

from .table import Table
from ..top_level_rns_fns import save


class PandasTable(Table):
    DEFAULT_FOLDER_PATH = '/runhouse/pandas-tables'
    STREAM_FORMAT = 'pandas'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # PyArrow will create this file and suffix for us, but with Pandas we need to do it ourselves.
        if self.file_name is None:
            self.file_name = f'{uuid.uuid4().hex}.parquet'

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
        if self._cached_data is not None:
            # TODO make overwrite work
            self.data.to_parquet(self.fsspec_url,
                                 partition_cols=self.partition_cols,
                                 storage_options=self.data_config)

        super().save(name=name, snapshot=snapshot, save_to=save_to, overwrite=overwrite, **snapshot_kwargs)

    def fetch(self, **kwargs):
        import pandas as pd
        # https://pandas.pydata.org/docs/reference/api/pandas.read_parquet.html
        self._cached_data = pd.read_parquet(self.fsspec_url, storage_options=self.data_config)
        return self._cached_data
