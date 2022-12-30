from typing import Optional, List

from .table import Table
from ..top_level_rns_fns import save


class CudfTable(Table):
    DEFAULT_FOLDER_PATH = '/runhouse/cudf-tables'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def from_config(config: dict, **kwargs):
        """ Load config values into the object. """
        return CudfTable(**config)

    def save(self,
             name: Optional[str] = None,
             snapshot: bool = False,
             save_to: Optional[List[str]] = None,
             overwrite: bool = False,
             **snapshot_kwargs):
        # https://docs.rapids.ai/api/cudf/nightly/api_docs/api/cudf.dataframe.to_parquet
        if self._cached_data is None or overwrite:
            self.data.to_parquet(self.fsspec_url)

        save(self,
             name=name,
             save_to=save_to if save_to is not None else self.save_to,
             snapshot=snapshot,
             overwrite=overwrite,
             **snapshot_kwargs)

    def fetch(self, **kwargs):
        self.import_package('cudf')

        import cudf
        # https://docs.rapids.ai/api/cudf/nightly/api_docs/api/cudf.read_parquet.html
        self._cached_data = cudf.read_parquet(self.fsspec_url)
        return self._cached_data
