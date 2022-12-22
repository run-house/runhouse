from pathlib import Path
import fsspec

from typing import Optional, List

from .table import Table
from ..top_level_rns_fns import save


class RayTable(Table):
    DEFAULT_FOLDER_PATH = '/runhouse/ray-tables'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fsspec_fs = fsspec.filesystem(self.fs, **self.data_config)

    @property
    def url(self):
        return self._folder.url

    @url.setter
    def url(self, new_url):
        self._folder.url = str(Path(new_url).parent)

    @staticmethod
    def from_config(config: dict, **kwargs):
        """ Load config values into the object. """
        return RayTable(**config)

    def save(self,
             name: Optional[str] = None,
             snapshot: bool = False,
             save_to: Optional[List[str]] = None,
             overwrite: bool = False,
             **snapshot_kwargs):
        if self._cached_data is None or overwrite:
            self.data.write_parquet(self.fsspec_url, filesystem=self.fsspec_fs)

        save(self,
             save_to=save_to if save_to is not None else self.save_to,
             snapshot=snapshot,
             overwrite=overwrite,
             **snapshot_kwargs)

    def fetch(self, **kwargs):
        self.import_package('ray')
        import ray
        # TODO [JL] This doesn't work, see the ray docs
        self._cached_data = ray.data.read_parquet(self.fsspec_url, filesystem=self.fsspec_fs)
        return self._cached_data
