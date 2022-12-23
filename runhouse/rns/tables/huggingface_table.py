from pathlib import Path
import fsspec
from typing import Optional, List

from .table import Table
from ..top_level_rns_fns import save


class HuggingFaceTable(Table):
    DEFAULT_FOLDER_PATH = '/runhouse/huggingface-tables'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def from_config(config: dict, **kwargs):
        """ Load config values into the object. """
        return HuggingFaceTable(**config)

    def save(self,
             name: Optional[str] = None,
             snapshot: bool = False,
             save_to: Optional[List[str]] = None,
             overwrite: bool = False,
             **snapshot_kwargs):
        # https://huggingface.co/docs/datasets/v2.8.0/en/process#save
        if self._cached_data is None or overwrite:
            self.data.save_to_disk(self.fsspec_url, fs=self._folder.fsspec_fs)

        save(self,
             save_to=save_to if save_to is not None else self.save_to,
             snapshot=snapshot,
             overwrite=overwrite,
             **snapshot_kwargs)

    def fetch(self, **kwargs):
        self.import_package('datasets')
        from datasets import load_from_disk
        # TODO [JL] Not currently working - seems to be some inconsistencies in the fsspec URL sometimes having
        #   an extra slash (ex: 's3:///runhouse-tests/huggingface')
        self._cached_data = load_from_disk(self.fsspec_url, fs=self._folder.fsspec_fs)
        return self._cached_data
