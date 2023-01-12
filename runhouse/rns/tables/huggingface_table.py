from typing import Optional, List

from .table import Table
from .. import Cluster


class HuggingFaceTable(Table):
    DEFAULT_FOLDER_PATH = '/runhouse/huggingface-tables'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def from_config(config: dict, dryrun=True):
        """ Load config values into the object. """
        if isinstance(config['fs'], dict):
            config['fs'] = Cluster.from_config(config['fs'], dryrun=dryrun)
        return HuggingFaceTable(**config, dryrun=dryrun)

    def save(self,
             name: Optional[str] = None,
             snapshot: bool = False,
             save_to: Optional[List[str]] = None,
             overwrite: bool = False,
             **snapshot_kwargs):

        hf_dataset = None
        if self._cached_data is not None:
            import datasets
            if isinstance(self.data, datasets.Dataset):
                # Convert to a pyarrow table before saving to the relevant file system
                pa_table = self.data.data.table
                self.data, hf_dataset = pa_table, self.data
            elif isinstance(self.data, datasets.DatasetDict):
                # TODO [JL] Add support for dataset dict
                raise NotImplementedError('Runhouse does not currently support DatasetDict objects, please convert to '
                                          'a Dataset before saving.')

        super().save(name=name,
                     save_to=save_to if save_to is not None else self.save_to,
                     snapshot=snapshot,
                     overwrite=overwrite,
                     **snapshot_kwargs)

        # Restore the original dataset
        if hf_dataset is not None:
            self.data = hf_dataset

    def fetch(self, **kwargs):
        # TODO [JL] Add support for dataset dict
        from datasets import Dataset
        # Read as pyarrow table, then convert back to HF dataset
        pa_table = super().fetch(**kwargs)
        self._cached_data = Dataset(pa_table)
        return self._cached_data

    @staticmethod
    def to_dataset(data):
        """Convert to a huggingface dataset"""
        from datasets import Dataset
        return Dataset.from_pandas(data.to_pandas())
