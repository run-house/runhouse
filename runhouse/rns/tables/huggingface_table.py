from typing import Optional, List
import pandas as pd

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
             overwrite: bool = True,
             **snapshot_kwargs):

        hf_dataset = None
        if self._cached_data is not None:
            import datasets
            if isinstance(self.data, datasets.Dataset):
                # Convert to a pyarrow table before saving to the relevant file system
                arrow_table = self.data.data.table
                self.data, hf_dataset = arrow_table, self.data
            elif isinstance(self.data, datasets.DatasetDict):
                # TODO [JL] Add support for dataset dict
                raise NotImplementedError('Runhouse does not currently support DatasetDict objects, please convert to '
                                          'a Dataset before saving.')
            else:
                raise TypeError('Unsupported data type for HuggingFaceTable. Please use a Dataset')

        super().save(name=name,
                     snapshot=snapshot,
                     overwrite=overwrite,
                     **snapshot_kwargs)

        # Restore the original dataset
        if hf_dataset is not None:
            self.data = hf_dataset

        return self

    def fetch(self, **kwargs):
        from datasets import Dataset
        # Read as pyarrow table, then convert back to HF dataset
        arrow_table = super().fetch(**kwargs)
        self._cached_data = self.to_dataset(arrow_table)
        return self._cached_data

    def stream(self, batch_size, drop_last: Optional[bool] = False, shuffle_seed: Optional[int] = None):
        for batch in super().stream(batch_size, drop_last, shuffle_seed):
            yield self.to_dataset(batch)

    @staticmethod
    def to_dataset(data):
        """Convert to a huggingface dataset"""
        from datasets import Dataset
        import pyarrow as pa

        if isinstance(data, dict):
            return Dataset.from_dict(data)

        if isinstance(data, pa.Table):
            data = data.to_pandas()

        if not isinstance(data, pd.DataFrame):
            raise TypeError(f'Data must be a dict, Pandas DataFrame, or PyArrow table, not {type(data)}')

        return Dataset.from_pandas(data)
