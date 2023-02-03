import logging
import uuid
from typing import Optional

from .table import Table
from .. import Cluster, Resource
from ..top_level_rns_fns import save

logger = logging.getLogger(__name__)


class PandasTable(Table):
    DEFAULT_FOLDER_PATH = '/runhouse/pandas-tables'
    DEFAULT_STREAM_FORMAT = 'pandas'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # PyArrow will create this file and suffix for us, but with Pandas we need to do it ourselves.
        if self.file_name is None:
            self.file_name = f'{uuid.uuid4().hex}.parquet'

    def __iter__(self):
        for block in self.stream(batch_size=self.DEFAULT_BATCH_SIZE):
            for idx, row in block.iterrows():
                yield row

    @staticmethod
    def from_config(config: dict, dryrun=True):
        """ Load config values into the object. """
        if isinstance(config['fs'], dict):
            config['fs'] = Cluster.from_config(config['fs'], dryrun=dryrun)
        return PandasTable(**config)

    def save(self,
             name: Optional[str] = None,
             snapshot: bool = False,
             overwrite: bool = True,
             **snapshot_kwargs):
        if self._cached_data is not None:
            # https://pandas.pydata.org/pandas-docs/version/1.1/reference/api/pandas.DataFrame.to_parquet.html
            # TODO [JL] this should work with any filesystem, not just SSHFS or SFTP
            self.data.to_parquet(self.fsspec_url,
                                 filesystem=self._folder.fsspec_fs if isinstance(self.fs, Resource) else None,
                                 partition_cols=self.partition_cols)
            logger.info(f'Saved {str(self)} to: {self.fsspec_url}')

        save(self,
             snapshot=snapshot,
             overwrite=overwrite,
             **snapshot_kwargs)

        return self

    def fetch(self, **kwargs):
        import pandas as pd
        # https://pandas.pydata.org/docs/reference/api/pandas.read_parquet.html
        self._cached_data = pd.read_parquet(self.fsspec_url, storage_options=self.data_config)
        return self._cached_data
