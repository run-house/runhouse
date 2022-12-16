import shutil
from pathlib import Path
from typing import Optional, List, Union
import time
import os

import fsspec
import pyarrow.parquet as pq
import pyarrow as pa
import ray.data

from .. import Resource
from runhouse.rns.folders.folder import Folder, PROVIDER_FS_LOOKUP
import runhouse as rh
from runhouse.rh_config import rns_client, configs
from ..top_level_rns_fns import save


class Table(Resource):
    RESOURCE_TYPE = 'table'

    def __init__(self,
                 url: str,
                 name: Optional[str] = None,
                 fs: Optional[str] = None,
                 save_to: Optional[List[str]] = None,
                 dryrun: bool = True,
                 partition_cols: Optional[List] = None,
                 **kwargs
                 ):
        super().__init__(name=name,
                         save_to=save_to,
                         dryrun=dryrun)
        self._cached_data = None
        self.partition_cols = partition_cols
        self.fs = fs
        self.fsspec_fs = fsspec.filesystem(self.fs)
        self.url = url

    @staticmethod
    def from_config(config: dict, dryrun=True):
        return Table(**config, dryrun=dryrun)

    @property
    def config_for_rns(self):
        config = super().config_for_rns
        table_config = {'url': self.url,
                        'type': self.RESOURCE_TYPE,
                        'fs': self.fs}
        config.update(table_config)
        return config

    @property
    def data(self):
        """Get the blob data"""
        if self._cached_data is not None:
            return self._cached_data
        data = self.fetch()
        return data

    @data.setter
    def data(self, new_data):
        """Update the data blob to new data"""
        self.save(new_data, overwrite=True)

    def fetch(self, columns: Optional[list] = None):
        # https://arrow.apache.org/docs/python/generated/pyarrow.parquet.read_table.html
        # TODO support columns to fetch a subset of the data
        self._cached_data = pq.read_table(self.fsspec_url, columns=columns)

        return self._cached_data

    @property
    def fsspec_url(self):
        return f'{self.fs}://{self.url}'

    def stream(self, batch_size, drop_last: bool = False, shuffle_seed: Optional[int] = None):
        df = ray.data.read_parquet(self.fsspec_url)
        # TODO the latest ray version supports local shuffle inside iter_batches, use that instead?
        # https://docs.ray.io/en/latest/data/api/dataset.html#ray.data.Dataset.iter_batches
        if shuffle_seed is not None:
            df.random_shuffle(seed=shuffle_seed)
        return df.iter_batches(batch_size=batch_size,
                               batch_format="pyarrow",
                               drop_last=drop_last)

    def save(self,
             new_data,
             snapshot: bool = False,
             save_to: Optional[List[str]] = None,
             overwrite: bool = False,
             partition_cols: Optional[list] = None):

        if not hasattr(new_data, 'to_parquet'):
            raise TypeError("Data saved to a runhouse Table must have a to_parquet method, "
                            "ideally backed by PyArrow's `to_parquet`.")

        if not partition_cols:
            # https://stackoverflow.com/questions/53416226/how-to-write-parquet-file-from-pandas-dataframe-in-s3-in-python
            new_data.to_parquet(self.url) if self.fs == rns_client.DEFAULT_FS else \
                new_data.to_parquet(self.fsspec_url)
        else:
            # Use pyarrow API to write partitioned data - adds an additional step to build the pyarrow table
            # based on the provided data's format
            pa_table: pa.Table = self.construct_table(new_data)
            pq.write_to_dataset(pa_table,
                                root_path=self.fsspec_url,
                                partition_cols=partition_cols)

        save(self,
             save_to=save_to if save_to is not None else self.save_to,
             snapshot=snapshot,
             overwrite=overwrite)

    def delete_in_fs(self, recursive: bool = True):
        try:
            self.fsspec_fs.rm(self.fsspec_url, recursive=recursive)
        except FileNotFoundError:
            pass

    def exists_in_fs(self):
        return self.fsspec_fs.exists(self.fsspec_url)

    # if provider == 'snowflake':
    #     new_table = SnowflakeTable.from_config(config, create=create)
    # elif provider == 'bigquery':
    #     new_table = BigQueryTable.from_config(config, create=create)
    # elif provider == 'redshift':
    #     new_table = RedshiftTable.from_config(config, create=create)
    # elif provider == 'postgres':
    #     new_table = PostgresTable.from_config(config, create=create)
    # elif provider == 'deltalake':
    #     new_table = DeltaLakeTable.from_config(config, create=create)

    @staticmethod
    def construct_table(data) -> pa.Table:
        # https://arrow.apache.org/docs/7.0/python/generated/pyarrow.Table.html
        if isinstance(data, list):
            # Construct a Table from list of rows / dictionaries.
            # pylist = [{'int': 1, 'str': 'a'}, {'int': 2, 'str': 'b'}]
            return pa.Table.from_pylist(data)

        elif isinstance(data, dict):
            # Construct a Table from Arrow arrays or columns.
            # pydict = {'int': [1, 2], 'str': ['a', 'b']}
            return pa.Table.from_pydict(data)

        elif isinstance(data, (pa.Array, pa.ChunkedArray)):
            # Construct a Table from Arrow arrays.
            # Equal-length arrays that should form the table.
            return pa.Table.from_arrays(data)

        elif isinstance(data, pa.RecordBatch):
            # Construct a Table from a sequence or iterator of Arrow RecordBatches.
            return pa.Table.from_batches(data)

        else:
            try:
                # If data is none of the above types, see if we have a pandas dataframe
                return pa.Table.from_pandas(data)
            except:
                # Save down to local disk, then upload to data source via pyarrow API
                tmp_path = f'/tmp/temp_{int(time.time())}.parquet'
                data.to_parquet(tmp_path)

                data: pa.Table = pq.read_table(tmp_path)

                dirpath = Path(tmp_path)
                if dirpath.exists() and dirpath.is_dir():
                    shutil.rmtree(dirpath)

                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

                return data


def table(data=None,
          name: Optional[str] = None,
          folder: Union[Folder, str] = None,
          data_url: Optional[str] = None,
          fs: Optional[str] = None,
          data_config: Optional[dict] = None,
          partition_cols: Optional[list] = None,
          save_to: Optional[List[str]] = None,
          load_from: Optional[List[str]] = None,
          mkdir: bool = False,
          dryrun: bool = True
          ):
    """ Returns a Table object, which can be used to interact with the table at the given url.
    If the table does not exist, it will be saved if `dryrun` is False.
    """
    config = rns_client.load_config(name, load_from=load_from)

    fs = fs or config.get('fs') or PROVIDER_FS_LOOKUP[configs.get('default_provider')]
    config['fs'] = fs

    # TODO [JL] account for some defaults if url or folder are not provided (similar to blob)
    data_url = data_url or config.get('url')
    config['url'] = data_url

    # TODO [JL] maybe we don't accept Folder as param and just infer it from the data url?
    existing_folder = folder or config.get('folder')
    if existing_folder is None:
        existing_folder = rh.folder(url=str(Path(data_url).parent), save_to=save_to, dryrun=dryrun, fs=fs)

    if isinstance(folder, str):
        existing_folder = rh.folder(url=folder, save_to=save_to, dryrun=dryrun, fs=fs)

    config['folder'] = existing_folder

    new_data = data if Resource.is_picklable(data) else config.get('data')
    config['data'] = new_data
    config['name'] = name or config.get('rns_address', None) or config.get('name')
    config['data_config'] = data_config or config.get('data_config')
    config['partition_cols'] = partition_cols or config.get('partition_cols')
    config['save_to'] = save_to

    new_table = Table.from_config(config, dryrun=dryrun)

    if mkdir:
        existing_folder.mkdir()

    if new_table.name and Resource.is_picklable(new_data) and not dryrun:
        new_table.save(new_data, overwrite=True, partition_cols=partition_cols)

    return new_table
