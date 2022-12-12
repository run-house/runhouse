import shutil
from pathlib import Path
from typing import Optional, List, Union
import time
import os

import pyarrow.parquet as pq
import pyarrow as pa
import ray.data

from .. import Resource
from runhouse.rns.folders.folder import Folder
import runhouse as rh
from runhouse.rh_config import rns_client


class Table(Resource):

    def __init__(self,
                 folder: Folder,
                 name: str = None,
                 data_source: str = None,
                 data_config: dict = None,
                 save_to: Optional[List[str]] = None,
                 dryrun: bool = True,
                 partition_cols: list = None,
                 **kwargs
                 ):
        super().__init__(name=name,
                         data_source=data_source,
                         data_config=data_config,
                         partition_cols=partition_cols,
                         save_to=save_to,
                         dryrun=dryrun,
                         folder=folder)

    @staticmethod
    def from_config(config: dict, dryrun=True):
        return Table(**config, dryrun=dryrun)

    def fetch(self, deserializer: Optional[str] = None, return_file_like: bool = False, columns: Optional[list] = None):
        # https://arrow.apache.org/docs/python/generated/pyarrow.parquet.read_table.html
        # TODO support columns to fetch a subset of the data
        pq_table: pa.Table = pq.read_table(self.root_path, columns=columns)
        return pq_table

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
             overwrite: bool = False,
             partition_cols: Optional[list] = None):

        if not hasattr(new_data, 'to_parquet'):
            raise TypeError("Data saved to a runhouse Table must have a to_parquet method, "
                            "ideally backed by PyArrow's `to_parquet`.")

        try:
            # We should be able to write the data directly to the cloud via fsspec
            # https://stackoverflow.com/questions/53416226/how-to-write-parquet-file-from-pandas-dataframe-in-s3-in-python
            # If we fail to write using fsspec, we'll do more manual work by building the table manually and using the
            # pyarrow API to upload to the cloud
            new_data.to_parquet(self.fsspec_url)
        except:
            pa_table: pa.Table = self.construct_table(new_data)
            pq.write_to_dataset(pa_table,
                                root_path=self.root_path,
                                partition_cols=partition_cols)

        rns_client.save_config(resource=self,
                               overwrite=overwrite)

    def history(self, entries=10):
        # TODO return the history of this URI, including each new url and which runs have overwritten it.
        pass

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
                # Saving down to local disk, then uploading to s3 via pyarrow API
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
          data_source: Optional[str] = None,
          data_config: Optional[dict] = None,
          partition_cols: Optional[list] = None,
          save_to: Optional[List[str]] = None,
          load_from: Optional[List[str]] = None,
          dryrun: bool = False
          ):
    """ Returns a Table object, which can be used to interact with the table at the given url.
    If the table does not exist, it will be created if `create` is True.
    """
    config = rns_client.load_config(name, load_from=load_from)

    if isinstance(folder, str):
        folder = rh.folder(name=folder, load_from=load_from)

    existing_folder = folder or config.get('folder')
    if not existing_folder:
        raise ValueError('A table must container a folder - no folder was provided or exists in the config')

    # TODO look at the data url to determine the folder to instantiate

    config['folder'] = existing_folder

    new_data = data if Resource.is_picklable(data) else config.get('data')
    config['data'] = new_data
    config['name'] = name or config.get('rns_address', None) or config.get('name')
    config['data_url'] = data_url or config.get('data_url')
    config['data_source'] = data_source or config.get('data_source')
    config['data_config'] = data_config or config.get('data_config')
    config['partition_cols'] = partition_cols or config.get('partition_cols')
    config['save_to'] = save_to

    new_table = Table.from_config(config, dryrun=dryrun)

    if new_table.name and Resource.is_picklable(new_data) and not dryrun:
        new_table.save(new_data, overwrite=True, partition_cols=partition_cols)

    return new_table
