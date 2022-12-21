import shutil
import uuid
from pathlib import Path
from typing import Optional, List, Union
import time
import os
import copy

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
    DEFAULT_FOLDER_PATH = '/runhouse/tables'

    def __init__(self,
                 url: str,
                 name: Optional[str] = None,
                 fs: Optional[str] = None,
                 # TODO hold a Folder object
                 data_config: Optional[dict] = None,
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
        self.data_config = data_config or {}

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
        with fsspec.open(self.fsspec_url, mode='rb', **self.data_config) as t:
            self._cached_data: pa.Table = pq.read_table(t, columns=columns)

        return self._cached_data

    def __getitem__(self, key: Optional[list] = None):
        return self.data.__getitem__(key)

    @property
    def fsspec_url(self):
        return f'{self.fs}://{self.url}'

    @property
    def root_path(self):
        return self.fsspec_url if self.fs != rns_client.DEFAULT_FS else self.url

    def stream(self, batch_size, drop_last: bool = False, shuffle_seed: Optional[int] = None):
        df = ray.data.read_parquet(self.root_path)
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

        # TODO [JL] NOTE: this should work but fsspec is pretty unreliable
        # https://stackoverflow.com/questions/53416226/how-to-write-parquet-file-from-pandas-dataframe-in-s3-in-python
        # new_data.to_parquet(self.fsspec_url)

        # Use pyarrow API to write partitioned data - adds an additional step to build the pyarrow table
        # based on the provided data's format
        pa_table: pa.Table = self.construct_table(new_data)
        pq.write_to_dataset(pa_table,
                            root_path=self.root_path,
                            partition_cols=partition_cols)

        save(self,
             save_to=save_to if save_to is not None else self.save_to,
             snapshot=snapshot,
             overwrite=overwrite)

    def delete_in_fs(self, recursive: bool = True):
        """Remove contents of all subdirectories (ex: partitioned data folders)"""
        # If file(s) are directories, recursively delete contents and then also remove the directory

        # TODO [JL] this should actually delete the folders themselves (per fsspec), but only deletes their contents
        # https://filesystem-spec.readthedocs.io/en/latest/api.html#fsspec.spec.AbstractFileSystem.rm
        self.fsspec_fs.rm(self.root_path, recursive=recursive)

    def exists_in_fs(self):
        # TODO [JL] a little hacky - this checks the contents of the folder to make sure the table file(s) were deleted
        return self.fsspec_fs.exists(self.root_path) and len(self.fsspec_fs.ls(self.root_path)) > 1

    def from_cluster(self, cluster):
        """ Create a remote folder from a url on a cluster. This will create a virtual link into the
        cluster's filesystem. If you want to create a local copy or mount of the folder, use
        `Folder(url=<local_url>).sync_from_cluster(<cluster>, <url>)` or
        `Folder('url').from_cluster(<cluster>).mount(<local_url>)`. """
        if not cluster.address:
            raise ValueError('Cluster must be started before copying data from it.')
        creds = cluster.ssh_creds()
        data_config = {'host': cluster.address,
                       'username': creds['ssh_user'],
                       'key_filename': str(Path(creds['ssh_private_key']).expanduser())}
        new_table = copy.deepcopy(self)
        new_table.fs = 'sftp'
        new_table.data_config = data_config
        return new_table

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
          dryrun: bool = False,
          ):
    """ Returns a Table object, which can be used to interact with the table at the given url.
    If the table does not exist, it will be saved if `dryrun` is False.
    """
    config = rns_client.load_config(name, load_from=load_from)

    # fs = fs or config.get('fs') or PROVIDER_FS_LOOKUP[configs.get('default_provider')]
    fs = fs or config.get('fs') or rns_client.DEFAULT_FS
    config['fs'] = fs

    name = name or config.get('rns_address') or config.get('name')
    name = name.lstrip('/') if name is not None else name

    data_url = data_url or config.get('url')

    if data_url is None:
        # TODO [JL] move some of the default params in this factory method to the defaults module for configurability
        if name is None:
            name = uuid.uuid4().hex
        if fs == rns_client.DEFAULT_FS:
            # create random url to store in .cache folder of local filesystem
            data_url = str(Path(f"~/.cache/tables/{name}").expanduser())
        else:
            # save to the default bucket
            data_url = f'{Table.DEFAULT_FOLDER_PATH}/{name}'

    config['name'] = name
    config['url'] = data_url

    existing_folder = folder or config.get('folder')
    if existing_folder is None:
        folder_url = str(Path(data_url).parent)
        existing_folder = rh.folder(url=folder_url, save_to=[], fs=fs)

    if isinstance(folder, str):
        existing_folder = rh.folder(url=folder, save_to=[], fs=fs)

    config['folder'] = existing_folder.name if existing_folder else Table.DEFAULT_FOLDER_PATH

    new_data = data if Resource.is_picklable(data) else config.get('data')
    config['data'] = new_data
    config['data_config'] = data_config or config.get('data_config')
    config['partition_cols'] = partition_cols or config.get('partition_cols')
    config['save_to'] = save_to

    new_table = Table.from_config(config, dryrun=dryrun)

    if mkdir and fs != rns_client.DEFAULT_FS:
        existing_folder.mkdir()

    if new_table.name and not dryrun:
        new_table.save(new_data, overwrite=True, partition_cols=partition_cols)

    return new_table
