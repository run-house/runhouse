import uuid
from pathlib import Path
from typing import Optional, List, Dict
import copy
import logging
import fsspec
import pyarrow.parquet as pq
import pyarrow as pa
import ray.data
import numpy as np

from .. import Resource, Cluster
from runhouse.rns.folders.folder import folder
import runhouse as rh
from runhouse.rh_config import rns_client

logger = logging.getLogger(__name__)


class Table(Resource):
    RESOURCE_TYPE = 'table'
    DEFAULT_FOLDER_PATH = '/runhouse/tables'
    DEFAULT_CACHE_FOLDER = '.cache/runhouse/tables/'
    DEFAULT_STREAM_FORMAT = 'pyarrow'
    DEFAULT_BATCH_SIZE = 256

    def __init__(self,
                 url: str,
                 name: Optional[str] = None,
                 file_name: Optional[str] = None,
                 fs: Optional[str] = None,
                 data_config: Optional[dict] = None,
                 dryrun: bool = True,
                 partition_cols: Optional[List] = None,
                 stream_format: Optional[str] = None,
                 metadata: Optional[Dict] = None,
                 **kwargs
                 ):
        super().__init__(name=name, dryrun=dryrun)
        self._filename = str(Path(url).name) if url else self.name
        # Use factory method so correct subclass for fs is returned
        self._folder = folder(url=url,
                              fs=fs,
                              data_config=data_config,
                              dryrun=dryrun)
        self._cached_data = None
        self.partition_cols = partition_cols
        self.file_name = file_name
        self.stream_format = stream_format or self.DEFAULT_STREAM_FORMAT
        self.metadata = metadata or {}

    @staticmethod
    def from_config(config: dict, dryrun=True):
        if isinstance(config['fs'], dict):
            config['fs'] = Cluster.from_config(config['fs'], dryrun=dryrun)
        return Table(**config, dryrun=dryrun)

    @property
    def config_for_rns(self):
        config = super().config_for_rns
        if isinstance(self._folder, Resource):
            config['fs'] = self._resource_string_for_subconfig(self.fs)
            config['data_config'] = self._folder._data_config
        else:
            config['fs'] = self.fs
        self.save_attrs_to_config(config, ['url', 'partition_cols', 'metadata'])
        config.update(config)

        return config

    @property
    def data(self) -> ray.data.dataset.Dataset:
        """Get the table data. If data is not already cached return a ray dataset. With the dataset object we can
        stream or convert to other types, for example:
            data.iter_batches()
            data.to_pandas()
            data.to_dask()
        """
        if self._cached_data is not None:
            return self._cached_data
        return self.ray_dataset

    @data.setter
    def data(self, new_data):
        """Update the data blob to new data"""
        self._cached_data = new_data
        # TODO should we save here?
        # self.save(overwrite=True)

    @property
    def fs(self):
        return self._folder.fs

    @fs.setter
    def fs(self, new_fs):
        self._folder.fs = new_fs

    @property
    def url(self):
        if self.file_name:
            return f'{self._folder.url}/{self.file_name}'
        return self._folder.url

    @url.setter
    def url(self, new_url):
        self._folder.url = new_url

    def set_metadata(self, key, val):
        self.metadata[key] = val

    @property
    def fsspec_url(self):
        if self.file_name:
            return f'{self._folder.fsspec_url}/{self.file_name}'
        return self._folder.fsspec_url

    @property
    def data_config(self):
        return self._folder.data_config

    @data_config.setter
    def data_config(self, new_data_config):
        self._folder.data_config = new_data_config

    def to(self, fs, url=None, data_config=None):
        new_table = copy.copy(self)
        new_table._folder = self._folder.to(fs=fs, url=url, data_config=data_config)
        return new_table

    def save(self,
             name: Optional[str] = None,
             snapshot: bool = False,
             overwrite: bool = True,
             **snapshot_kwargs):
        if self._cached_data is not None:
            pq.write_to_dataset(self.data,
                                root_path=self.fsspec_url,
                                partition_cols=self.partition_cols,
                                existing_data_behavior='overwrite_or_ignore' if overwrite else 'error')

            if hasattr(self.data, '__len__'):
                # Store the number of rows if we use for training later without having to read in the whole table
                self.set_metadata('num_rows', len(self.data))

        return super().save(name=name, snapshot=snapshot, overwrite=overwrite, **snapshot_kwargs)

    def fetch(self, columns: Optional[list] = None) -> pa.Table:
        # https://arrow.apache.org/docs/python/generated/pyarrow.parquet.read_table.html
        try:
            with fsspec.open(self.fsspec_url, mode='rb', **self.data_config) as t:
                self._cached_data = pq.read_table(t.full_name, columns=columns)
        except:
            # When trying to read as file like object could fail for a couple of reasons:
            # IsADirectoryError: The folder URL is actually a directory and the file has been automatically
            # generated for us inside the folder (ex: pyarrow table)

            # The file system is SFTP: since the SFTPFileSystem takes the host as a separate param, we cannot
            # pass in the data config as a single data_config kwarg

            # When specifying the filesystem don't pass in the fsspec url (which includes the file system prepended)
            self._cached_data = pq.read_table(self.url,
                                              columns=columns,
                                              filesystem=self._folder.fsspec_fs)
        return self._cached_data

    def __getitem__(self, key: Optional[list] = None):
        return self.data.__getitem__(key)

    def __getstate__(self):
        """Override the pickle method to clear _cached_data before pickling"""
        state = self.__dict__.copy()
        state['_cached_data'] = None
        return state

    def __iter__(self):
        return self

    def __next__(self):
        return self.stream(batch_size=1)

    def __len__(self):
        len_dataset = self.metadata.get('num_rows')
        if len_dataset is not None:
            return len_dataset

        if isinstance(self.data, ray.data.dataset.Dataset):
            len_dataset = self.data.count()
            self.set_metadata('num_rows', len_dataset)
            return len_dataset

        if not hasattr(self.data, '__len__') or not self.data:
            raise RuntimeError("Cannot get len for dataset.")

        len_dataset = len(self.data)
        self.set_metadata('num_rows', len_dataset)
        return len_dataset

    def stream(self, batch_size, drop_last: Optional[bool] = False, shuffle_seed: Optional[int] = None):
        # https://github.com/ray-project/ray/issues/30915
        print("inside table stream")
        df = self.data
        print("df in table stream", df)
        print("stream format", self.stream_format)
        if self.stream_format == 'torch':
            # https://docs.ray.io/en/master/data/api/doc/ray.data.Dataset.iter_torch_batches.html#ray.data.Dataset.iter_torch_batches
            return df.iter_torch_batches(batch_size=batch_size,
                                         drop_last=drop_last,
                                         local_shuffle_seed=shuffle_seed)

        elif self.stream_format == 'tf':
            # https://docs.ray.io/en/master/data/api/doc/ray.data.Dataset.iter_tf_batches.html
            return df.iter_tf_batches(batch_size=batch_size,
                                      drop_last=drop_last,
                                      local_shuffle_seed=shuffle_seed)
        else:
            # https://docs.ray.io/en/latest/data/api/dataset.html#ray.data.Dataset.iter_batches
            return df.iter_batches(batch_size=batch_size,
                                   batch_format=self.stream_format,
                                   drop_last=drop_last,
                                   local_shuffle_seed=shuffle_seed)

    @property
    def ray_dataset(self):
        return ray.data.read_parquet(self.url, filesystem=self._folder.fsspec_fs)

    def delete_in_fs(self, recursive: bool = True):
        """Remove contents of all subdirectories (ex: partitioned data folders)"""
        # If file(s) are directories, recursively delete contents and then also remove the directory

        # TODO [JL] this should actually delete the folders themselves (per fsspec), but only deletes their contents
        # https://filesystem-spec.readthedocs.io/en/latest/api.html#fsspec.spec.AbstractFileSystem.rm
        self._folder.rm('', recursive=recursive)  # Passing in an empty string to delete the contents of the folder

    def exists_in_fs(self):
        return self._folder.exists_in_fs() and len(self._folder.ls(self.fsspec_url)) > 1

    def from_cluster(self, cluster):
        """ Create a remote folder from a url on a cluster. This will create a virtual link into the
        cluster's filesystem. If you want to create a local copy or mount of the folder, use
        `Folder(url=<local_url>).sync_from_cluster(<cluster>, <url>)` or
        `Folder('url').from_cluster(<cluster>).mount(<local_url>)`. """
        if not cluster.address:
            raise ValueError('Cluster must be started before copying data from it.')
        new_table = copy.deepcopy(self)
        new_table._folder.fs = cluster
        return new_table


def _load_table_subclass(data, config: dict, dryrun: bool):
    """Load the relevant Table subclass based on the config or data type provided"""
    resource_subtype = config.get('resource_subtype', Table.__name__)

    try:
        import datasets
        if resource_subtype == 'HuggingFaceTable' or isinstance(data, datasets.Dataset):
            from .huggingface_table import HuggingFaceTable
            return HuggingFaceTable.from_config(config)
    except ModuleNotFoundError:
        pass
    except Exception as e:
        raise e

    try:
        import pandas as pd
        if isinstance(data, pd.DataFrame) or resource_subtype == 'PandasTable':
            from .pandas_table import PandasTable
            return PandasTable.from_config(config)
    except ModuleNotFoundError:
        pass
    except Exception as e:
        raise e

    try:
        import dask.dataframe as dd
        if isinstance(data, dd.DataFrame) or resource_subtype == 'DaskTable':
            from .dask_table import DaskTable
            return DaskTable.from_config(config)
    except ModuleNotFoundError:
        pass
    except Exception as e:
        raise e

    try:
        import ray
        if isinstance(data, ray.data.dataset.Dataset) or resource_subtype == 'RayTable':
            from .ray_table import RayTable
            return RayTable.from_config(config)
    except ModuleNotFoundError:
        pass
    except Exception as e:
        raise e

    try:
        import cudf
        if isinstance(data, cudf.DataFrame) or resource_subtype == 'CudfTable':
            from .rapids_table import RapidsTable
            return RapidsTable.from_config(config)
    except ModuleNotFoundError:
        pass
    except Exception as e:
        raise e

    if isinstance(data, pa.Table) or resource_subtype == 'Table':
        new_table = Table.from_config(config, dryrun=dryrun)
        return new_table
    else:
        raise TypeError(f'Unsupported data type {type(data)} for Table construction. '
                        f'For converting data to pyarrow see: '
                        f'https://arrow.apache.org/docs/7.0/python/generated/pyarrow.Table.html')


def table(data=None,
          name: Optional[str] = None,
          url: Optional[str] = None,
          fs: Optional[str] = None,
          data_config: Optional[dict] = None,
          partition_cols: Optional[list] = None,
          mkdir: Optional[bool] = False,
          dryrun: Optional[bool] = False,
          stream_format: Optional[str] = None,
          metadata: Optional[Dict] = None,
          ):
    """ Returns a Table object, which can be used to interact with the table at the given url.
    If the table does not exist, it will be saved if `dryrun` is False.
    """
    config = rns_client.load_config(name)

    config['fs'] = fs or config.get('fs') or rns_client.DEFAULT_FS
    if isinstance(config['fs'], str) and rns_client.exists(config['fs'], resource_type='cluster'):
        config['fs'] = rns_client.load_config(config['fs'])

    name = name or config.get('rns_address') or config.get('name')
    name = name.lstrip('/') if name is not None else name

    data_url = url or config.get('url')
    file_name = None
    if data_url:
        # Extract the file name from the url if provided
        full_path = Path(data_url)
        file_suffix = full_path.suffix
        if file_suffix:
            data_url = str(full_path.parent)
            file_name = full_path.name

    if data_url is None:
        # TODO [JL] move some of the default params in this factory method to the defaults module for configurability
        if config['fs'] == rns_client.DEFAULT_FS:
            # create random url to store in .cache folder of local filesystem
            data_url = str(Path(f"~/{Table.DEFAULT_CACHE_FOLDER}/{name or uuid.uuid4().hex}").expanduser())
        else:
            # save to the default bucket
            data_url = f'{Table.DEFAULT_FOLDER_PATH}/{name}'

    config['name'] = name
    config['url'] = data_url
    config['file_name'] = file_name or config.get('file_name')
    config['data_config'] = data_config or config.get('data_config')
    config['partition_cols'] = partition_cols or config.get('partition_cols')
    config['stream_format'] = stream_format or config.get('stream_format')
    config['metadata'] = metadata or config.get('metadata')

    if mkdir:
        # create the remote folder for the table
        rh.folder(url=data_url, fs=fs, dryrun=True).mkdir()

    new_table = _load_table_subclass(data, config, dryrun)
    if data is not None:
        new_table.data = data

    return new_table
