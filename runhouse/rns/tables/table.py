import uuid
from pathlib import Path
from typing import Optional, List
import copy
import logging
import fsspec
import pyarrow.parquet as pq
import pyarrow as pa
import ray.data

from .. import Resource
from runhouse.rns.folders.folder import folder
import runhouse as rh
from runhouse.rh_config import rns_client
from ..top_level_rns_fns import save

logger = logging.getLogger(__name__)


class Table(Resource):
    RESOURCE_TYPE = 'table'
    DEFAULT_FOLDER_PATH = '/runhouse/tables'

    def __init__(self,
                 url: str,
                 name: Optional[str] = None,
                 fs: Optional[str] = None,
                 data_config: Optional[dict] = None,
                 save_to: Optional[List[str]] = None,
                 dryrun: bool = True,
                 partition_cols: Optional[List] = None,
                 **kwargs
                 ):
        super().__init__(name=name, dryrun=dryrun, save_to=save_to)
        self._filename = str(Path(url).name) if url else self.name
        # Use factory method so correct subclass for fs is returned
        self._folder = folder(url=url,
                              fs=fs,
                              data_config=data_config,
                              dryrun=dryrun,
                              save_to=save_to)
        self._cached_data = None
        self.partition_cols = partition_cols

    @staticmethod
    def from_config(config: dict, dryrun=True):
        return Table(**config, dryrun=dryrun)

    @property
    def config_for_rns(self):
        config = super().config_for_rns
        table_config = {'url': self.url,
                        'resource_type': self.RESOURCE_TYPE,
                        'fs': self.fs,
                        'resource_subtype': self.__class__.__name__}
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
        # TODO [DG] Do we want adding a filename to be the default behavior?
        #   ex: partitioning, dask, ray, huggingface will store multiple files in the same folder
        return self._folder.url + '/' + self._filename

    @url.setter
    def url(self, new_url):
        self._folder.url = str(Path(new_url).parent)

    @property
    def fsspec_url(self):
        return self._folder.fsspec_url

    @property
    def data_config(self):
        return self._folder.data_config

    @data_config.setter
    def data_config(self, new_data_config):
        self._folder.data_config = new_data_config

    def save(self, name: Optional[str] = None, snapshot: bool = False, save_to: Optional[List[str]] = None,
             overwrite: bool = False, **snapshot_kwargs):
        if self._cached_data is None or overwrite:
            pq.write_to_dataset(self.data,
                                root_path=self.fs,
                                partition_cols=self.partition_cols)

        save(self,
             save_to=save_to if save_to is not None else self.save_to,
             snapshot=snapshot,
             overwrite=overwrite,
             **snapshot_kwargs)

    def fetch(self, columns: Optional[list] = None):
        # https://arrow.apache.org/docs/python/generated/pyarrow.parquet.read_table.html
        with fsspec.open(self.fsspec_url, mode='rb', **self.data_config) as t:
            self._cached_data: pa.Table = pq.read_table(t, columns=columns)

        return self._cached_data

    def __getitem__(self, key: Optional[list] = None):
        return self.data.__getitem__(key)

    def stream(self, batch_size, drop_last: bool = False, shuffle_seed: Optional[int] = None):
        df = ray.data.read_parquet(self.fsspec_url)
        # TODO the latest ray version supports local shuffle inside iter_batches, use that instead?
        # https://docs.ray.io/en/latest/data/api/dataset.html#ray.data.Dataset.iter_batches
        if shuffle_seed is not None:
            df.random_shuffle(seed=shuffle_seed)
        return df.iter_batches(batch_size=batch_size,
                               batch_format="pyarrow",
                               drop_last=drop_last)

    def delete_in_fs(self, recursive: bool = True):
        """Remove contents of all subdirectories (ex: partitioned data folders)"""
        # If file(s) are directories, recursively delete contents and then also remove the directory

        # TODO [JL] this should actually delete the folders themselves (per fsspec), but only deletes their contents
        # https://filesystem-spec.readthedocs.io/en/latest/api.html#fsspec.spec.AbstractFileSystem.rm
        self._folder.rm('', recursive=recursive)  # Passing in an empty string to delete the contents of the folder

    def exists_in_fs(self):
        # TODO [JL] a little hacky - this checks the contents of the folder to make sure the table file(s) were deleted
        return self._folder.exists(self.fsspec_url) and len(self._folder.ls(self.fsspec_url)) > 1

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

    @staticmethod
    def import_package(package_name: str):
        try:
            from importlib import import_module
            import_module(package_name)
        except ModuleNotFoundError:
            raise ModuleNotFoundError(f'`{package_name}` not found in site-packages')


def _load_table_subclass(data, config: dict, dryrun: bool):
    """Load the relevant Table subclass based on the config or data type provided"""
    resource_subtype = config.get('resource_subtype', Table.__name__)

    try:
        import datasets
        if resource_subtype == 'HuggingFaceTable' or isinstance(data, datasets.dataset_dict.DatasetDict):
            from .huggingface_table import HuggingFaceTable
            return HuggingFaceTable.from_config(config)
    except ModuleNotFoundError:
        pass

    try:
        import pandas as pd
        if resource_subtype == 'PandasTable' or isinstance(data, pd.DataFrame):
            from .pandas_table import PandasTable
            return PandasTable.from_config(config)
    except ModuleNotFoundError:
        pass

    try:
        import dask.dataframe as dd
        if resource_subtype == 'DaskTable' or isinstance(data, dd.DataFrame):
            from .dask_table import DaskTable
            return DaskTable.from_config(config)
    except ModuleNotFoundError:
        pass

    try:
        import ray
        if resource_subtype == 'RayTable' or isinstance(data, ray.data.dataset.Dataset):
            from .ray_table import RayTable
            return RayTable.from_config(config)
    except ModuleNotFoundError:
        pass

    try:
        import cudf
        if resource_subtype == 'CudfTable' or isinstance(data, cudf.DataFrame):
            from .cudf_table import CudfTable
            return CudfTable.from_config(config)
    except ModuleNotFoundError:
        pass

    if resource_subtype == 'Table' or isinstance(data, pa.Table):
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
          save_to: Optional[List[str]] = None,
          load_from: Optional[List[str]] = None,
          mkdir: bool = False,
          dryrun: bool = False,
          ):
    """ Returns a Table object, which can be used to interact with the table at the given url.
    If the table does not exist, it will be saved if `dryrun` is False.
    """
    config = rns_client.load_config(name, load_from=load_from)

    fs = fs or config.get('fs') or rns_client.DEFAULT_FS
    config['fs'] = fs

    name = name or config.get('rns_address') or config.get('name')
    name = name.lstrip('/') if name is not None else name

    data_url = url or config.get('url')

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
    config['data_config'] = data_config or config.get('data_config')
    config['partition_cols'] = partition_cols or config.get('partition_cols')
    config['save_to'] = save_to

    if mkdir:
        # create the remote folder for the table
        # TODO [JL / DG] this creates a folder in the wrong location when running with local filesystems
        rh.folder(name=data_url, fs=fs, save_to=[], dryrun=True).mkdir()

    new_table = _load_table_subclass(data, config, dryrun)
    new_table.data = data

    if new_table.name and not dryrun:
        new_table.save(overwrite=True)

    return new_table
