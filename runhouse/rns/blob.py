import logging
from typing import Optional, List, Dict, Union
import fsspec
import os

from ray import cloudpickle as pickle

import runhouse as rh
from runhouse.rh_config import rns_client
from runhouse.rns.folders.folder import Folder
from runhouse.rns.resource import Resource

logger = logging.getLogger(__name__)


class Blob(Resource):
    # TODO rename to "File" and take out serialization?
    RESOURCE_TYPE = 'blob'
    DEFAULT_FS = 'file'
    DEFAULT_SERIALIZER = 'pickle'

    def __init__(self,
                 folder: Folder,
                 data_url: str,
                 name: Optional[str] = None,
                 data_source: Optional[str] = None,
                 data_config: Optional[Dict] = None,
                 partition_cols: Optional[List] = None,
                 serializer: Optional[str] = None,
                 dryrun: bool = True,
                 save_to: Optional[List[str]] = None,
                 load_from: Optional[List[str]] = None,
                 **kwargs
                 ):
        """

        Args:
            name ():
            data_source (): FSSpec protocol, e.g. 's3', 'gcs'. See/run `fsspec.available_protocols()`.
                Default is "file", the local filesystem to whereever the blob is created.
            data_config ():
            serializer ():
        """
        super().__init__(name=name, dryrun=dryrun, save_to=save_to, load_from=load_from)
        self._cached_data = None

        # TODO set default data_url to be '(project_name or filename)_varname'
        self.data_source = data_source
        self.partition_cols = partition_cols

        self.data_config = data_config or {}
        self.serializer = serializer
        self.folder = folder
        self._data_url = data_url

    # TODO do we need a del?

    @property
    def config_for_rns(self):
        config = super().config_for_rns
        blob_config = {'data_url': self.data_url,
                       'data_source': self.data_source,
                       'data_config': self.data_config,
                       'serializer': self.serializer,
                       'partition_cols': self.partition_cols,
                       'folder': self.folder.rns_address
                       }
        config.update(blob_config)
        return config

    @staticmethod
    def from_config(config: dict, dryrun=True):
        return Blob(**config, dryrun=dryrun)

    @property
    def data(self):
        """Get the blob data"""
        # TODO this caching is dumb, either get rid of it or replace with caching from fsspec
        if self._cached_data is not None:
            return self._cached_data
        data = self.fetch()
        return data

    @data.setter
    def data(self, new_data):
        """Update the data blob to new data"""
        self.save(new_data, overwrite=True)

    @property
    def data_url(self):
        if self._data_url.startswith("/"):
            return self._data_url
        return f'/{self._data_url}'

    @property
    def fsspec_url(self):
        """Generate the FSSpec URL using the data_source and data_url of the blob"""
        return f'{self.data_source}:/{self.data_url}'

    @property
    def root_path(self) -> str:
        """Root path of the blob, e.g. the s3 bucket path to the data.
        If the data is partitioned, we store the data in a separate partitions directory"""
        url = self.fsspec_url
        return url if not self.partition_cols else f'{url}/partitions'

    @staticmethod
    def folder_url(url):
        return os.path.dirname(url)

    def open(self, mode='rb'):
        """Get a file-like object of the blob data"""
        return fsspec.open(self.fsspec_url, mode=mode, **self.data_config)

    def fetch(self, return_file_like=False):
        fss_file = fsspec.open(self.fsspec_url, mode='rb', **self.data_config)
        if return_file_like:
            return fss_file
        with fss_file as f:
            if self.serializer is not None:
                if self.serializer == 'pickle':
                    self._cached_data = pickle.load(f)
                else:
                    raise f'Cannot load blob with unrecognized serializer {self.serializer}'
            else:
                self._cached_data = f.read()
        return self._cached_data

    def save(self,
             new_data,
             overwrite: bool = False):
        self._cached_data = new_data
        # TODO figure out default behavior for not overwriting but still saving
        # if not overwrite:
        #     TODO check if data_url is already in use
        #     time = datetime.today().strftime('%Y-%m-%d_%H:%M:%S')
        #     self.data_url = self.data_url + time or time

        fss_file = fsspec.open(self.fsspec_url, mode='wb', **self.data_config)
        with fss_file as f:
            if self.serializer is not None:
                new_data = self.serialize_data(new_data)

            if not isinstance(new_data, bytes):
                # Avoid TypeError: a bytes-like object is required
                raise TypeError(f'Cannot save blob with data of type {type(new_data)}, add a serializer to the blob')

            f.write(new_data)

        rns_client.save_config(resource=self,
                               overwrite=overwrite)

    def serialize_data(self, new_data):
        if self.serializer == 'pickle':
            return pickle.dumps(new_data)
        else:
            raise f'Cannot store blob with unrecognized serializer {self.serializer}'

    def delete_in_fs(self, recursive: bool = True):
        fs = fsspec.filesystem(self.data_source)
        try:
            fs.rm(self.data_url, recursive=recursive)
        except FileNotFoundError:
            pass

    def exists_in_fs(self):
        fs = fsspec.filesystem(self.data_source)
        fs_exists = fs.exists(self.data_url)

        # TODO check both here? (i.e. what is defined in config + fsspec filesystem)?
        return fs_exists or rh.rns.top_level_rns_fns.exists(self.data_url)


def blob(data=None,
         name: Optional[str] = None,
         folder: Union[Folder, str] = None,
         data_url: Optional[str] = None,
         data_source: Optional[str] = None,
         data_config: Optional[dict] = None,
         serializer: Optional[str] = Blob.DEFAULT_SERIALIZER,
         load_from: Optional[List[str]] = None,
         save_to: Optional[List[str]] = None,
         dryrun: bool = True):
    """ Returns a Blob object, which can be used to interact with the resource at the given url """
    config = rns_client.load_config(name, load_from=load_from)

    data_source = data_source or config.get('data_source')
    if data_source is None:
        data_source = Folder.DEFAULT_FS

    if data_source not in fsspec.available_protocols():
        raise ValueError('Invalid data source')

    config['data_source'] = data_source

    data_url = data_url or config.get('data_url')
    config['data_url'] = data_url

    folder = folder or config.get('folder')
    folder_url = Blob.folder_url(data_url)
    if folder is None:
        if not config['data_url']:
            raise ValueError('data_url must exist if folder is not provided')
        # Pass the folder path as the URL (since the URL provided here is the full path to the blob)
        existing_folder = rh.folder(url=folder_url,
                                    load_from=load_from,
                                    fs=data_source,
                                    dryrun=dryrun)
    elif isinstance(folder, str):
        existing_folder = rh.folder(name=folder,
                                    url=folder_url,
                                    load_from=load_from,
                                    fs=data_source,
                                    dryrun=dryrun)
    elif isinstance(folder, Folder):
        existing_folder = folder
    else:
        raise TypeError('Folder must be a string or a Folder object')

    config['folder'] = existing_folder

    new_data = data if Resource.is_picklable(data) else config.get('data')
    config['name'] = name or config.get('rns_address', None) or config.get('name')
    config['serializer'] = serializer or config.get('serializer')

    config['data_config'] = data_config or config.get('data_config')
    config['save_to'] = save_to

    new_blob = Blob.from_config(config, dryrun=dryrun)

    if new_blob.name and new_blob.is_picklable(new_data) and not dryrun:
        new_blob.save(new_data, overwrite=True)

    return new_blob
