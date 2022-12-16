import copy
import logging
import uuid
from pathlib import Path
from typing import Optional, List, Dict
import fsspec

import runhouse as rh
from runhouse.rns.top_level_rns_fns import save
from runhouse.rh_config import rns_client, configs
from runhouse.rns.folders.folder import Folder, PROVIDER_FS_LOOKUP
from runhouse.rns.resource import Resource

logger = logging.getLogger(__name__)


class Blob(Resource):
    # TODO rename to "File" and take out serialization?
    RESOURCE_TYPE = 'blob'
    DEFAULT_FOLDER_PATH = '/runhouse/blobs'

    def __init__(self,
                 url: Optional[str] = None,
                 name: Optional[str] = None,
                 fs: Optional[str] = Folder.DEFAULT_FS,
                 data_config: Optional[Dict] = None,
                 dryrun: bool = True,
                 save_to: Optional[List[str]] = None,
                 load_from: Optional[List[str]] = None,
                 **kwargs
                 ):
        """

        Args:
            name ():
            fs (): FSSpec protocol, e.g. 's3', 'gcs'. See/run `fsspec.available_protocols()`.
                Default is "file", the local filesystem to wherever the blob is created.
            data_config ():
            serializer ():
        """
        super().__init__(name=name, dryrun=dryrun, save_to=save_to, load_from=load_from)
        self._cached_data = None
        self.fs = fs
        self.fsspec_fs = fsspec.filesystem(self.fs)
        self.url = url
        self.data_config = data_config

    # TODO do we need a del?

    @property
    def config_for_rns(self):
        config = super().config_for_rns
        blob_config = {'url': self.url,  # pair with data source to create the physical URL
                       'type': self.RESOURCE_TYPE,
                       'fs': self.fs
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
    def fsspec_url(self):
        return f'{self.fs}://{self.url}'

    def open(self, mode='rb'):
        """Get a file-like (OpenFile container object) of the blob data"""
        return fsspec.open(self.fsspec_url, mode=mode)

    def fetch(self):
        """Return the data for the user to deserialize"""
        with self.open() as f:
            try:
                self._cached_data = f.read()
            except:
                raise ValueError(f'Could not read blob data from: {self.fsspec_url}')

        return self._cached_data

    def save(self,
             new_data,
             save_to: Optional[List[str]] = None,
             snapshot: bool = False,
             overwrite: bool = True):
        self._cached_data = new_data
        # TODO figure out default behavior for not overwriting but still saving
        # if not overwrite:
        #     TODO check if data_url is already in use
        #     time = datetime.today().strftime('%Y-%m-%d_%H:%M:%S')
        #     self.data_url = self.data_url + time or time

        fss_file = self.open(mode='wb')
        with fss_file as f:
            if not isinstance(new_data, bytes):
                # Avoid TypeError: a bytes-like object is required
                raise TypeError(f'Cannot save blob with data of type {type(new_data)}, data must be serialized')

            f.write(new_data)

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

    def sync_from_cluster(self, cluster, url: Optional[str] = None):
        """ Efficiently rsync down a blob from a cluster, into the url of the current Blob object. """
        if not cluster.address:
            raise ValueError('Cluster must be started before copying data to it.')
        # TODO support fsspec urls (e.g. nonlocal fs's)?

        cluster.rsync(source=self.url, dest=url, up=False)

    def from_cluster(self, cluster):
        """ Create a remote blob from a url on a cluster. This will create a virtual link into the
        cluster's filesystem. If you want to create a local copy or mount of the blob, use
        `Blob(url=<local_url>).sync_from_cluster(<cluster>, <url>)` or
        `Blob('url').from_cluster(<cluster>).mount(<local_url>)`. """
        if not cluster.address:
            raise ValueError('Cluster must be started before copying data from it.')
        creds = cluster.ssh_creds()
        data_config = {'host': cluster.address,
                       'ssh_creds': {'username': creds['ssh_user'],
                                     'pkey': creds['ssh_private_key']}
                       }
        new_blob = copy.deepcopy(self)
        new_blob.fs = 'sftp'
        new_blob.data_config = data_config
        return new_blob


def blob(data=None,
         name: Optional[str] = None,
         url: Optional[str] = None,
         fs: Optional[str] = None,
         data_config: Optional[Dict] = None,
         load_from: Optional[List[str]] = None,
         save_to: Optional[List[str]] = None,
         mkdir: bool = False,
         snapshot: bool = False,
         dryrun: bool = True):
    """ Returns a Blob object, which can be used to interact with the resource at the given url

    Examples:
    # Creating the blob data - note the data should be provided as a serialized object, runhouse does not provide the
    # serialization functionality
    data = json.dumps(list(range(50))

    # 1. Create a remote blob with a name and no URL
    # provide a folder path for which to save in the remote file system
    # Since no URL is explicitly provided, we will save to a bucket called runhouse/blobs/my-blob
    rh.blob(name="my-blob", data=data, data_source='s3', save_to=['rns'], dryrun=False)

    # 2. Create a remote blob with a name and URL
    rh.blob(name='my-blob', url='/runhouse-tests/my_blob.pickle', data=data, fs='s3', save_to=['rns'], dryrun=False)

    # 3. Create a local blob with a name and a URL
    # save the blob to the local filesystem
    rh.blob(name=name, data=data, url=str(Path.cwd() / "my_blob.pickle"), save_to=['local'], dryrun=False)

    # 4. Create a local blob with a name and no URL
    # Since no URL is explicitly provided, we will save to ~/.cache/blobs/my-blob
    rh.blob(name="my-blob", data=data, save_to=['local'], dryrun=False)

    # Loading a blob
    my_local_blob = rh.blob(name="my_blob", load_from=['local'])
    my_s3_blob = rh.blob(name="my_blob", load_from=['rns'])

    """
    config = rns_client.load_config(name, load_from=load_from)
    config['name'] = name or config.get('rns_address', None) or config.get('name')

    fs = fs or config.get('fs') or PROVIDER_FS_LOOKUP[configs.get('default_provider')]
    config['fs'] = fs

    data_url = url or config.get('url')
    if data_url is None:
        # TODO [JL] move some of the default params in this factory method to the defaults module for configurability
        if fs == rns_client.DEFAULT_FS:
            # create random url to store in .cache folder of local filesystem
            data_url = str(Path(f"~/.cache/blobs/{uuid.uuid4().hex}").expanduser())
        else:
            # save to the default bucket
            name = name.lstrip('/')
            data_url = f'{Blob.DEFAULT_FOLDER_PATH}/{name}'

    config['url'] = data_url

    new_data = data or config.get('data')

    config['save_to'] = save_to
    config['data_config'] = data_config or config.get('data_config')

    if mkdir and fs != rns_client.DEFAULT_FS:
        # create the remote folder for the blob
        folder_url = str(Path(data_url).parent)
        rh.folder(name=folder_url, fs=fs, save_to=save_to, dryrun=dryrun).mkdir()

    new_blob = Blob.from_config(config, dryrun=dryrun)

    if new_blob.name and not dryrun:
        new_blob.save(new_data, snapshot=snapshot, overwrite=True)

    return new_blob
