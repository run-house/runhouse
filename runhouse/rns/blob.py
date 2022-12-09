import os
import logging
from typing import Optional
import fsspec

from ray import cloudpickle
from runhouse.rns.resource import Resource
from ray import cloudpickle as pickle

import runhouse.rns.top_level_rns_fns

logger = logging.getLogger(__name__)


class Blob(Resource):
    # TODO rename to "File" and take out serialization?
    RESOURCE_TYPE = 'blob'
    DEFAULT_FS = 'file'

    def __init__(self,
                 data=None,  # Filepath or pickleable python object
                 name: str = None,
                 data_url: str = None,
                 data_source: str = None,
                 data_config: dict = None,
                 partition_cols: list = None,
                 serializer: str = None,
                 dryrun: bool = True
                 ):
        """

        Args:
            name ():
            data_url ():
            data_source (): FSSpec protocol, e.g. 's3', 'gcs'. See/run `fsspec.available_protocols()`.
                Default is "file", the local filesystem to whereever the blob is created.
            data_config ():
            serializer ():
        """
        super().__init__(name=name, dryrun=dryrun)
        self._cached_data = None

        # TODO set default data_url to be '(project_name or filename)_varname'
        self.data_url = data_url
        self.data_source = data_source
        self.partition_cols = partition_cols

        self.data_config = data_config or {}
        self.serializer = serializer

        if self.is_picklable(data) and create and name is not None:
            self.save(new_data=data, overwrite=True, partition_cols=self.partition_cols)

    # TODO do we need a del?

    @property
    def config_for_rns(self):
        config = super().config_for_rns
        blob_config = {'data_url': self.data_url,
                       'data_source': self.data_source,
                       'data_config': self.data_config,
                       'serializer': self.serializer,
                       'partition_cols': self.partition_cols
                       }
        config.update(blob_config)
        return config

    @property
    def data(self):
        """Get the blob data"""
        # TODO this caching is dumb, either get rid of it or replace with caching from fsspec
        if self._cached_data is not None:
            return self._cached_data
        return self.fetch()

    @data.setter
    def data(self, new_data):
        """Update the data blob to new data"""
        self.save(new_data, overwrite=True)

    def open(self, mode='rb'):
        """Get a file-like object of the blob data"""
        return fsspec.open(self.fsspec_url, mode=mode, **self.data_config)

    @property
    def fsspec_url(self):
        """Generate the FSSpec URL using the data_source and data_url"""
        # if self.data_source == 'file'
        #     return self.data_url
        return f'{self.data_source}://{self.data_url}'

    @property
    def root_path(self) -> str:
        """Root path of the blob, e.g. the s3 bucket path to the data.
        If the data is partitioned, we store the data in a separate partitions directory"""
        url = self.fsspec_url
        return url if not self.partition_cols else f'{url}/partitions'

    @staticmethod
    def is_picklable(obj) -> bool:
        try:
            pickle.dumps(obj)
        except pickle.PicklingError:
            return False
        return True

    def fetch(self, return_file_like=False):
        fss_file = fsspec.open(self.fsspec_url, mode='rb', **self.data_config)
        if return_file_like:
            return fss_file
        with fss_file as f:
            if self.serializer is not None:
                if self.serializer == 'pickle':
                    self._cached_data = cloudpickle.load(f)
                else:
                    raise f'Cannot load blob with unrecognized serializer {self.serializer}'
            else:
                self._cached_data = f.read()
        return self._cached_data

    def save(self, new_data, serializer: Optional[str] = None, overwrite: bool = False):
        self._cached_data = new_data
        # TODO figure out default behavior for not overwriting but still saving
        # if not overwrite:
        #     TODO check if data_url is already in use
        #     time = datetime.today().strftime('%Y-%m-%d_%H:%M:%S')
        #     self.data_url = self.data_url + time or time
        fss_file = fsspec.open(self.fsspec_url, mode='wb', **self.data_config)
        with fss_file as f:
            if self.serializer is not None:
                if self.serializer == 'pickle':
                    cloudpickle.dump(new_data, f)
                else:
                    raise f'Cannot store blob with unrecognized serializer {self.serializer}'
            else:
                f.write(new_data)

    def delete_in_fs(self, recursive: bool = True):
        # TODO [JL] sfs3 isn't working well here, so using boto directly
        if self.data_source == 's3':
            try:
                import boto3
                s3 = boto3.resource('s3')
                bucket_name = self.data_url.split('/')[0]
                key = '/'.join(self.data_url.split('/')[1:])
                bucket = s3.Bucket(bucket_name)
                bucket.objects.filter(Prefix=f'{key}/').delete()
            except:
                pass
        else:
            fs = fsspec.filesystem(self.data_source)
            fs.rm(self.data_url, recursive=recursive)

    def exists_in_fs(self):
        # TODO check both here? (i.e. what is defined in config + fsspec filesystem)?
        fs = fsspec.filesystem(self.data_source)
        return fs.exists(self.data_url) or runhouse.rns.top_level_rns_fns.exists(self.data_url)


def blob(data=None,
         data_url=None,
         data_source='file',
         data_config=None,
         serializer=None,
         partition_cols=None,
         name=None,
         dryrun=False):
    """Create a Blob object"""
    # TODO [DG]
    return Blob(data=data, data_url=data_url, data_source=data_source, data_config=data_config,
                serializer=serializer, partition_cols=partition_cols, name=name,
                dryrun=dryrun)