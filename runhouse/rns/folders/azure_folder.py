from .folder import Folder


class AzureFolder(Folder):
    RESOURCE_TYPE = 'folder'
    DEFAULT_FS = 'azure'

    def __init__(self, dryrun: bool, **kwargs):
        # TODO [JL]
        super().__init__(dryrun=dryrun, **kwargs)

    @staticmethod
    def from_config(config: dict, dryrun=True):
        """ Load config values into the object. """
        return AzureFolder(**config, dryrun=dryrun)

    def upload(self, src_url):
        raise NotImplementedError

    def download(self, dest_url):
        raise NotImplementedError

    def upload_command(self, otherstuff):
        raise NotImplementedError

    def download_command(self, otherstuff):
        raise NotImplementedError

    def to_cluster(self, cluster, url=None, mount=False, return_dest_folder=False):
        raise NotImplementedError

    def to_local(self, url, data_config):
        raise NotImplementedError

    def to_blob_storage(self, fs, url=None, data_config=None):
        raise NotImplementedError
