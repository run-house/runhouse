from .folder import Folder


class GCSFolder(Folder):
    RESOURCE_TYPE = 'folder'
    DEFAULT_FS = 'gcp'

    def __init__(self, dryrun: bool, **kwargs):
        # TODO [JL]
        super().__init__(dryrun=dryrun, **kwargs)

    @staticmethod
    def from_config(config: dict, dryrun=True):
        """ Load config values into the object. """
        return GCSFolder(**config, dryrun=dryrun)

    def upload(self, src_url):
        pass

    def download(self, dest_url):
        pass

    def upload_command(self, otherstuff):
        pass

    def download_command(self, otherstuff):
        pass

    def to_cluster(self, cluster, url=None, mount=False, return_dest_folder=False):
        pass

    def to_local(self, url, data_config):
        pass

    def to_blob_storage(self, fs, url=None, data_config=None):
        pass


