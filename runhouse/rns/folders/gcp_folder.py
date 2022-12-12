from .folder import Folder


class GCPFolder(Folder):
    RESOURCE_TYPE = 'folder'
    DEFAULT_FS = 'gcp'

    def __init__(self, dryrun: bool, **kwargs):
        # TODO [JL]
        super().__init__(dryrun=dryrun, **kwargs)

    @staticmethod
    def from_config(config: dict, dryrun=True):
        """ Load config values into the object. """
        return GCPFolder(**config, dryrun=dryrun)

    def rsync(self):
        pass


