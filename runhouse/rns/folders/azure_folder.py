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

    def rsync(self):
        pass