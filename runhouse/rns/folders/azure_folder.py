from typing import Optional

from .folder import Folder


class AzureFolder(Folder):
    RESOURCE_TYPE = "folder"
    DEFAULT_FS = "azure"

    def __init__(self, dryrun: bool, **kwargs):
        super().__init__(dryrun=dryrun, **kwargs)

    @staticmethod
    def from_config(config: dict, dryrun=True):
        """Load config values into the object."""
        raise NotImplementedError

    def empty_folder(self):
        """Remove Azure folder contents, but not the folder itself."""
        raise NotImplementedError

    def delete_in_fs(self, recurse=True, *kwargs):
        """Delete the Azure folder itself."""
        raise NotImplementedError

    def upload(self, src: str, region: Optional[str] = None):
        """Upload a folder to an Azure bucket."""
        raise NotImplementedError

    def upload_command(self, src: str, dest: str):
        raise NotImplementedError

    def download(self, dest):
        """Download a folder from an Azure bucket to local dir."""
        raise NotImplementedError

    def download_command(self, src, dest):
        raise NotImplementedError

    def to_cluster(self, dest_cluster, url=None, mount=False, return_dest_folder=False):
        raise NotImplementedError

    def to_local(
        self, dest_url: str, data_config: dict, return_dest_folder: bool = False
    ):
        """Copy a folder from an Azure bucket to local dir."""
        raise NotImplementedError

    def to_data_store(
        self,
        fs: str,
        data_store_url: Optional[str] = None,
        data_config: Optional[dict] = None,
        return_dest_folder: bool = True,
    ):
        """Copy folder from Azure to another remote data store (ex: GCS, S3, Azure)"""
        raise NotImplementedError
