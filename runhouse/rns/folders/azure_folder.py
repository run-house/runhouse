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
        return AzureFolder(**config, dryrun=dryrun)

    def delete_in_system(self):
        """Delete Azure folder along with its contents."""
        raise NotImplementedError

    def delete_bucket(self):
        """Delete the Azure bucket."""
        raise NotImplementedError

    def upload(self, src: str, region: Optional[str] = None):
        """Upload a folder to an Azure bucket."""
        raise NotImplementedError

    def upload_command(self, src: str, dest: str):
        raise NotImplementedError

    def download(self, dest):
        """Download a folder from a Azure bucket to local dir."""
        raise NotImplementedError

    def download_command(self, src, dest):
        raise NotImplementedError

    def to_cluster(self, dest_cluster, path=None, mount=False):
        raise NotImplementedError

    def to_local(self, dest_path: str, data_config: dict):
        """Copy a folder from an Azure bucket to local dir."""
        raise NotImplementedError

    def to_data_store(
        self,
        system: str,
        data_store_path: Optional[str] = None,
        data_config: Optional[dict] = None,
    ):
        """Copy folder from Azure to another remote data store (ex: GCS, S3, Azure)"""
        raise NotImplementedError
