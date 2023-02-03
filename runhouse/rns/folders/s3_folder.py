import logging
import subprocess
from typing import Optional

from .folder import Folder

logger = logging.getLogger(__name__)


class S3Folder(Folder):
    RESOURCE_TYPE = "folder"
    DEFAULT_FS = "s3"

    def __init__(self, dryrun: bool, **kwargs):
        from s3fs import S3FileSystem

        # s3fs normally caches directory listings, because these lookups can be expensive.
        # Turn off this caching system to force refresh on every access
        S3FileSystem.clear_instance_cache()

        self.s3 = S3FileSystem(anon=False)
        self.s3.invalidate_cache()

        super().__init__(dryrun=dryrun, **kwargs)

    @staticmethod
    def from_config(config: dict, dryrun=True):
        """Load config values into the object."""
        return S3Folder(**config, dryrun=dryrun)

    def empty_folder(self):
        """Remove s3 folder contents, but not the folder itself."""
        for p in self.s3.ls(self.url):
            self.s3.rm(p)

    def delete_in_fs(self, recurse=True, *kwargs):
        """Delete the s3 folder itself along with its contents."""
        try:
            from sky.data.storage import S3Store

            S3Store(
                name=self.bucket_name_from_url(self.url), source=self._fsspec_fs
            ).delete()
        except Exception as e:
            raise e

    def upload(self, src: str, region: Optional[str] = None):
        """Upload a folder to an S3 bucket."""
        from sky.data.storage import S3Store

        # NOTE: The sky S3Store.upload() API does not let us specify directories within the bucket to upload to.
        # This means we have to use the CLI command for performing the actual upload using sky's `run_upload_cli`

        # Initialize the S3Store object which creates the bucket if it does not exist
        s3_store = S3Store(
            name=self.bucket_name_from_url(self.url), source=src, region=region
        )

        sync_dir_command = self.upload_command(src=src, dest=self.url)
        self.run_upload_cli_cmd(
            sync_dir_command, access_denied_message=s3_store.ACCESS_DENIED_MESSAGE
        )

    def upload_command(self, src: str, dest: str):
        # https://github.com/skypilot-org/skypilot/blob/983f5fa3197fe7c4b5a28be240f7b027f7192b15/sky/data/storage.py#L922
        dest = dest.lstrip("/")
        upload_command = (
            'aws s3 sync --no-follow-symlinks --exclude ".git/*" '
            f"{src} "
            f"s3://{dest}"
        )

        return upload_command

    def download(self, dest):
        """Download a folder from an S3 bucket to local dir."""
        # NOTE: Sky doesn't support this API yet for each provider
        # https://github.com/skypilot-org/skypilot/blob/983f5fa3197fe7c4b5a28be240f7b027f7192b15/sky/data/storage.py#L231
        remote_dir = self.url.lstrip("/")
        remote_dir = f"s3://{remote_dir}"
        subprocess.run(
            ["aws", "s3", "sync", remote_dir, dest],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )

    def download_command(self, src, dest):
        from sky.cloud_stores import S3CloudStorage

        download_command = S3CloudStorage().make_sync_dir_command(src, dest)
        return download_command

    def to_cluster(self, dest_cluster, url=None, mount=False, return_dest_folder=False):
        """Copy the folder from a s3 bucket onto a cluster."""
        download_command = self.download_command(src=self.fsspec_url, dest=url)
        dest_cluster.run([download_command])
        if return_dest_folder:
            return S3Folder(url=url, dryrun=True).from_cluster(dest_cluster)

    def to_local(
        self, dest_url: str, data_config: dict, return_dest_folder: bool = False
    ):
        """Copy a folder from an S3 bucket to local dir."""
        self.download(dest=dest_url)
        if return_dest_folder:
            return self.destination_folder(
                dest_url=dest_url, dest_fs="file", data_config=data_config
            )

    def to_data_store(
        self,
        fs: str,
        data_store_url: Optional[str] = None,
        data_config: Optional[dict] = None,
        return_dest_folder: bool = True,
    ):
        """Copy folder from S3 to another remote data store (ex: S3, GCP, Azure)"""
        if fs == "s3":
            # Transfer between S3 folders
            from sky.data.storage import S3Store

            sync_dir_command = self.upload_command(
                src=self.fsspec_url, dest=data_store_url
            )
            self.run_upload_cli_cmd(
                sync_dir_command, access_denied_message=S3Store.ACCESS_DENIED_MESSAGE
            )
        elif fs == "gs":
            from sky.data import data_transfer

            # Note: The sky data transfer API only allows for transfers between buckets, not specific directories.
            logger.warning(
                "Transfer from S3 to GCS currently supported for buckets only, not specific directories."
            )
            data_store_url = self.bucket_name_from_url(data_store_url)
            data_transfer.s3_to_gcs(
                s3_bucket_name=self.bucket_name_from_url(self.url),
                gs_bucket_name=data_store_url,
            )
        elif fs == "azure":
            raise NotImplementedError("Azure not yet supported")
        else:
            raise ValueError(f"Invalid fs: {fs}")

        if return_dest_folder:
            return self.destination_folder(
                dest_url=data_store_url, dest_fs=fs, data_config=data_config
            )
