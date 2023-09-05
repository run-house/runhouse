import logging
import subprocess
from typing import Optional

from .folder import Folder

logger = logging.getLogger(__name__)


class S3Folder(Folder):
    RESOURCE_TYPE = "folder"
    DEFAULT_FS = "s3"

    def __init__(self, dryrun: bool, **kwargs):
        super().__init__(dryrun=dryrun, **kwargs)

    @staticmethod
    def from_config(config: dict, dryrun=False):
        """Load config values into the object."""
        return S3Folder(**config, dryrun=dryrun)

    def delete_bucket(self):
        """Delete the s3 bucket."""
        try:
            from sky.data.storage import S3Store

            S3Store(
                name=self._bucket_name_from_path(self.path), source=self._fsspec_fs
            ).delete()
        except Exception as e:
            raise e

    def _upload(self, src: str, region: Optional[str] = None):
        """Upload a folder to an S3 bucket."""
        from sky.data.storage import S3Store

        # Initialize the S3Store object which creates the bucket if it does not exist
        s3_store = S3Store(
            name=self._bucket_name_from_path(self.path), source=src, region=region
        )

        sync_dir_command = self._upload_command(src=src, dest=self.path)
        self._run_upload_cli_cmd(
            sync_dir_command, access_denied_message=s3_store._ACCESS_DENIED_MESSAGE
        )

    def _upload_command(self, src: str, dest: str):
        # https://github.com/skypilot-org/skypilot/blob/983f5fa3197fe7c4b5a28be240f7b027f7192b15/sky/data/storage.py#L922
        dest = dest.lstrip("/")
        upload_command = (
            'aws s3 sync --no-follow-symlinks --exclude ".git/*" '
            f"{src} "
            f"s3://{dest}"
        )

        return upload_command

    def _download(self, dest):
        """Download a folder from an S3 bucket to local dir."""
        # NOTE: Sky doesn't support this API yet for each provider
        # https://github.com/skypilot-org/skypilot/blob/983f5fa3197fe7c4b5a28be240f7b027f7192b15/sky/data/storage.py#L231
        remote_dir = self.path.lstrip("/")
        remote_dir = f"s3://{remote_dir}"
        subprocess.run(
            ["aws", "s3", "sync", remote_dir, dest],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )

    def _download_command(self, src, dest):
        from sky.cloud_stores import S3CloudStorage

        download_command = S3CloudStorage().make_sync_dir_command(src, dest)
        return download_command

    def _to_cluster(self, dest_cluster, path=None, mount=False):
        """Copy the folder from a s3 bucket onto a cluster."""
        download_command = self._download_command(src=self.fsspec_url, dest=path)
        dest_cluster.run([download_command])
        return S3Folder(path=path, system=dest_cluster, dryrun=True)

    def _to_local(self, dest_path: str, data_config: dict):
        """Copy a folder from an S3 bucket to local dir."""
        self._download(dest=dest_path)
        return self.destination_folder(
            dest_path=dest_path, dest_system="file", data_config=data_config
        )

    def _to_data_store(
        self,
        system: str,
        data_store_path: Optional[str] = None,
        data_config: Optional[dict] = None,
    ):
        """Copy folder from S3 to another remote data store (ex: S3, GCP, Azure)"""
        if system == "s3":
            # Transfer between S3 folders
            from sky.data.storage import S3Store

            sync_dir_command = self._upload_command(
                src=self.fsspec_url, dest=data_store_path
            )
            self._run_upload_cli_cmd(
                sync_dir_command, access_denied_message=S3Store._ACCESS_DENIED_MESSAGE
            )
        elif system == "gs":
            from sky.data import data_transfer

            # Note: The sky data transfer API only allows for transfers between buckets, not specific directories.
            logger.warning(
                "Transfer from S3 to GCS currently supported for buckets only, not specific directories."
            )
            data_store_path = self._bucket_name_from_path(data_store_path)
            data_transfer.s3_to_gcs(
                s3_bucket_name=self._bucket_name_from_path(self.path),
                gs_bucket_name=data_store_path,
            )
        elif system == "azure":
            raise NotImplementedError("Azure not yet supported")
        else:
            raise ValueError(f"Invalid system: {system}")

        return self.destination_folder(
            dest_path=data_store_path, dest_system=system, data_config=data_config
        )
