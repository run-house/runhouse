import logging
import subprocess
from typing import Optional

from .folder import Folder

logger = logging.getLogger(__name__)


class GCSFolder(Folder):
    RESOURCE_TYPE = "folder"
    DEFAULT_FS = "gcp"

    def __init__(self, dryrun: bool, **kwargs):
        super().__init__(dryrun=dryrun, **kwargs)

    @staticmethod
    def from_config(config: dict, dryrun=False):
        """Load config values into the object."""
        return GCSFolder(**config, dryrun=dryrun)

    def delete_bucket(self):
        """Delete the gcs bucket."""
        # https://github.com/skypilot-org/skypilot/blob/3517f55ed074466eadd4175e152f68c5ea3f5f4c/sky/data/storage.py#L1775
        bucket_name = self._bucket_name_from_path(self.path)
        remove_obj_command = f"rm -r gs://{bucket_name}"

        try:
            subprocess.check_output(
                remove_obj_command,
                stderr=subprocess.STDOUT,
                shell=True,
                executable="/bin/bash",
            )
        except subprocess.CalledProcessError as e:
            raise e

    def _upload(self, src: str, region: Optional[str] = None):
        """Upload a folder to an GCS bucket."""
        sync_dir_command = self._upload_command(src=src, dest=self.path)
        self._run_upload_cli_cmd(sync_dir_command)

    def _upload_command(self, src: str, dest: str):
        # https://github.com/skypilot-org/skypilot/blob/983f5fa3197fe7c4b5a28be240f7b027f7192b15/sky/data/storage.py#L1240
        dest = dest.lstrip("/")
        return f"gsutil -m rsync -r -x '.git/*' {src} gs://{dest}"

    def _download(self, dest):
        """Download a folder from a GCS bucket to local dir."""
        # NOTE: Sky doesn't support this API yet for each provider
        # https://github.com/skypilot-org/skypilot/blob/983f5fa3197fe7c4b5a28be240f7b027f7192b15/sky/data/storage.py#L231
        remote_dir = self.path.lstrip("/")
        remote_dir = f"gs://{remote_dir}"
        subprocess.run(
            ["gsutil", "-m", "rsync", "-r", "-x", ".git/*", remote_dir, dest],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )

    def _download_command(self, src, dest):
        # https://github.com/skypilot-org/skypilot/blob/3517f55ed074466eadd4175e152f68c5ea3f5f4c/sky/cloud_stores.py#L139
        download_via_gsutil = f"gsutil -m rsync -e -r {src} {dest}"
        return download_via_gsutil

    def _to_cluster(self, dest_cluster, path=None, mount=False):
        upload_command = self._upload_command(src=self.path, dest=path)
        dest_cluster.run([upload_command])
        return GCSFolder(path=path, system=dest_cluster, dryrun=True)

    def _to_local(self, dest_path: str, data_config: dict):
        """Copy a folder from an GCS bucket to local dir."""
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
        """Copy folder from GCS to another remote data store (ex: GCS, S3, Azure)"""
        if system == "gs":
            # Transfer between GCS folders
            sync_dir_command = self._upload_command(
                src=self.fsspec_url, dest=data_store_path
            )
            self._run_upload_cli_cmd(sync_dir_command)
        elif system == "s3":
            # Note: The sky data transfer API only allows for transfers between buckets, not specific directories.
            logger.warning(
                "Transfer from GCS to S3 currently supported for buckets only, not specific directories."
            )
            gs_bucket_name = self._bucket_name_from_path(self.path)
            s3_bucket_name = self._bucket_name_from_path(data_store_path)
            self.gcs_to_s3(
                gs_bucket_name=gs_bucket_name,
                s3_bucket_name=s3_bucket_name,
            )
        elif system == "azure":
            raise NotImplementedError("Azure not yet supported")
        else:
            raise ValueError(f"Invalid system: {system}")

        return self.destination_folder(
            dest_path=data_store_path, dest_system=system, data_config=data_config
        )

    def gcs_to_s3(self, gs_bucket_name: str, s3_bucket_name: str) -> None:
        # https://github.com/skypilot-org/skypilot/blob/3517f55ed074466eadd4175e152f68c5ea3f5f4c/sky/data/data_transfer.py#L138
        disable_multiprocessing_flag = '-o "GSUtil:parallel_process_count=1"'
        sync_command = f"gsutil -m {disable_multiprocessing_flag} rsync -rd gs://{gs_bucket_name} s3://{s3_bucket_name}"
        try:
            subprocess.run(sync_command, shell=True)
        except subprocess.CalledProcessError as e:
            raise e

    @staticmethod
    def add_bucket_iam_member(
        bucket_name: str, role: str, member: str, project_id: str
    ) -> None:
        # https://github.com/skypilot-org/skypilot/blob/983f5fa3197fe7c4b5a28be240f7b027f7192b15/sky/data/data_transfer.py#L132
        from google.cloud import storage

        storage_client = storage.Client(project=project_id)
        bucket = storage_client.bucket(bucket_name)

        policy = bucket.get_iam_policy(requested_policy_version=3)
        policy.bindings.append({"role": role, "members": {member}})

        bucket.set_iam_policy(policy)

        logger.debug(f"Added {member} with role {role} to {bucket_name}.")
