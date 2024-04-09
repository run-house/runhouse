import logging
import subprocess
import time
from typing import Optional

from .folder import Folder

logger = logging.getLogger(__name__)

MAX_POLLS = 120000
POLL_INTERVAL = 1
TIMEOUT_SECONDS = 3600


class S3Folder(Folder):
    RESOURCE_TYPE = "folder"
    DEFAULT_FS = "s3"

    def __init__(self, dryrun: bool, **kwargs):
        super().__init__(dryrun=dryrun, **kwargs)

    @staticmethod
    def from_config(config: dict, dryrun=False, _resolve_children=True):
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
        sync_dir_command = self._upload_command(src=src, dest=self.path)
        self._run_upload_cli_cmd(sync_dir_command)

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
        try:
            subprocess.run(
                ["aws", "s3", "sync", remote_dir, dest],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            raise e

    def _download_command(self, src: str, dest: str):
        # https://github.com/skypilot-org/skypilot/blob/983f5fa3197fe7c4b5a28be240f7b027f7192b15/sky/cloud_stores.py#L68
        download_via_awscli = "aws s3 sync --no-follow-symlinks " f"{src} {dest}"
        return download_via_awscli

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
            sync_dir_command = self._upload_command(
                src=self.fsspec_url, dest=data_store_path
            )
            self._run_upload_cli_cmd(sync_dir_command)
        elif system == "gs":
            # Note: The sky data transfer API only allows for transfers between buckets, not specific directories.
            logger.warning(
                "Transfer from S3 to GCS currently supported for buckets only, not specific directories."
            )
            s3_bucket_name = self._bucket_name_from_path(self.path)
            gs_bucket_name = self._bucket_name_from_path(data_store_path)
            self.s3_to_gcs(
                s3_bucket_name=s3_bucket_name,
                gs_bucket_name=gs_bucket_name,
            )
        elif system == "azure":
            raise NotImplementedError("Azure not yet supported")
        else:
            raise ValueError(f"Invalid system: {system}")

        return self.destination_folder(
            dest_path=data_store_path, dest_system=system, data_config=data_config
        )

    def s3_to_gcs(self, s3_bucket_name: str, gs_bucket_name: str):
        import boto3
        from google import auth
        from googleapiclient import discovery
        from oauth2client.client import GoogleCredentials

        from runhouse import GCSFolder

        # Adapted from:
        # https://github.com/skypilot-org/skypilot/blob/983f5fa3197fe7c4b5a28be240f7b027f7192b15/sky/data/data_transfer.py#L36

        oauth_credentials = GoogleCredentials.get_application_default()
        storagetransfer = discovery.build(
            "storagetransfer", "v1", credentials=oauth_credentials
        )

        try:
            credentials, project_id = auth.default()
        except Exception as e:
            raise e

        if project_id is None:
            raise ValueError(
                "Failed to get GCP project id. Please make sure you have "
                "run the following: gcloud init; "
                "Alternatively, set the project ID using: gcloud config set project <project_id> "
                "or with an environment variable: export GOOGLE_CLOUD_PROJECT=<project_id>"
            )

        session = boto3.Session()

        aws_credentials = session.get_credentials().get_frozen_credentials()

        storage_account = (
            storagetransfer.googleServiceAccounts().get(projectId=project_id).execute()
        )

        GCSFolder.add_bucket_iam_member(
            gs_bucket_name,
            "roles/storage.admin",
            "serviceAccount:" + storage_account["accountEmail"],
            project_id=project_id,
        )

        transfer_job = {
            "description": f"Transferring data from S3 Bucket \
            {s3_bucket_name} to GCS Bucket {gs_bucket_name}",
            "status": "ENABLED",
            "projectId": project_id,
            "transferSpec": {
                "awsS3DataSource": {
                    "bucketName": s3_bucket_name,
                    "awsAccessKey": {
                        "accessKeyId": aws_credentials.access_key,
                        "secretAccessKey": aws_credentials.secret_key,
                    },
                },
                "gcsDataSink": {
                    "bucketName": gs_bucket_name,
                },
            },
        }

        response = storagetransfer.transferJobs().create(body=transfer_job).execute()

        operation = (
            storagetransfer.transferJobs()
            .run(jobName=response["name"], body={"projectId": project_id})
            .execute()
        )
        logger.info(
            f"Transfer job scheduled: s3://{s3_bucket_name} -> gs://{gs_bucket_name}"
        )

        logger.info("Waiting for the transfer to finish")

        timeout = False
        start = time.time()

        while True:
            # Get the status of the transfer operation
            result = (
                storagetransfer.transferOperations()
                .get(name=operation["name"])
                .execute()
            )

            if "error" in result:
                raise RuntimeError(result["error"])

            if "done" in result and result["done"]:
                logger.info(
                    f"Transfer finished in {(time.time() - start) / 60:.2f} minutes."
                )
                break

            # Check if the elapsed time exceeds the timeout
            if (time.time() - start) > TIMEOUT_SECONDS:
                timeout = True
                break

            time.sleep(POLL_INTERVAL)

        if timeout:
            logger.info(
                f"Transfer timed out after {(time.time() - start) / TIMEOUT_SECONDS:.2f} "
                "hours. Please check the status of the transfer job in the GCP "
                "Storage Transfer Service console at "
                "https://cloud.google.com/storage-transfer-service"
            )
