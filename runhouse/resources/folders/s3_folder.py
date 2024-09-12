import copy
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

from runhouse.logger import get_logger

from .folder import Folder

MAX_POLLS = 120000
POLL_INTERVAL = 1
TIMEOUT_SECONDS = 3600

logger = get_logger(__name__)


class S3Folder(Folder):
    RESOURCE_TYPE = "folder"
    DEFAULT_FS = "s3"

    def __init__(self, dryrun: bool, **kwargs):
        import boto3

        super().__init__(dryrun=dryrun, **kwargs)
        self.client = boto3.client("s3")
        self._urlpath = "s3://"

    @staticmethod
    def from_config(config: Dict, dryrun: bool = False, _resolve_children: bool = True):
        """Load config values into the object."""
        return S3Folder(**config, dryrun=dryrun)

    def _to_local(self, dest_path: str):
        """Copies folder to local."""
        from runhouse import Cluster

        if self._fs_str == "file":
            shutil.copytree(src=self.path, dst=dest_path)
        elif isinstance(self.system, Cluster):
            return self._cluster_to_local(cluster=self.system, dest_path=dest_path)
        else:
            self._s3_copy_to_local(dest_path)

        return self._destination_folder(dest_path=dest_path, dest_system="file")

    def _s3_copy_to_local(self, dest_path: str):
        """Copy S3 folder to local."""
        Path(dest_path).mkdir(parents=True, exist_ok=True)

        bucket_name = self._bucket_name
        key = self._key

        s3_objects = self.client.list_objects_v2(Bucket=bucket_name, Prefix=key)

        for obj in s3_objects.get("Contents", []):
            obj_key = obj["Key"]
            dest_file_path = Path(dest_path) / Path(obj_key).relative_to(key)
            dest_file_path.parent.mkdir(parents=True, exist_ok=True)
            self.client.download_file(bucket_name, obj_key, str(dest_file_path))

    def _cluster_to_local(self, cluster, dest_path):
        if not cluster.address:
            raise ValueError("Cluster must be started before copying data from it.")

        Path(dest_path).expanduser().mkdir(parents=True, exist_ok=True)
        cluster.rsync(
            source=self.path,
            dest=str(Path(dest_path).expanduser()),
            up=False,
            contents=True,
        )
        new_folder = copy.deepcopy(self)
        new_folder.path = dest_path
        new_folder.system = "file"
        return new_folder

    def _move_within_s3(self, new_path):
        bucket_name = self._bucket_name
        key = self._key
        s3_objects = self.client.list_objects_v2(Bucket=bucket_name, Prefix=key)
        for obj in s3_objects.get("Contents", []):
            old_key = obj["Key"]
            new_key = new_path + old_key[len(key) :]
            self.client.copy_object(
                Bucket=bucket_name,
                CopySource={"Bucket": bucket_name, "Key": old_key},
                Key=new_key,
            )
            self.client.delete_object(Bucket=bucket_name, Key=old_key)

    def _s3_to_local(self, local_path):
        bucket_name = self._bucket_name
        key = self._key
        Path(local_path).mkdir(parents=True, exist_ok=True)
        s3_objects = self.client.list_objects_v2(Bucket=bucket_name, Prefix=key)
        for obj in s3_objects.get("Contents", []):
            obj_key = obj["Key"]
            dest_file_path = Path(local_path) / Path(obj_key).relative_to(key)
            dest_file_path.parent.mkdir(parents=True, exist_ok=True)
            self.client.download_file(bucket_name, obj_key, str(dest_file_path))
            self.client.delete_object(Bucket=bucket_name, Key=obj_key)

    def _s3_copy(self, new_path):
        bucket_name = self._bucket_name
        key = self._key
        s3_objects = self.client.list_objects_v2(Bucket=bucket_name, Prefix=key)
        for obj in s3_objects.get("Contents", []):
            old_key = obj["Key"]
            new_key = new_path + old_key[len(key) :]
            self.client.copy_object(
                Bucket=bucket_name,
                CopySource={"Bucket": bucket_name, "Key": old_key},
                Key=new_key,
            )

    def put(
        self,
        contents: Union["S3Folder", Dict],
        overwrite: bool = False,
        mode: str = "wb",
    ):
        """Put given contents in folder."""
        self.mkdir()
        if isinstance(contents, list):
            for resource in contents:
                self.put(resource, overwrite=overwrite)
            return

        key = self._key
        bucket_name = self._bucket_name

        if isinstance(contents, S3Folder):
            if contents.folder_path is None:
                contents.folder_path = key + "/" + contents.folder_path
            return

        if not isinstance(contents, Dict):
            raise TypeError(
                "`contents` argument must be a dict mapping filenames to file-like objects"
            )

        if overwrite is False:
            # Check if files exist and raise an error if they do
            existing_files = [
                obj["Key"]
                for obj in self.client.list_objects_v2(
                    Bucket=bucket_name, Prefix=key
                ).get("Contents", [])
            ]
            intersection = set(existing_files).intersection(set(contents.keys()))
            if intersection:
                raise FileExistsError(
                    f"File(s) {intersection} already exist(s) at path {key}, "
                    f"cannot save them without overwriting."
                )

        for filename, file_obj in contents.items():
            file_key = key + filename
            try:
                body = self._serialize_file_obj(file_obj)
                self.client.put_object(Bucket=bucket_name, Key=file_key, Body=body)

            except Exception as e:
                raise RuntimeError(f"Failed to upload {filename} to S3: {e}")

    def mv(self, system, path: Optional[str] = None):
        """Move the folder to a new filesystem or cluster."""
        if path is None:
            path = "rh/" + self.rns_address

        if system == "s3":
            self._move_within_s3(path)
        elif system == "file":
            self._s3_to_local(path)
        else:
            raise NotImplementedError("System not supported")

        self.path = path
        self.system = system

    def ls(self, full_paths: bool = True, sort: bool = False) -> List:
        """List the contents of the folder.

        Args:
            full_paths (Optional[bool]): Whether to list the full paths of the folder contents.
                Defaults to ``True``.
            sort (Optional[bool]): Whether to sort the folder contents by time modified.
                Defaults to ``False``.
        """
        paginator = self.client.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=self._bucket_name, Prefix=self._key)

        paths = []
        for page in pages:
            for obj in page.get("Contents", []):
                paths.append(obj)

        if sort:
            paths = sorted(paths, key=lambda f: f["LastModified"], reverse=True)

        if full_paths:
            return [
                self._urlpath + f"{self._bucket_name}/{obj['Key']}" for obj in paths
            ]
        else:
            return [Path(obj["Key"]).name for obj in paths]

    def exists_in_system(self):
        """Whether the folder exists in the filesystem."""
        try:
            # Check if there are any objects with the given prefix
            response = self.client.list_objects_v2(
                Bucket=self._bucket_name, Prefix=self._key, MaxKeys=1
            )
            return "Contents" in response
        except Exception as e:
            logger.error(f"Failed to check if folder exists: {e}")
            return False

    def open(self, name, mode="rb", encoding=None):
        """Returns an S3 object stream which must be used as a content manager to be opened.

        Example:
            >>> with my_folder.open('obj_name') as my_file:
            >>>     pickle.load(my_file)
        """
        obj_key = self._key + name
        bucket_name = self._bucket_name

        try:
            obj = self.client.get_object(Bucket=bucket_name, Key=obj_key)
            if "r" not in mode:
                raise NotImplementedError(f"{mode} mode is not implemented yet for S3")

            return obj["Body"]

        except Exception as e:
            raise e

    def mkdir(self):
        """Create the folder in specified file system if it doesn't already exist."""
        try:
            bucket = self._bucket_name
            key = self._key
            self.client.put_object(Bucket=bucket, Key=key)
            logger.info(
                f"Folder with path: {key} created successfully in bucket {bucket}."
            )
            return self

        except Exception as e:
            raise e

    def rm(self, contents: list = None, recursive: bool = True):
        """Delete a folder from the S3 bucket. Optionally provide a list of folder contents to delete.

        Args:
            contents (Optional[List]): Specific contents to delete in the folder.
            recursive (bool): Delete the folder itself (including all its contents).
                Defaults to ``True``.
        """
        key = self._key
        bucket = self._bucket_name
        if contents:
            objects_to_delete = [{"Key": f"{key}{content}"} for content in contents]
        else:
            if recursive:
                paginator = self.client.get_paginator("list_objects_v2")
                pages = paginator.paginate(Bucket=bucket, Prefix=key)
                objects_to_delete = [
                    {"Key": obj["Key"]}
                    for page in pages
                    for obj in page.get("Contents", [])
                ]
            else:
                objects_to_delete = [{"Key": key}]

        if objects_to_delete:
            self.client.delete_objects(
                Bucket=bucket, Delete={"Objects": objects_to_delete}
            )

    def delete_bucket(self):
        """Delete the s3 bucket."""
        try:
            from sky.data.storage import S3Store

            S3Store(name=self._bucket_name, source=self._urlpath).delete()
        except Exception as e:
            raise e

    def _upload(self, src: str, region: Optional[str] = None):
        """Upload a folder to an S3 bucket."""
        sync_dir_command = self._upload_command(src=src, dest=self.path)
        self._upload_folder_to_bucket(sync_dir_command)

    def _upload_command(self, src: str, dest: str):
        # https://github.com/skypilot-org/skypilot/blob/983f5fa3197fe7c4b5a28be240f7b027f7192b15/sky/data/storage.py#L922
        dest = dest.lstrip("/")
        upload_command = (
            'aws s3 sync --no-follow-symlinks --exclude ".git/*" '
            f"{src} "
            f"{self._urlpath}{dest}"
        )

        return upload_command

    def _download(self, dest):
        """Download a folder from an S3 bucket to local dir."""
        # NOTE: Sky doesn't support this API yet for each provider
        # https://github.com/skypilot-org/skypilot/blob/983f5fa3197fe7c4b5a28be240f7b027f7192b15/sky/data/storage.py#L231
        remote_dir = self.path.lstrip("/")
        remote_dir = self._urlpath + remote_dir
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

    def _to_cluster(self, dest_cluster, path=None):
        """Copy the folder from a s3 bucket onto a cluster."""
        download_command = self._download_command(src=self.fsspec_url, dest=path)
        dest_cluster.run([download_command])
        return S3Folder(path=path, system=dest_cluster, dryrun=True)

    def _to_local(self, dest_path: str):
        """Copy a folder from an S3 bucket to local dir."""
        self._download(dest=dest_path)
        return self._destination_folder(dest_path=dest_path, dest_system="file")

    def _to_data_store(
        self,
        system: str,
        data_store_path: Optional[str] = None,
    ):
        """Copy folder from S3 to another remote data store (ex: S3, GCP, Azure)"""
        if system == "s3":
            # Transfer between S3 folders
            sync_dir_command = self._upload_command(
                src=self.fsspec_url, dest=data_store_path
            )
            self._upload_folder_to_bucket(sync_dir_command)
        elif system == "gs":
            # Note: The sky data transfer API only allows for transfers between buckets, not specific directories.
            logger.warning(
                "Transfer from S3 to GCS currently supported for buckets only, not specific directories."
            )
            s3_bucket_name = self._bucket_name
            gs_bucket_name = self._bucket_name_from_path(data_store_path)
            self.s3_to_gcs(
                s3_bucket_name=s3_bucket_name,
                gs_bucket_name=gs_bucket_name,
            )
        elif system == "azure":
            raise NotImplementedError("Azure not yet supported")
        else:
            raise ValueError(f"Invalid system: {system}")

        return self._destination_folder(dest_path=data_store_path, dest_system=system)

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
            f"Transfer job scheduled: {self._urlpath}{s3_bucket_name} -> gs://{gs_bucket_name}"
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
