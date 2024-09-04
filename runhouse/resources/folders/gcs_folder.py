import copy
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

from runhouse.logger import get_logger

from .folder import Folder

logger = get_logger(__name__)


class GCSFolder(Folder):
    RESOURCE_TYPE = "folder"
    DEFAULT_FS = "gcp"

    def __init__(self, dryrun: bool, **kwargs):
        from google.cloud import storage

        super().__init__(dryrun=dryrun, **kwargs)
        self.client = storage.Client()
        self._urlpath = "gs://"

    @staticmethod
    def from_config(config: Dict, dryrun: bool = False, _resolve_children: bool = True):
        """Load config values into the object."""
        return GCSFolder(**config, dryrun=dryrun)

    @property
    def bucket(self):
        return self.client.bucket(self._bucket_name)

    def _to_local(self, dest_path: str):
        """Copies folder to local."""
        from runhouse import Cluster

        if self._fs_str == "file":
            shutil.copytree(src=self.path, dst=dest_path)
        elif isinstance(self.system, Cluster):
            return self._cluster_to_local(cluster=self.system, dest_path=dest_path)
        else:
            self._gcs_copy_to_local(dest_path)

        return self._destination_folder(dest_path=dest_path, dest_system="file")

    def _gcs_copy_to_local(self, dest_path: str):
        """Copy GCS folder to local."""
        Path(dest_path).mkdir(parents=True, exist_ok=True)
        key = self._key
        blobs = self.client.list_blobs(self.bucket.name, prefix=key)
        for blob in blobs:
            dest_file_path = Path(dest_path) / Path(blob.name).relative_to(key)
            dest_file_path.parent.mkdir(parents=True, exist_ok=True)
            blob.download_to_filename(str(dest_file_path))

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

    def _move_within_gcs(self, new_path):
        key = self._key
        blobs = self.client.list_blobs(self.bucket.name, prefix=key)
        for blob in blobs:
            old_name = blob.name
            new_name = new_path + old_name[len(key) :]
            self.bucket.copy_blob(blob, self.bucket, new_name)
            blob.delete()

    def _gcs_to_local(self, local_path):
        key = self._key
        Path(local_path).mkdir(parents=True, exist_ok=True)
        blobs = self.client.list_blobs(self.bucket.name, prefix=key)
        for blob in blobs:
            dest_file_path = Path(local_path) / Path(blob.name).relative_to(key)
            dest_file_path.parent.mkdir(parents=True, exist_ok=True)
            blob.download_to_filename(str(dest_file_path))
            blob.delete()

    def _gcs_copy(self, new_path):
        key = self._key
        blobs = self.client.list_blobs(self.bucket.name, prefix=key)
        for blob in blobs:
            old_name = blob.name
            new_name = new_path + old_name[len(key) :]
            new_blob = self.bucket.blob(new_name)
            new_blob.rewrite(blob)

    def put(self, contents, overwrite=False, mode: str = "wb"):
        """Put given contents in folder."""
        self.mkdir()
        if isinstance(contents, list):
            for resource in contents:
                self.put(resource, overwrite=overwrite)
            return

        key = self._key

        if isinstance(contents, GCSFolder):
            if contents.folder_path is None:
                contents.folder_path = key + "/" + contents.folder_path
            return

        if not isinstance(contents, dict):
            raise TypeError(
                "`contents` argument must be a dict mapping filenames to file-like objects"
            )

        if overwrite is False:
            # Check if files exist and raise an error if they do
            existing_files = [
                blob.name for blob in self.client.list_blobs(self.bucket, prefix=key)
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
                blob = self.bucket.blob(file_key)
                file_obj = self._serialize_file_obj(file_obj)
                blob.upload_from_file(file_obj)

            except Exception as e:
                raise RuntimeError(f"Failed to upload {filename} to GCS: {e}")

    def mv(self, system, path: Optional[str] = None):
        """Move the folder to a new filesystem or cluster."""
        if path is None:
            path = "rh/" + self.rns_address

        if system == "gcs":
            self._move_within_gcs(path)
        elif system == "file":
            self._gcs_to_local(path)
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
        blobs = list(self.client.list_blobs(self.bucket, prefix=self._key))

        if sort:
            blobs = sorted(blobs, key=lambda f: f.updated, reverse=True)

        if full_paths:
            return [self._urlpath + f"{self.bucket.name}/{blob.name}" for blob in blobs]
        else:
            return [Path(blob.name).name for blob in blobs]

    def exists_in_system(self):
        """Whether the folder exists in the filesystem."""
        try:
            blobs = list(
                self.client.list_blobs(self.bucket, prefix=self._key, max_results=1)
            )
            return len(blobs) > 0

        except Exception as e:
            logger.error(f"Failed to check if folder exists: {e}")
            return False

    def open(self, name, mode="rb", encoding=None):
        """Returns a GCS blob stream which must be used as a content manager to be opened.

        Example:
            >>> with my_folder.open('obj_name') as my_file:
            >>>     pickle.load(my_file)
        """
        blob_name = self._key + name
        blob = self.bucket.blob(blob_name)

        try:
            if "r" not in mode:
                raise NotImplementedError(f"{mode} mode is not implemented yet for GCS")

            return blob.open(mode=mode, encoding=encoding)

        except Exception as e:
            raise e

    def mkdir(self):
        """Create the folder in specified file system if it doesn't already exist."""
        try:
            key = self._key
            blob = self.bucket.blob()
            blob.upload_from_string("")
            logger.info(
                f"Directory {key} created successfully in bucket {self._bucket_name}."
            )
            return self
        except Exception as e:
            raise e

    def rm(self, contents: list = None, recursive: bool = True):
        """Delete a folder from the GCS bucket. Optionally provide a list of folder contents to delete.

        Args:
            contents (Optional[List]): Specific contents to delete in the folder.
            recursive (bool): Delete the folder itself (including all its contents).
                Defaults to ``True``.
        """
        key = self._key
        bucket = self._bucket_name
        if contents:
            blobs_to_delete = [
                self.bucket.blob(f"{key}{content}") for content in contents
            ]
        else:
            if recursive:
                blobs_to_delete = list(self.client.list_blobs(bucket, prefix=key))
            else:
                blobs_to_delete = [self.bucket.blob(key)]

        for blob in blobs_to_delete:
            blob.delete()

    def delete_bucket(self):
        """Delete the gcs bucket."""
        # https://github.com/skypilot-org/skypilot/blob/3517f55ed074466eadd4175e152f68c5ea3f5f4c/sky/data/storage.py#L1775
        bucket_name = self._bucket_name
        remove_obj_command = f"rm -r {self._urlpath}{bucket_name}"

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
        self._upload_folder_to_bucket(sync_dir_command)

    def _upload_command(self, src: str, dest: str):
        # https://github.com/skypilot-org/skypilot/blob/983f5fa3197fe7c4b5a28be240f7b027f7192b15/sky/data/storage.py#L1240
        dest = dest.lstrip("/")
        return f"gsutil -m rsync -r -x '.git/*' {src} {self._urlpath}{dest}"

    def _download(self, dest):
        """Download a folder from a GCS bucket to local dir."""
        # NOTE: Sky doesn't support this API yet for each provider
        # https://github.com/skypilot-org/skypilot/blob/983f5fa3197fe7c4b5a28be240f7b027f7192b15/sky/data/storage.py#L231
        remote_dir = self.path.lstrip("/")
        remote_dir = f"{self._urlpath}{remote_dir}"
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

    def _to_cluster(self, dest_cluster, path=None):
        upload_command = self._upload_command(src=self.path, dest=path)
        dest_cluster.run([upload_command])
        return GCSFolder(path=path, system=dest_cluster, dryrun=True)

    def _to_local(self, dest_path: str):
        """Copy a folder from an GCS bucket to local dir."""
        self._download(dest=dest_path)
        return self._destination_folder(dest_path=dest_path, dest_system="file")

    def _to_data_store(
        self,
        system: str,
        data_store_path: Optional[str] = None,
    ):
        """Copy folder from GCS to another remote data store (ex: GCS, S3, Azure)"""
        if system == "gs":
            # Transfer between GCS folders
            sync_dir_command = self._upload_command(
                src=self.fsspec_url, dest=data_store_path
            )
            self._upload_folder_to_bucket(sync_dir_command)
        elif system == "s3":
            # Note: The sky data transfer API only allows for transfers between buckets, not specific directories.
            logger.warning(
                "Transfer from GCS to S3 currently supported for buckets only, not specific directories."
            )
            gs_bucket_name = self._bucket_name
            s3_bucket_name = self._bucket_name_from_path(data_store_path)
            self.gcs_to_s3(
                gs_bucket_name=gs_bucket_name,
                s3_bucket_name=s3_bucket_name,
            )
        elif system == "azure":
            raise NotImplementedError("Azure not yet supported")
        else:
            raise ValueError(f"Invalid system: {system}")

        return self._destination_folder(dest_path=data_store_path, dest_system=system)

    def gcs_to_s3(self, gs_bucket_name: str, s3_bucket_name: str) -> None:
        # https://github.com/skypilot-org/skypilot/blob/3517f55ed074466eadd4175e152f68c5ea3f5f4c/sky/data/data_transfer.py#L138
        disable_multiprocessing_flag = '-o "GSUtil:parallel_process_count=1"'
        sync_command = f"gsutil -m {disable_multiprocessing_flag} rsync -rd {self._urlpath}{gs_bucket_name} s3://{s3_bucket_name}"
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
