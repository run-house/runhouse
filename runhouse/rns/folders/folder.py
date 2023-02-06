import copy
import logging
import os
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import fsspec

# TODO [DG] flip this when we switch to sshfs
import sshfs

import runhouse as rh
from runhouse.rh_config import configs, rns_client
from runhouse.rns.api_utils.resource_access import ResourceAccess
from runhouse.rns.resource import Resource

fsspec.register_implementation("ssh", sshfs.SSHFileSystem)
# SSHFileSystem is not yet builtin.
# Line above suggested by fsspec devs: https://github.com/fsspec/filesystem_spec/issues/1071

logger = logging.getLogger(__name__)

PROVIDER_FS_LOOKUP = {
    "aws": "s3",
    "gcp": "gs",
    "azure": "abfs",
    "oracle": "ocifs",
    "databricks": "dbfs",
    "github": "github",
}


class Folder(Resource):
    RESOURCE_TYPE = "folder"
    DEFAULT_FS = "file"
    CLUSTER_FS = "ssh"
    DEFAULT_FOLDER_PATH = "/runhouse"
    DEFAULT_CACHE_FOLDER = "~/.cache/runhouse"

    def __init__(
        self,
        name: Optional[str] = None,
        url: Optional[str] = None,
        fs: Optional[str] = DEFAULT_FS,
        dryrun: bool = True,
        local_mount: bool = False,
        data_config: Optional[Dict] = None,
        **kwargs,  # We have this here to ignore extra arguments when calling from from_config
    ):
        """
        TODO [DG] Update
        Include loud warning that relative paths are relative to the git root / working directory!
        Args:
            name ():
            parent (): string path to parent folder, or
            data_source (): FSSpec protocol, e.g. 's3', 'gs'. See/run `fsspec.available_protocols()`.
                Default is "file", the local filesystem to wherever the blob is created.
            data_config ():
            local_path ():
        """
        super().__init__(name=name, dryrun=dryrun)

        self.fs = fs

        # TODO [DG] Should we ever be allowing this to be None?
        self._url = (
            self.default_url(self.rns_address, fs)
            if url is None
            else url
            if isinstance(fs, Resource)
            else url
            if Path(url).expanduser().is_absolute()
            else str(Path(rns_client.locate_working_dir()) / url)
        )
        self.data_config = data_config or {}

        self.local_mount = local_mount
        self._local_mount_path = None
        if local_mount:
            self.mount(tmp=True)
        if not self.dryrun and self.is_local():
            Path(self.url).mkdir(parents=True, exist_ok=True)

    @classmethod
    def default_url(cls, rns_address, fs):
        from runhouse.rns.hardware import Cluster

        if fs == Folder.DEFAULT_FS or isinstance(fs, Cluster):
            if rns_address:
                return str(
                    Path.cwd() / rns_client.split_rns_name_and_path(rns_address)[0]
                )  # saves to cwd / name
            return f"{Folder.DEFAULT_CACHE_FOLDER}/{uuid.uuid4().hex}"
        else:
            # If no URL provided for a remote file system default to its name if provided
            if rns_address:
                name = rns_address[1:].replace("/", "_") + f".{cls.RESOURCE_TYPE}"
                return f"{Folder.DEFAULT_FOLDER_PATH}/{name}"
            return f"{Folder.DEFAULT_FOLDER_PATH}/{uuid.uuid4().hex}"

    # ----------------------------------
    @staticmethod
    def from_config(config: dict, dryrun=True):
        """Load config values into the object."""
        if config["fs"] == "s3":
            from .s3_folder import S3Folder

            return S3Folder.from_config(config, dryrun=dryrun)
        elif config["fs"] == "gs":
            from .gcs_folder import GCSFolder

            return GCSFolder.from_config(config, dryrun=dryrun)
        elif config["fs"] == "azure":
            from .azure_folder import AzureFolder

            return AzureFolder.from_config(config, dryrun=dryrun)
        elif isinstance(config["fs"], dict):
            from runhouse.rns.hardware import Cluster

            config["fs"] = Cluster.from_config(config["fs"], dryrun=dryrun)
        return Folder(**config, dryrun=dryrun)

    @classmethod
    def from_name(cls, name, dryrun=False):
        config = rns_client.load_config(name=name)
        if not config:
            raise ValueError(f"Resource {name} not found.")

        config["name"] = name

        fs = config["fs"]
        if isinstance(fs, str) and fs.startswith("/"):
            # if the fs is set to a cluster
            cluster_config: dict = rns_client.load_config(name=fs)
            if not cluster_config:
                raise Exception(f"No cluster config saved for {fs}")

            # set the cluster config as the fs
            config["fs"] = cluster_config

        # Uses child class's from_config
        return cls.from_config(config=config, dryrun=dryrun)

    @property
    def url(self):
        if self._url is not None:
            if self.fs == Folder.DEFAULT_FS:
                return str(Path(self._url).expanduser())
            elif self._fs_str == self.CLUSTER_FS and self._url.startswith("~/"):
                # sftp takes relative urls to the home directory but doesn't understand '~'
                return self._url[2:]
            return self._url
        else:
            return None

    @url.setter
    def url(self, url):
        self._url = url
        self._local_mount_path = None

    @property
    def fs(self):
        return self._fs

    @fs.setter
    def fs(self, data_source):
        self._fs = data_source
        self._fsspec_fs = None

    @property
    def data_config(self):
        if isinstance(self.fs, Resource):  # if fs is a cluster
            if not self.fs.address:
                self.fs.populate_vars_from_status(dryrun=False)
                if not self.fs.address:
                    raise ValueError(
                        "Cluster must be started before copying data from it."
                    )
            creds = self.fs.ssh_creds()
            config_creds = {
                "host": self.fs.address,
                "username": creds["ssh_user"],
                # 'key_filename': str(Path(creds['ssh_private_key']).expanduser())}  # For SFTP
                "client_keys": [str(Path(creds["ssh_private_key"]).expanduser())],
            }  # For SSHFS
            ret_config = self._data_config.copy()
            ret_config.update(config_creds)
            return ret_config
        return self._data_config

    @data_config.setter
    def data_config(self, data_config):
        self._data_config = data_config
        self._fsspec_fs = None

    @property
    def _fs_str(self):
        if isinstance(self.fs, Resource):  # if fs is a cluster
            # TODO [DG] Return 'file' if we're on this cluster
            return self.CLUSTER_FS
        else:
            return self.fs

    @property
    def fsspec_fs(self):
        if self._fsspec_fs is None:
            self._fsspec_fs = fsspec.filesystem(self._fs_str, **self.data_config)
        return self._fsspec_fs

    @property
    def local_path(self):
        if self.is_local():
            return self._local_mount_path or str(Path(self.url).expanduser())
        else:
            return None

    def is_writable(self):
        # If the filesystem hasn't overridden mkdirs, it's a no-op and the filesystem is probably readonly
        # (e.g. https://filesystem-spec.readthedocs.io/en/latest/_modules/fsspec/implementations/github.html).
        # In that case, we should just create a new folder in the default
        # location and add it as a child to the parent folder.
        return self.fsspec_fs.__class__.mkdirs == fsspec.AbstractFileSystem.mkdirs

    def mv(self, fs, url=None, data_config=None) -> None:
        """Move the folder to a new filesystem.

        Args:
            fs (str): file system.
            url (:obj:`str`, optional): fsspec URL.
            data_config(:obj:`dict`, optional): Config to move.
        """
        # TODO [DG] create get_default_url for fs method to be shared
        if url is None:
            url = "rh/" + self.rns_address
        data_config = data_config or {}
        with fsspec.open(self.fsspec_url, **self.data_config) as src:
            with fsspec.open(f"{fs}://{url}", **data_config) as dest:
                # NOTE For packages, maybe use the `ignore` param here to only copy python files.
                new_url = shutil.move(src, dest)
        self.url = new_url
        self.fs = fs
        self.data_config = data_config or {}

    def to(self, fs, url: Optional[str] = None, data_config: Optional[dict] = None):
        """Copy the folder to a new filesystem, and return a new Folder object pointing to the new location."""
        # silly syntactic sugar to allow `my_remote_folder.to('here')`, clearer than `to('file')`
        if fs == "here":
            fs = "file"
            url = str(Path.cwd() / self.url.split("/")[-1]) if url is None else url

        url = str(
            url or self.default_url(self.name, fs)
        )  # Make sure it's a string and not a Path

        fs_str = getattr(
            fs, "name", fs
        )  # Use fs.name if available, i.e. fs is a cluster
        logging.info(
            f"Copying folder from {self.fsspec_url} to: {fs_str}, with url: {url}"
        )

        # to_local, to_cluster and to_data_store are also overridden by subclasses to dispatch
        # to more performant cloud-specific APIs
        from runhouse.rns.hardware import Cluster

        if fs == "file":
            return self.to_local(
                dest_url=url, data_config=data_config, return_dest_folder=True
            )
        elif isinstance(fs, Cluster):  # If fs is a cluster
            # TODO [DG] change default behavior to return_dest_folder=False
            return self.to_cluster(dest_cluster=fs, url=url, return_dest_folder=True)
        elif fs in ["s3", "gs", "azure"]:
            return self.to_data_store(
                fs=fs, data_store_url=url, data_config=data_config
            )
        else:
            self.fsspec_copy(fs, url, data_config)
            new_folder = copy.deepcopy(self)
            new_folder.url = url
            new_folder.fs = fs
            new_folder.data_config = data_config or {}
            return new_folder

    def fsspec_copy(self, fs: str, url: str, data_config: dict):
        # Fallback for other fsspec filesystems, but very slow:
        if self.is_local():
            self.fsspec_fs.put(self.url, f"{fs}://{url}", recursive=True)
        else:
            # TODO this is really really slow, maybe use skyplane, as follows:
            # src_url = f'local://{self.url}' if self.is_local() else self.fsspec_url
            # subprocess.run(['skyplane', 'sync', src_url, f'{fs}://{url}'])

            # FYI: from https://github.com/fsspec/filesystem_spec/issues/909
            # TODO [DG]: Copy chunks https://github.com/fsspec/filesystem_spec/issues/909#issuecomment-1204212507
            src = fsspec.get_mapper(self.fsspec_url, create=False, **self.data_config)
            dest = fsspec.get_mapper(f"{fs}://{url}", create=True, **data_config)
            # dest.fs.mkdir(dest.root, create_parents=True)
            import tqdm

            for k in tqdm.tqdm(src):
                # NOTE For packages, maybe use the `ignore` param here to only copy python files.
                dest[k] = src[k]
                # dst.write(src.read())

    def destination_folder(
        self,
        dest_url: str,
        dest_fs: Optional[str] = "file",
        data_config: Optional[dict] = None,
    ):
        """Return a new Folder object pointing to the destination folder."""
        new_folder = copy.deepcopy(self)
        new_folder.url = dest_url
        new_folder.fs = dest_fs
        new_folder.data_config = data_config or {}
        return new_folder

    def to_local(
        self, dest_url: str, data_config: dict, return_dest_folder: bool = False
    ):
        from runhouse.rns.hardware import Cluster

        if self.fs == "file":
            # Simply move the files within local fs
            shutil.copytree(src=self.url, dst=dest_url)
        elif isinstance(self.fs, Cluster):
            return self.from_cluster(cluster=self.fs, dest_url=dest_url)
        else:
            self.fsspec_copy("file", dest_url, data_config)

        if return_dest_folder:
            return self.destination_folder(
                dest_url=dest_url, dest_fs="file", data_config=data_config
            )

    # TODO [DG] Any reason to keep this?
    # def to_sftp(self, url, data_config):
    #     from runhouse.rns.hardware import Cluster
    #     if self.fs == 'file':
    #         # Rsync up the files to the remote fs
    #         self.rsync(local=self.url, remote=url, data_config=data_config, up=True)
    #     elif self.fs == 'sftp':
    #         # Simply move the files within stfp fs
    #         # TODO [DG] speculation
    #         self.fsspec_fs.mv(self.url, url)
    #     elif isinstance(self.fs, Cluster):
    #         self.fs.run([f'rsync {self.url} {data_config["username"]}@{data_config["host"]}:{url} '
    #                      f'--password_file {data_config["key_filename"]}'])
    #     else:
    #         self.fsspec_copy('file', url, data_config)

    def to_data_store(
        self,
        fs: str,
        data_store_url: Optional[str] = None,
        data_config: Optional[dict] = None,
        return_dest_folder: bool = True,
    ):
        """Local or cluster to blob storage"""
        from runhouse.rns.hardware import Cluster

        local_folder_url = self.url

        folder_config = self.config_for_rns
        folder_config["fs"] = fs
        folder_config["url"] = data_store_url
        folder_config["data_config"] = data_config
        new_folder = Folder.from_config(folder_config)

        if self.fs == "file":
            new_folder.upload(src=local_folder_url)
        elif isinstance(self.fs, Cluster):
            self.fs.run(
                [new_folder.upload_command(src=local_folder_url, dest=new_folder.url)]
            )
        else:
            self.fsspec_copy("file", data_store_url, data_config)

        return new_folder

    @staticmethod
    def rsync(local, remote, data_config, up=True):
        # TODO convert this to generate rsync command between two clusters
        dest_str = f'{data_config["username"]}@{data_config["host"]}:{remote}'
        src_str = local
        if not up:
            src_str, dest_str = dest_str, src_str
        subprocess.check_call(
            f"rsync {src_str} {dest_str} "
            f'--password_file {data_config["key_filename"]}'
        )

    def mkdir(self):
        """create the folder in specified file system if it doesn't already exist"""
        folder_url = self.url
        if Path(os.path.basename(folder_url)).suffix != "":
            folder_url = str(Path(folder_url).parent)

        logging.info(f"Creating new {self._fs_str} folder: {folder_url}")
        self.fsspec_fs.mkdirs(folder_url, exist_ok=True)

    def mount(self, url: Optional[str] = None, tmp: bool = False) -> str:
        """Mount the folder locally."""
        # TODO check that fusepy and FUSE are installed
        if tmp:
            self._local_mount_path = tempfile.mkdtemp()
        else:
            self._local_mount_path = url
        remote_fs = self.fsspec_fs
        fsspec.fuse.run(fs=remote_fs, path=self.url, mount_point=self._local_mount_path)
        return self._local_mount_path

    def to_cluster(self, dest_cluster, url=None, mount=False, return_dest_folder=True):
        """Copy the folder from a file or cluster source onto a cluster."""
        if not dest_cluster.address:
            raise ValueError("Cluster must be started before copying data to it.")

        # Create tmp_mount if needed
        if not self.is_local() and mount:
            self.mount(tmp=True)

        dest_url = url or f"~/{Path(self.url).stem}"

        # Need to add slash for rsync to copy the contents of the folder
        dest_folder = copy.deepcopy(self)
        dest_folder.url = dest_url
        dest_folder.fs = dest_cluster
        dest_folder.mkdir()

        if self.fs == "file":
            src_url = self.local_path + "/"
            dest_cluster.rsync(source=src_url, dest=dest_url, up=True, contents=True)

        elif isinstance(self.fs, Resource):
            src_url = dest_url

            cluster_creds = self.fs.ssh_creds()
            creds_file = cluster_creds["ssh_private_key"]

            command = (
                f"rsync -Pavz --filter='dir-merge,- .gitignore' -e \"ssh -i '{creds_file} '"
                f"-o StrictHostKeyChecking=no -o IdentitiesOnly=yes -o ExitOnForwardFailure=yes "
                f"-o ServerAliveInterval=5 -o ServerAliveCountMax=3 -o ConnectTimeout=30s -o ForwardAgent=yes "
                f'-o ControlMaster=auto -o ControlPersist=300s" {src_url}/ {dest_cluster.address}:{dest_url}'
            )
            status_codes = self.fs.run([command])
            if status_codes[0][0] != 0:
                raise Exception(
                    f"Error syncing folder to destination cluster ({dest_cluster.name}). "
                    f"Make sure the source cluster ({self.fs.name}) has the sky keys "
                    f"loaded in path: {creds_file}. "
                    f"For example: `my_cluster.send_secrets(providers=['sky'])`"
                )

        else:
            raise TypeError(
                f"`to_cluster` not supported for filesystem type {type(self.fs)}"
            )

        if return_dest_folder:
            dest_folder.fs = "file"

        return dest_folder

    def from_cluster(self, cluster, dest_url=None):
        """Create a remote folder from a url on a cluster.

        If `dest_url=None`, this will not perform any copy, and simply convert the resource to have a remote
        sftp filesystem into the cluster. If `dest_url` is set, it will rsync down the data and return a folder
        with fs=='file'.

        """
        if dest_url:
            if not cluster.address:
                raise ValueError("Cluster must be started before copying data from it.")
            # TODO support fsspec urls (e.g. nonlocal fs's)?
            Path(dest_url).expanduser().mkdir(parents=True, exist_ok=True)
            cluster.rsync(
                source=self.url,
                dest=str(Path(dest_url).expanduser()),
                up=False,
                contents=True,
            )
            new_folder = copy.deepcopy(self)
            new_folder.url = dest_url
            new_folder.fs = "file"
            # Don't need to do anything with _data_config because cluster creds are injected virtually through the
            # data_config property
            return new_folder
        else:
            new_folder = copy.deepcopy(self)
            new_folder.fs = cluster
            return new_folder

    def is_local(self):
        return (
            self.fs == "file" and self.url is not None and Path(self.url).exists()
        ) or self._local_mount_path

    def share(
        self,
        users: list,
        access_type: Union[ResourceAccess, str] = ResourceAccess.read,
        snapshot: bool = True,
        snapshot_fs: str = None,
        snapshot_compression: str = None,
        snapshot_url: str = None,
    ) -> Tuple[Dict[str, ResourceAccess], Dict[str, ResourceAccess]]:
        """Granting access to the resource for list of users (via their emails). If a user has a Runhouse account they
        will receive an email notifying them of their new access. If the user does not have a Runhouse account they will
        also receive instructions on creating one, after which they will be able to have access to the Resource.
        Note: You can only grant resource access to other users if you have Write / Read privileges for the Resource"""
        if self.is_local() and snapshot:
            # raise ValueError('Cannot share a local resource.')
            fs = snapshot_fs or PROVIDER_FS_LOOKUP[configs.get("default_provider")]
            if fs not in fsspec.available_protocols():
                raise ValueError(
                    f"Invalid mount_fs: {snapshot_fs}. Must be one of {fsspec.available_protocols()}"
                )
            if snapshot_compression not in fsspec.available_compressions():
                raise ValueError(
                    f"Invalid mount_compression: {snapshot_compression}. Must be one of "
                    f"{fsspec.available_compressions()}"
                )
            data_config = (
                {"compression": snapshot_compression} if snapshot_compression else {}
            )
            snapshot_folder = self.to(fs=fs, url=snapshot_url, data_config=data_config)

            # Is this a bad idea? Better to store the snapshot config as the source of truth than the local url
            rns_address = rns_client.local_to_remote_address(self.rns_address)
            snapshot_folder.save(name=rns_address)

            return snapshot_folder.share(
                users=users, access_type=access_type, snapshot=False
            )

        # TODO just call super().share
        if isinstance(access_type, str):
            access_type = ResourceAccess(access_type)
        if not rns_client.exists(self.rns_address):
            self.save(name=rns_client.local_to_remote_address(self.rns_address))
        added_users, new_users = rns_client.grant_resource_access(
            resource_name=self.name, user_emails=users, access_type=access_type
        )
        return added_users, new_users

    def empty_folder(self):
        """Remove folder contents, but not the folder itself."""
        for p in self.fsspec_fs.ls(self.url):
            self.fsspec_fs.rm(p)

    def upload(self, src: str, region: Optional[str] = None):
        """Upload a folder to a remote bucket."""
        raise NotImplementedError

    def upload_command(self, src: str, dest: str):
        """CLI command for uploading folder to remote bucket. Needed when uploading a folder from a cluster."""
        raise NotImplementedError

    def run_upload_cli_cmd(self, sync_dir_command: str, access_denied_message: str):
        """Uploads a folder to a remote bucket.
        Based on the CLI command skypilot uses to upload the folder"""
        from sky.data.data_utils import run_upload_cli

        run_upload_cli(
            command=sync_dir_command,
            access_denied_message=access_denied_message,
            bucket_name=self.bucket_name_from_url(self.url),
        )

    def download(self, dest):
        raise NotImplementedError

    def download_command(self, src, dest):
        """CLI command for downloading folder from remote bucket. Needed when downloading a folder to a cluster."""
        raise NotImplementedError

    @property
    def config_for_rns(self):
        config = super().config_for_rns
        config_attrs = ["local_mount", "data_config"]
        self.save_attrs_to_config(config, config_attrs)

        if self.fs == Folder.DEFAULT_FS:
            # If folder is local check whether path is relative, and if so take it relative to the working director
            # rather than to the home directory. If absolute, it's left alone.
            config["url"] = (
                self._url_relative_to_rh_workdir(self.url) if self.url else None
            )
        else:
            # if not a local filesystem save path as is (e.g. bucket/path)
            config["url"] = self.url

        if isinstance(self.fs, Resource):  # If fs is a cluster
            config["fs"] = self._resource_string_for_subconfig(self.fs)
        else:
            config["fs"] = self.fs

        return config

    @staticmethod
    def _url_relative_to_rh_workdir(url):
        rh_workdir = Path(rns_client.locate_working_dir())
        try:
            return str(Path(url).relative_to(rh_workdir))
        except ValueError:
            return url

    @property
    def fsspec_url(self):
        """Generate the FSSpec URL using the file system and url of the folder"""
        if self.url.startswith("/") and self._fs_str != rns_client.DEFAULT_FS:
            return f"{self._fs_str}:/{self.url}"
        else:
            return f"{self._fs_str}://{self.url}"

    def ls(self, full_paths: bool = True):
        """List the contents of the folder"""
        paths = self.fsspec_fs.ls(path=self.url) if self.url else []
        if full_paths:
            return paths
        else:
            return [Path(path).name for path in paths]

    def resources(self, full_paths: bool = False, resource_type: str = None):
        """List the resources in the *RNS* folder.

        Args:
            full_paths (bool): If True, return the full RNS path for each resource. If False, return the
                resource name only.
            resource_type (str): If provided, only return resources of the specified type.
        """
        # TODO filter by type
        # TODO allow '*' wildcard for listing all resources (and maybe other wildcard things)
        try:
            resources = [
                path for path in self.ls() if (Path(path) / "config.json").exists()
            ]
        except FileNotFoundError:
            return []

        if full_paths:
            return [self.rns_address + "/" + Path(path).stem for path in resources]
        else:
            return [Path(path).stem for path in resources]

    @property
    def rns_address(self):
        """Traverse up the filesystem until reaching one of the directories in rns_base_folders,
        then compute the relative path to that.

        Maybe later, account for folders along the path with a different RNS name."""

        if self.name is None:  # Anonymous folders have no rns address
            return None

        # Only should be necessary when a new base folder is being added (therefore isn't in rns_base_folders yet)
        if self._rns_folder:
            return str(Path(self._rns_folder) / self.name)

        if self.url in rns_client.rns_base_folders.values():
            if self._rns_folder:
                return self._rns_folder + "/" + self.name
            else:
                return rns_client.default_folder + "/" + self.name

        segment = Path(self.url)
        while (
            str(segment) not in rns_client.rns_base_folders.values()
            and not segment == Path.home()
            and not segment == segment.parent
        ):
            segment = segment.parent

        if (
            segment == Path.home() or segment == segment.parent
        ):  # TODO throw an error instead?
            return rns_client.default_folder + "/" + self.name
        else:
            base_folder = Folder(url=str(segment), dryrun=True)
            base_folder_path = base_folder.rns_address
            relative_path = str(Path(self.url).relative_to(base_folder.url))
            return base_folder_path + "/" + relative_path

    def contains(self, name_or_path) -> bool:
        url, fs = self.locate(name_or_path)
        return url is not None

    def locate(self, name_or_path) -> (str, str):
        """Locate the local url of a Folder given an rns path.

        Keep in mind we're using both _rns_ path and physical path logic below. Be careful!
        """

        # If the path is already given relative to the current folder:
        if (Path(self.url) / name_or_path).exists():
            return str(Path(self.url) / name_or_path), self.fs

        # If name or path uses ~/ or ./, need to resolve with folder url
        abs_path = rns_client.resolve_rns_path(name_or_path)
        rns_path = self.rns_address

        # If this folder is anonymous, it has no rns contents
        if rns_path is None:
            return None, None

        if abs_path == rns_path:
            return self.url, self.fs
        try:
            child_url = Path(self.url) / Path(abs_path).relative_to(rns_path)
            if child_url.exists():
                return str(child_url), self.fs
        except ValueError:
            pass

        # Last resort, recursively search inside sub-folders and children.

        segments = abs_path.lstrip("/").split("/")
        if len(segments) == 1:
            return (
                None,
                None,
            )  # If only a single element, would have been found in ls above.

        # Look for lowest folder in the path that exists in filesystem, and recurse from that folder
        greatest_common_folder = Path(self.url)
        i = 0
        for i, seg in enumerate(segments):
            if not (greatest_common_folder / seg).exists():
                break
            greatest_common_folder = greatest_common_folder / seg
        if not str(greatest_common_folder) == self.url:
            return Folder(
                url=str(greatest_common_folder), fs=self.fs, dryrun=True
            ).locate("/".join(segments[i + 1 :]))

        return None, None

    def open(self, name, mode="rb", encoding=None):
        """Returns an fsspec file, which must be used as a content manager to be opened!
        e.g. with my_folder.open('obj_name') as my_file:
                pickle.load(my_file)
        """
        return fsspec.open(
            urlpath=self.fsspec_url + "/" + name,
            mode=mode,
            encoding=encoding,
            **self.data_config,
        )

    def get(self, name, mode="rb", encoding=None):
        """Returns the contents of a file as a string or bytes."""
        with self.open(name, mode=mode, encoding=encoding) as f:
            return f.read()

    # TODO [DG] fix this to follow the correct convention above
    def get_all(self):
        # TODO we're not closing these, do we need to extract file-like objects so we can close them?
        return fsspec.open_files(self.fsspec_url, mode="rb", **self.data_config)

    def exists_in_fs(self):
        return self.fsspec_fs.exists(
            self.fsspec_url
        ) or rh.rns.top_level_rns_fns.exists(self.url)

    def delete_in_fs(self, recursive: bool = True):
        try:
            self.fsspec_fs.rmdir(self.url)
        except Exception as e:
            raise Exception(f"Failed to delete from file system: {e}")

    def rm(self, name, recursive: bool = True):
        """Remove a resource from the folder."""
        try:
            self.fsspec_fs.rm(self.fsspec_url + "/" + name, recursive=recursive)
        except FileNotFoundError:
            pass

    def put(self, contents, overwrite=False):
        """
        files: Either 1) A dict with keys being the file names and values being the
            file-like objects to write, 2) a Resource, or 3) a list of Resources.
        """
        # TODO create the bucket if it doesn't already exist
        # Handle lists of resources just for convenience
        if isinstance(contents, list):
            for resource in contents:
                self.put(resource, overwrite=overwrite)
            return

        if isinstance(contents, Folder):
            if not self.is_writable():
                raise RuntimeError(
                    f"Cannot put files into non-writable folder {self.name or self.url}"
                )
            if contents.url is None:  # Should only be the case when Folder is created
                contents.url = self.url + "/" + contents.name
                contents.fs = self.fs
                # The parent can be anonymous, e.g. the 'rh' folder.
                # TODO not sure if this should be allowed - if parent folder has no rns address, why would child
                # just be put into the default rns folder?
                # TODO If the base is named later, figure out what to do with the contents (rename, resave, etc.).
                if self.rns_address is None:
                    contents.rns_path = rns_client.default_folder + "/" + contents.name
                    rns_client.rns_base_folders.update(
                        {contents.rns_address: contents.url}
                    )
                # We don't need to call .save here to write down because it will be called at the end of the
                # folder or resource constructor
            else:
                if contents.name is None:  # Anonymous resource
                    i = 1
                    new_name = contents.RESOURCE_TYPE + str(i)
                    # Resolve naming conflicts if necessary
                    while rns_client.exists(self.url + "/" + new_name):
                        i += 1
                        new_name = contents.RESOURCE_TYPE + str(i)
                else:
                    new_name = contents.name

                # NOTE For intercloud transfer, we should use Skyplane
                with fsspec.open(
                    self.fsspec_url + "/" + new_name, **self.data_config
                ) as dest:
                    with fsspec.open(
                        contents.fsspec_url, **contents.data_config
                    ) as src:
                        # NOTE For packages, maybe use the `ignore` param here to only copy python files.
                        shutil.move(src, dest)
            return

        if not isinstance(contents, dict):
            raise TypeError(
                "`files` argument to `.put` must be Resource, list of Resources, or dict mapping "
                "filenames to file-like-objects"
            )

        if overwrite is False:
            folder_contents = self.resources()
            intersection = set(folder_contents).intersection(set(contents.keys()))
            if intersection != set():
                raise FileExistsError(
                    f"File(s) {intersection} already exist(s) at url"
                    f"{self.url}, cannot save them without overwriting."
                )
        # TODO figure out default behavior for not overwriting but still saving
        # if not overwrite:
        #     time = datetime.today().strftime('%Y-%m-%d_%H:%M:%S')
        #     self.data_url = self.data_url + time or time
        filenames = list(contents)
        fss_files = fsspec.open_files(
            self.fsspec_url + "/*",
            mode="wb",
            **self.data_config,
            num=len(contents),
            name_function=filenames.__getitem__,
        )
        for (fss_file, raw_file) in zip(fss_files, contents.values()):
            with fss_file as f:
                f.write(raw_file)

    @staticmethod
    def bucket_name_from_url(url: str) -> str:
        """Extract the bucket name from a URL (e.g. '/my-bucket/my-folder/my-file.txt' -> 'my-bucket')"""
        return Path(url).parts[1]


def folder(
    name: Optional[str] = None,
    url: Optional[Union[str, Path]] = None,
    fs: Optional[str] = None,
    dryrun: bool = True,
    local_mount: bool = False,
    data_config: Optional[Dict] = None,
):
    """Returns a folder object, which can be used to interact with the folder at the given url.
    The folder will be saved if `dryrun` is False.
    """
    config = rns_client.load_config(name)
    config["name"] = name or config.get("rns_address", None) or config.get("name")
    config["url"] = url or config.get("url")
    config["local_mount"] = local_mount or config.get("local_mount")
    config["data_config"] = data_config or config.get("data_config")

    file_system = fs or config.get("fs") or Folder.DEFAULT_FS
    config["fs"] = file_system
    if isinstance(file_system, str):
        if file_system in ["file", "github", "sftp", "ssh"]:
            new_folder = Folder.from_config(config, dryrun=dryrun)
        elif file_system == "s3":
            from .s3_folder import S3Folder

            new_folder = S3Folder.from_config(config, dryrun=dryrun)
        elif file_system == "gs":
            from .gcs_folder import GCSFolder

            new_folder = GCSFolder.from_config(config, dryrun=dryrun)
        elif file_system == "azure":
            from .azure_folder import AzureFolder

            new_folder = AzureFolder.from_config(config, dryrun=dryrun)
        elif file_system in fsspec.available_protocols():
            logger.warning(
                f"fsspec file system {file_system} not officially supported. Use at your own risk."
            )
            new_folder = Folder.from_config(config, dryrun=dryrun)
        elif rns_client.exists(file_system, resource_type="cluster"):
            config["fs"] = rns_client.load_config(file_system)
        else:
            raise ValueError(
                f"File system {file_system} not found. Have you installed the "
                f"necessary packages for this fsspec protocol? (e.g. s3fs for s3). If the file system "
                f"is a cluster (ex: /my-user/rh-cpu), make sure the cluster config has been saved."
            )

    # If cluster is passed as the fs.
    if isinstance(config["fs"], dict) or isinstance(
        config["fs"], Resource
    ):  # if fs is a cluster
        new_folder = Folder.from_config(config, dryrun=dryrun)

    return new_folder
