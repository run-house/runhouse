import copy
import logging
import os
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, Dict, Optional, Union

import fsspec

import sshfs

from runhouse.globals import rns_client
from runhouse.resources.hardware import _current_cluster, _get_cluster_from, Cluster
from runhouse.resources.resource import Resource
from runhouse.rns.top_level_rns_fns import exists
from runhouse.rns.utils.api import generate_uuid

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
    DEFAULT_FOLDER_PATH = "/runhouse-folder"
    DEFAULT_CACHE_FOLDER = "~/.cache/runhouse"

    def __init__(
        self,
        name: Optional[str] = None,
        path: Optional[str] = None,
        system: Union[str, Cluster] = None,
        dryrun: bool = False,
        local_mount: bool = False,
        data_config: Optional[Dict] = None,
        **kwargs,  # We have this here to ignore extra arguments when calling from from_config
    ):
        """
        Runhouse Folder object.

        .. note::
            To build a folder, please use the factory method :func:`folder`.
        """
        super().__init__(name=name, dryrun=dryrun)

        self._system = None
        self._fsspec_fs = None
        self._fsspec_fs_str = None

        current_cluster_config = _current_cluster(key="config")
        if current_cluster_config and system is None:
            self.system = Cluster.from_config(current_cluster_config)
        elif isinstance(system, dict):
            self.system = Cluster.from_config(system)
        else:
            self.system = system or self.DEFAULT_FS

        # TODO [DG] Should we ever be allowing this to be None?
        self._path = (
            self.default_path(self.rns_address, system)
            if path is None
            else path
            if system != "file"
            else path
            if Path(path).expanduser().is_absolute()
            else str(Path(rns_client.locate_working_dir()) / path)
        )
        self.data_config = data_config or {}

        self.local_mount = local_mount
        self._local_mount_path = None
        if local_mount:
            self.mount(tmp=True)

    def __getstate__(self):
        """Override the pickle method to clear _fsspec_fs before pickling."""
        state = self.__dict__.copy()
        state["_fsspec_fs"] = None
        return state

    @classmethod
    def default_path(cls, rns_address, system):
        name = (
            rns_client.split_rns_name_and_path(rns_address)[0]
            if rns_address
            else generate_uuid()
        )

        if system == Folder.DEFAULT_FS:
            return str(Path.cwd() / name)
        elif isinstance(system, Cluster):
            return f"{Folder.DEFAULT_CACHE_FOLDER}/{name}"
        else:
            return f"{Folder.DEFAULT_FOLDER_PATH}/{name}"

    # ----------------------------------
    @staticmethod
    def from_config(config: dict, dryrun=False, _resolve_children=True):
        if _resolve_children:
            config = Folder._check_for_child_configs(config)

        """Load config values into the object."""
        if config["system"] == "s3":
            from .s3_folder import S3Folder

            return S3Folder.from_config(config, dryrun=dryrun)
        elif config["system"] == "gs":
            from .gcs_folder import GCSFolder

            return GCSFolder.from_config(config, dryrun=dryrun)
        elif config["system"] == "azure":
            from .azure_folder import AzureFolder

            return AzureFolder.from_config(config, dryrun=dryrun)
        elif isinstance(config["system"], dict):
            config["system"] = Cluster.from_config(config["system"], dryrun=dryrun)
        return Folder(**config, dryrun=dryrun)

    @classmethod
    def _check_for_child_configs(cls, config: dict):
        """Overload by child resources to load any resources they hold internally."""
        system = config.get("system")
        if isinstance(system, str) or isinstance(system, dict):
            config["system"] = _get_cluster_from(system)
        return config

    @property
    def path(self):
        if self._path is not None:
            if self.system == Folder.DEFAULT_FS:
                return str(Path(self._path).expanduser())
            elif self._fs_str == self.CLUSTER_FS and self._path.startswith("~/"):
                # sftp takes relative paths to the home directory but doesn't understand '~'
                return str(self._path[2:])
            return str(self._path)
        else:
            return None

    @path.setter
    def path(self, path):
        self._path = path
        self._local_mount_path = None

    @property
    def system(self):
        return self._system

    @system.setter
    def system(self, data_source):
        self._system = data_source
        self._fsspec_fs = None

    # Maybe figure out how to free sshfs properly (https://github.com/ronf/asyncssh/issues/112)
    # def __del__(self):
    #     if self.local_mount:
    #         self.unmount()
    #     if self._fsspec_fs and hasattr(self._fsspec_fs, "close"):
    #         self._fsspec_fs.close()

    @property
    def data_config(self):
        if isinstance(self.system, Resource):  # if system is a cluster
            # handle case cluster is itself
            if self.system.on_this_cluster():
                return self._data_config

            if not self.system.address:
                self.system._update_from_sky_status(dryrun=False)
                if not self.system.address:
                    raise ValueError(
                        "Cluster must be started before copying data from it."
                    )
            creds = self.system.creds_values

            client_keys = (
                [str(Path(creds["ssh_private_key"]).expanduser())]
                if creds.get("ssh_private_key")
                else []
            )
            password = creds.get("password", None)
            config_creds = {
                "host": creds.get("ssh_host") or self.system.address,
                "username": creds.get("ssh_user"),
                # 'key_filename': str(Path(creds['ssh_private_key']).expanduser())}  # For SFTP
                "client_keys": client_keys,  # For SSHFS
                "password": password,
                "connect_timeout": "3s",
                "proxy_command": creds.get("ssh_proxy_command"),
            }
            ret_config = self._data_config.copy()
            ret_config.update(config_creds)
            if creds and self.system.ssh_port:
                ret_config["port"] = self.system.ssh_port
            return ret_config
        return self._data_config

    @data_config.setter
    def data_config(self, data_config):
        self._data_config = data_config
        self._fsspec_fs = None

    @property
    def _fs_str(self):
        if isinstance(self.system, Resource):  # if system is a cluster
            if self.system.on_this_cluster():
                return self.DEFAULT_FS
            return self.CLUSTER_FS
        else:
            return self.system

    @property
    def fsspec_fs(self):
        if self._fsspec_fs_str != self._fs_str or self._fsspec_fs is None:
            self._fsspec_fs_str = self._fs_str
            self._fsspec_fs = fsspec.filesystem(self._fsspec_fs_str, **self.data_config)
        return self._fsspec_fs

    @property
    def local_path(self):
        if self.is_local():
            return self._local_mount_path or str(Path(self.path).expanduser())
        else:
            return None

    def is_writable(self):
        """Whether the folder is writable.

        Example:
            >>> if my_folder.is_writable():
            >>>     ....
        """
        # If the filesystem hasn't overridden mkdirs, it's a no-op and the filesystem is probably readonly
        # (e.g. https://filesystem-spec.readthedocs.io/en/latest/_modules/fsspec/implementations/github.html).
        # In that case, we should just create a new folder in the default
        # location and add it as a child to the parent folder.
        return self.fsspec_fs.__class__.mkdirs == fsspec.AbstractFileSystem.mkdirs

    def mv(
        self, system, path: Optional[str] = None, data_config: Optional[dict] = None
    ) -> None:
        """Move the folder to a new filesystem or cluster.

        Example:
            >>> folder = rh.folder(path="local/path")
            >>> folder.mv(my_cluster)
            >>> folder.mv("s3", "s3_bucket/path")
        """
        # TODO [DG] use _generate_default_path
        if path is None:
            path = "rh/" + self.rns_address
        data_config = data_config or {}
        with fsspec.open(self.fsspec_url, **self.data_config) as src:
            with fsspec.open(f"{system}://{path}", **data_config) as dest:
                # NOTE For packages, maybe use the `ignore` param here to only copy python files.
                new_path = shutil.move(src, dest)
        self.path = new_path
        self.system = system
        self.data_config = data_config or {}

    def to(
        self,
        system: Union[str, "Cluster"],
        path: Optional[Union[str, Path]] = None,
        data_config: Optional[dict] = None,
    ):
        """Copy the folder to a new filesystem, and return a new Folder object pointing to the new location."""
        if system == "here":
            current_cluster_config = _current_cluster(key="config")
            if current_cluster_config:
                system = Cluster.from_config(current_cluster_config)
            else:
                system = "file"
            path = str(Path.cwd() / self.path.split("/")[-1]) if path is None else path

        path = str(
            path or self.default_path(self.rns_address, system)
        )  # Make sure it's a string and not a Path

        system_str = getattr(
            system, "name", system
        )  # Use system.name if available, i.e. system is a cluster
        logger.info(
            f"Copying folder from {self.fsspec_url} to: {system_str}, with path: {path}"
        )

        # to_local, to_cluster and to_data_store are also overridden by subclasses to dispatch
        # to more performant cloud-specific APIs
        system = _get_cluster_from(system)

        if system == "file":
            return self._to_local(dest_path=path, data_config=data_config)
        elif isinstance(system, Cluster):  # If system is a cluster
            return self._to_cluster(dest_cluster=system, path=path)
        elif system in ["s3", "gs", "azure"]:
            return self._to_data_store(
                system=system, data_store_path=path, data_config=data_config
            )
        else:
            self._fsspec_copy(system, path, data_config)
            new_folder = copy.deepcopy(self)
            new_folder.path = path
            new_folder.system = system
            new_folder.data_config = data_config or {}
            return new_folder

    def _fsspec_copy(self, system: str, path: str, data_config: dict):
        """Copy the fsspec filesystem to the given new filesystem and path."""
        # Fallback for other fsspec filesystems, but very slow:
        system = system or Folder.DEFAULT_FS
        if self.is_local():
            self.fsspec_fs.put(self.path, f"{system}://{path}", recursive=True)
        else:
            # This is really really slow, maybe use skyplane, as follows:
            # src_url = f'local://{self.path}' if self.is_local() else self.fsspec_url
            # subprocess.run(['skyplane', 'sync', src_url, f'{system}://{path}'])

            # FYI: from https://github.com/fsspec/filesystem_spec/issues/909
            # Maybe copy chunks https://github.com/fsspec/filesystem_spec/issues/909#issuecomment-1204212507
            src = fsspec.get_mapper(self.fsspec_url, create=False, **self.data_config)
            dest = fsspec.get_mapper(f"{system}://{path}", create=True, **data_config)
            # dest.system.mkdir(dest.root, create_parents=True)
            import tqdm

            for k in tqdm.tqdm(src):
                # NOTE For packages, maybe use the `ignore` param here to only copy python files.
                dest[k] = src[k]
                # dst.write(src.read())

    def destination_folder(
        self,
        dest_path: str,
        dest_system: Optional[str] = "file",
        data_config: Optional[dict] = None,
    ):
        """Returns a new Folder object pointing to the destination folder."""
        new_folder = copy.deepcopy(self)
        new_folder.path = dest_path
        new_folder.system = dest_system
        new_folder.data_config = data_config or {}
        return new_folder

    def _to_local(self, dest_path: str, data_config: dict):
        """Copies folder to local."""
        if (
            self._fs_str == "file"
        ):  # Also covers the case where we're on the cluster at system
            # Simply move the files within local system
            shutil.copytree(src=self.path, dst=dest_path)
        elif isinstance(self.system, Cluster):
            return self._cluster_to_local(cluster=self.system, dest_path=dest_path)
        else:
            self._fsspec_copy("file", dest_path, data_config)

        return self.destination_folder(
            dest_path=dest_path, dest_system="file", data_config=data_config
        )

    def _to_data_store(
        self,
        system: str,
        data_store_path: Optional[str] = None,
        data_config: Optional[dict] = None,
    ):
        """Local or cluster to blob storage."""
        local_folder_path = self.path

        folder_config = self.config()
        folder_config["system"] = system
        folder_config["path"] = data_store_path
        folder_config["data_config"] = data_config
        new_folder = Folder.from_config(folder_config)

        if (
            self._fs_str == "file"
        ):  # Also covers the case where we're on the cluster at system
            new_folder._upload(src=local_folder_path)
        elif isinstance(self.system, Cluster):
            self.system.run(
                [
                    new_folder._upload_command(
                        src=local_folder_path, dest=new_folder.path
                    )
                ]
            )
        else:
            self._fsspec_copy("file", data_store_path, data_config)

        return new_folder

    @staticmethod
    def rsync(local, remote, data_config, up=True):
        """Rsync local folder to remote."""
        dest_str = f'{data_config["username"]}@{data_config["host"]}:{remote}'
        src_str = local
        if not up:
            src_str, dest_str = dest_str, src_str
        cmd = (
            f'rsync {src_str} {dest_str} --password_file {data_config["key_filename"]}'
        )
        subprocess.run(shlex.split(cmd), check=True)

    def mkdir(self):
        """Create the folder in specified file system if it doesn't already exist."""
        folder_path = self.path
        if Path(os.path.basename(folder_path)).suffix != "":
            folder_path = str(Path(folder_path).parent)

        logger.info(
            f"Creating new {self._fs_str} folder if it does not already exist in path: {folder_path}"
        )
        self.fsspec_fs.mkdirs(folder_path, exist_ok=True)

        return self

    def mount(self, path: Optional[str] = None, tmp: bool = False) -> str:
        """Mount the folder locally.

        Example:
            remote_folder = rh.folder("folder/path", system="s3")
            local_mount = remote_folder.mount()
        """
        # TODO check that fusepy and FUSE are installed
        if tmp:
            self._local_mount_path = tempfile.mkdtemp()
        else:
            self._local_mount_path = path
        remote_fs = self.fsspec_fs
        fsspec.fuse.run(
            fs=remote_fs, path=self.path, mount_point=self._local_mount_path
        )
        return self._local_mount_path

    def _to_cluster(self, dest_cluster, path=None, mount=False):
        """Copy the folder from a file or cluster source onto a destination cluster."""
        if not dest_cluster.address:
            raise ValueError("Cluster must be started before copying data to it.")

        # Create tmp_mount if needed
        if not self.is_local() and mount:
            self.mount(tmp=True)

        dest_path = path or f"~/{Path(self.path).name}"

        # Need to add slash for rsync to copy the contents of the folder
        dest_folder = copy.deepcopy(self)
        dest_folder.path = dest_path
        dest_folder.system = dest_cluster

        if self._fs_str == "file" and dest_cluster.name is not None:
            # Includes case where we're on the cluster itself
            # And the destination is a cluster, not rh.here
            dest_cluster._rsync(
                source=self.path, dest=dest_path, up=True, contents=True
            )

        elif isinstance(self.system, Resource):
            if self.system.endpoint(external=False) == dest_cluster.endpoint(
                external=False
            ):
                # We're on the same cluster, so we can just move the files
                if not path:
                    # If user didn't specify a path, we can just return self
                    return self
                else:
                    dest_cluster.run(
                        [f"mkdir -p {dest_path}", f"cp -r {self.path}/* {dest_path}"],
                    )
            else:
                self._cluster_to_cluster(dest_cluster, dest_path)

        else:
            # data store folders have their own specific _to_cluster functions
            raise TypeError(
                f"`Sending from filesystem type {type(self.system)} is not supported"
            )

        return dest_folder

    def _cluster_to_cluster(self, dest_cluster, dest_path):
        src_path = self.path

        cluster_creds = self.system.creds_values

        if not cluster_creds.get("password") and not dest_cluster.creds_values.get(
            "password"
        ):
            creds_file = cluster_creds.get("ssh_private_key")
            creds_cmd = f"-i '{creds_file}' " if creds_file else ""

            dest_cluster.run([f"mkdir -p {dest_path}"])
            command = (
                f"rsync -Pavz --filter='dir-merge,- .gitignore' -e \"ssh {creds_cmd}"
                f"-o StrictHostKeyChecking=no -o IdentitiesOnly=yes -o ExitOnForwardFailure=yes "
                f"-o ServerAliveInterval=5 -o ServerAliveCountMax=3 -o ConnectTimeout=30s -o ForwardAgent=yes "
                f'-o ControlMaster=auto -o ControlPersist=300s" {src_path}/ {dest_cluster.address}:{dest_path}'
            )
            status_codes = self.system.run([command])
            if status_codes[0][0] != 0:
                raise Exception(
                    f"Error syncing folder to destination cluster ({dest_cluster.name}). "
                    f"Make sure the source cluster ({self.system.name}) has the necessary provider keys "
                    f"if applicable. "
                )
        else:
            local_folder = self._cluster_to_local(self.system, self.path)
            local_folder._to_cluster(dest_cluster, dest_path)

    def _cluster_to_local(self, cluster, dest_path):
        """Create a local folder with dest_path from the cluster.

        This function rsyncs down the data and return a folder with system=='file'.
        """
        if not cluster.address:
            raise ValueError("Cluster must be started before copying data from it.")
        Path(dest_path).expanduser().mkdir(parents=True, exist_ok=True)
        cluster._rsync(
            source=self.path,
            dest=str(Path(dest_path).expanduser()),
            up=False,
            contents=True,
        )
        new_folder = copy.deepcopy(self)
        new_folder.path = dest_path
        new_folder.system = "file"
        # Don't need to do anything with _data_config because cluster creds are injected virtually through the
        # data_config property
        return new_folder

    def is_local(self):
        """Whether the folder is on the local filesystem.

        Example:
            >>> is_local = my_folder.is_local()
        """
        return (
            self._fs_str == "file"
            and self.path is not None
            and Path(self.path).expanduser().exists()
        ) or self._local_mount_path

    def _upload(self, src: str, region: Optional[str] = None):
        """Upload a folder to a remote bucket."""
        raise NotImplementedError

    def _upload_command(self, src: str, dest: str):
        """CLI command for uploading folder to remote bucket. Needed when uploading a folder from a cluster."""
        raise NotImplementedError

    def _run_upload_cli_cmd(self, command: str):
        """Uploads a folder to a remote bucket.
        Based on the CLI command skypilot uses to upload the folder"""
        # Adapted from: https://github.com/skypilot-org/skypilot/blob/983f5fa3197fe7c4b5a28be240f7b027f7192b15/sky/data/data_utils.py#L165 # noqa
        with subprocess.Popen(
            command, stderr=subprocess.PIPE, stdout=subprocess.DEVNULL, shell=True
        ) as process:
            stderr = []
            while True:
                line = process.stderr.readline()
                if not line:
                    break
                str_line = line.decode("utf-8")
                stderr.append(str_line)
            returncode = process.wait()
            if returncode != 0:
                raise RuntimeError(" ".join(stderr).strip())

    def _download(self, dest):
        raise NotImplementedError

    def _download_command(self, src, dest):
        """CLI command for downloading folder from remote bucket. Needed when downloading a folder to a cluster."""
        raise NotImplementedError

    def config(self, condensed=True):
        config = super().config(condensed)
        config_attrs = ["local_mount", "data_config"]
        self.save_attrs_to_config(config, config_attrs)

        if self.system == Folder.DEFAULT_FS:
            # If folder is local check whether path is relative, and if so take it relative to the working director
            # rather than to the home directory. If absolute, it's left alone.
            config["path"] = (
                self._path_relative_to_rh_workdir(self.path) if self.path else None
            )
        else:
            # if not a local filesystem save path as is (e.g. bucket/path)
            config["path"] = self.path

        if isinstance(self.system, Resource):  # If system is a cluster
            config["system"] = self._resource_string_for_subconfig(
                self.system, condensed
            )
        else:
            config["system"] = self.system

        return config

    def _save_sub_resources(self, folder: str = None):
        if isinstance(self.system, Resource):
            self.system.save(folder=folder)

    @staticmethod
    def _path_relative_to_rh_workdir(path):
        rh_workdir = Path(rns_client.locate_working_dir())
        try:
            return str(Path(path).relative_to(rh_workdir))
        except ValueError:
            return path

    @property
    def fsspec_url(self):
        """Generate the FSSpec URL using the file system and path of the folder"""
        if self.path.startswith("/") and self._fs_str not in [
            rns_client.DEFAULT_FS,
            self.CLUSTER_FS,
        ]:
            return f"{self._fs_str}:/{self.path}"
        else:
            # For local, ssh / sftp filesystems we need both slashes
            # e.g.: 'ssh:///home/ubuntu/.cache/runhouse/tables/dede71ef83ce45ffa8cb27d746f97ee8'
            return f"{self._fs_str}://{self.path}"

    def ls(self, full_paths: bool = True, sort: bool = False) -> list:
        """List the contents of the folder.

        Args:
            full_paths (Optional[bool]): Whether to list the full paths of the folder contents.
                Defaults to ``True``.
            sort (Optional[bool]): Whether to sort the folder contents by time modified.
                Defaults to ``False``.
        """
        paths = self.fsspec_fs.ls(path=self.path) if self.path else []
        if sort:
            paths = sorted(
                paths, key=lambda f: self.fsspec_fs.info(f)["mtime"], reverse=True
            )
        if full_paths:
            return paths
        else:
            return [Path(path).name for path in paths]

    def resources(self, full_paths: bool = False):
        """List the resources in the *RNS* folder.

        Example:
            >>> resources = my_folder.resources()
        """
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
        """
        # TODO Maybe later, account for folders along the path with a different RNS name.

        if self.name is None:  # Anonymous folders have no rns address
            return None

        # Only should be necessary when a new base folder is being added (therefore isn't in rns_base_folders yet)
        if self._rns_folder:
            return str(Path(self._rns_folder) / self.name)

        if self.path in rns_client.rns_base_folders.values():
            if self._rns_folder:
                return self._rns_folder + "/" + self.name
            else:
                return rns_client.default_folder + "/" + self.name

        segment = Path(self.path)
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
            base_folder = Folder(path=str(segment), dryrun=True)
            base_folder_path = base_folder.rns_address
            relative_path = str(Path(self.path).relative_to(base_folder.path))
            return base_folder_path + "/" + relative_path

    def contains(self, name_or_path) -> bool:
        """Whether path of a Folder exists locally.

        Example:
            >>> my_folder = rh.folder("local/folder/path")
            >>> in_folder = my_folder.contains("filename")
        """
        path, _ = self.locate(name_or_path)
        return path is not None

    def locate(self, name_or_path) -> (str, str):
        """Locate the local path of a Folder given an rns path.

        Example:
            >>> my_folder = rh.folder("local/folder/path")
            >>> local_path = my_folder.locate("file_name")
        """
        # Note: Keep in mind we're using both _rns_ path and physical path logic below. Be careful!

        # If the path is already given relative to the current folder:
        if (Path(self.path) / name_or_path).exists():
            return str(Path(self.path) / name_or_path), self.system

        # If name or path uses ~/ or ./, need to resolve with folder path
        abs_path = rns_client.resolve_rns_path(name_or_path)
        rns_path = self.rns_address

        # If this folder is anonymous, it has no rns contents
        if rns_path is None:
            return None, None

        if abs_path == rns_path:
            return self.path, self.system
        try:
            child_path = Path(self.path) / Path(abs_path).relative_to(rns_path)
            if child_path.exists():
                return str(child_path), self.system
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
        greatest_common_folder = Path(self.path)
        i = 0
        for i, seg in enumerate(segments):
            if not (greatest_common_folder / seg).exists():
                break
            greatest_common_folder = greatest_common_folder / seg
        if not str(greatest_common_folder) == self.path:
            return Folder(
                path=str(greatest_common_folder), system=self.system, dryrun=True
            ).locate("/".join(segments[i + 1 :]))

        return None, None

    def open(self, name, mode="rb", encoding=None):
        """Returns an fsspec file, which must be used as a content manager to be opened.

        Example:
            >>> with my_folder.open('obj_name') as my_file:
            >>>        pickle.load(my_file)
        """
        return self.fsspec_fs.open(self.path + "/" + name, mode=mode, encoding=encoding)

    def get(self, name, mode="rb", encoding=None):
        """Returns the contents of a file as a string or bytes.

        Example:
            >>> contents = my_folder.get(file_name)
        """
        with self.open(name, mode=mode, encoding=encoding) as f:
            return f.read()

    # TODO [DG] fix this to follow the correct convention above
    def get_all(self):
        # TODO add docs for this
        # TODO we're not closing these, do we need to extract file-like objects so we can close them?
        return fsspec.open_files(self.fsspec_url, mode="rb", **self.data_config)

    def exists_in_system(self):
        """Whether the folder exists in the filesystem.

        Example:
            >>> exists_on_system = my_folder.exists_in_system()
        """
        return self.fsspec_fs.exists(self.path) or exists(self.path)

    def rm(self, contents: list = None, recursive: bool = True):
        """Delete a folder from the file system. Optionally provide a list of folder contents to delete.

        Args:
            contents (Optional[List]): Specific contents to delete in the folder.
            recursive (bool): Delete the folder itself (including all its contents).
                Defaults to ``True``.

        Example:
            >>> my_folder.rm()
        """
        if not contents:
            try:
                self.fsspec_fs.rm(self.path, recursive=recursive)
            except FileNotFoundError:
                pass

        else:
            for file_name in contents:
                try:
                    self.fsspec_fs.rm(f"{self.path}/{file_name}")
                except FileNotFoundError:
                    pass

    def put(
        self, contents, overwrite=False, mode: str = "wb", write_fn: Callable = None
    ):
        """Put given contents in folder.

        Args:
            contents (Dict[str, Any] or Resource or List[Resource]): Contents to put in folder.
                Must be a dict with keys being the file names (without full paths) and values being the file-like
                objects to write, or a Resource object, or a list of Resources.
            overwrite (bool): Whether to dump the file contents as json. By default expects data to be encoded.
                Defaults to ``False``.
            mode (Optional(str)): Write mode to use for fsspec. Defaults to ``wb``.
            write_fn (Optional(Callable)): Function to use for writing file contents.
                Example: ``write_fn = lambda f, data: json.dump(data, f)

        Example:
            >>> my_folder.put(contents={"filename.txt": data})
        """
        self.mkdir()
        # Handle lists of resources just for convenience
        if isinstance(contents, list):
            for resource in contents:
                self.put(resource, overwrite=overwrite)
            return

        if isinstance(contents, Folder):
            if not self.is_writable():
                raise RuntimeError(
                    f"Cannot put files into non-writable folder {self.name or self.path}"
                )
            if contents.path is None:  # Should only be the case when Folder is created
                contents.path = self.path + "/" + contents.name
                contents.system = self.system
                # The parent can be anonymous, e.g. the 'rh' folder.
                # TODO not sure if this should be allowed - if parent folder has no rns address, why would child
                # just be put into the default rns folder?
                # TODO If the base is named later, figure out what to do with the contents (rename, resave, etc.).
                if self.rns_address is None:
                    contents.rns_path = rns_client.default_folder + "/" + contents.name
                    rns_client.rns_base_folders.update(
                        {contents.rns_address: contents.path}
                    )
                # We don't need to call .save here to write down because it will be called at the end of the
                # folder or resource constructor
            else:
                if contents.name is None:  # Anonymous resource
                    i = 1
                    new_name = contents.RESOURCE_TYPE + str(i)
                    # Resolve naming conflicts if necessary
                    while rns_client.exists(self.path + "/" + new_name):
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
                    f"File(s) {intersection} already exist(s) at path"
                    f"{self.path}, cannot save them without overwriting."
                )
        # TODO figure out default behavior for not overwriting but still saving
        # if not overwrite:
        #     time = datetime.today().strftime('%Y-%m-%d_%H:%M:%S')
        #     self.data_url = self.data_url + time or time
        filenames = list(contents)
        fss_files = fsspec.open_files(
            self.fsspec_url + "/*",
            mode=mode,
            **self.data_config,
            num=len(contents),
            name_function=filenames.__getitem__,
        )
        for (fss_file, raw_file) in zip(fss_files, contents.values()):
            with fss_file as f:
                if write_fn is not None:
                    write_fn(raw_file, f)
                else:
                    f.write(raw_file)

    @staticmethod
    def _bucket_name_from_path(path: str) -> str:
        """Extract the bucket name from a path (e.g. '/my-bucket/my-folder/my-file.txt' -> 'my-bucket')"""
        return Path(path).parts[1]
