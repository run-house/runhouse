import copy
import os
import pickle
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

from runhouse.globals import rns_client

from runhouse.logger import logger
from runhouse.resources.hardware import _current_cluster, _get_cluster_from, Cluster
from runhouse.resources.module import Module
from runhouse.resources.resource import Resource
from runhouse.rns.utils.api import generate_uuid
from runhouse.utils import locate_working_dir


class Folder(Module):
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
        super().__init__(name=name, dryrun=dryrun, system=system)

        self._filesystem = None

        # TODO [DG] Should we ever be allowing this to be None?
        if path is None:
            self._path = self.default_path(self.rns_address, system)
        else:
            if system != "file":
                self._path = path
            else:
                self._path = self._path_absolute_to_rh_workdir(path)
        self.data_config = data_config or {}

        self.local_mount = local_mount
        self._local_mount_path = None
        if local_mount:
            self.mount(tmp=True)

    def __getstate__(self):
        """Override the pickle method to clear _filesystem before pickling."""
        state = self.__dict__.copy()
        state["_filesystem"] = None
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
            config["system"] = Cluster.from_config(
                config["system"], dryrun=dryrun, _resolve_children=_resolve_children
            )
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
            elif self._fs_str == self.CLUSTER_FS and str(self._path).startswith("~/"):
                # sftp takes relative paths to the home directory but doesn't understand '~'
                return str(self._path[2:])
            return str(self._path)
        else:
            return None

    @path.setter
    def path(self, path):
        self._path = path
        self._local_mount_path = None

    # TODO [JL] we can probably kill this entirely
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
        self._filesystem = None

    @property
    def _fs_str(self):
        if isinstance(self.system, Resource):  # if system is a cluster
            if self.system.on_this_cluster():
                return self.DEFAULT_FS
            return self.CLUSTER_FS
        else:
            return self.system

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
        test_file_path = Path(self.path) / "writability_test_file.txt"
        try:
            with open(test_file_path, "w") as test_file:
                test_file.write("")
            test_file_path.unlink()  # Delete the test file
            return True

        except IOError:
            return False

    def mv(
        self, system, path: Optional[str] = None, data_config: Optional[dict] = None
    ) -> None:
        """Move the folder to a new filesystem or cluster.

        Example:
            >>> folder = rh.folder(path="local/path")
            >>> folder.mv(my_cluster)
            >>> folder.mv("s3", "s3_bucket/path")
        """
        if path is None:
            raise ValueError("A destination path must be specified.")

        dest_path = Path(path).expanduser()
        src_path = Path(self.path).expanduser()

        if not src_path.exists():
            raise FileNotFoundError(f"The source path {src_path} does not exist.")

        if system == self.DEFAULT_FS:
            if dest_path.exists():
                raise FileExistsError(
                    f"The destination path {dest_path} already exists."
                )

            # Create the destination directory if it doesn't exist
            dest_path.parent.mkdir(parents=True, exist_ok=True)

            # Move the directory
            shutil.move(str(src_path), str(dest_path))

            # Update the path attribute
            self.path = str(dest_path)
            self.system = "file"
            self.data_config = data_config or {}

        else:
            # TODO [JL] support moving to other systems
            raise NotImplementedError(f"System {system} not supported for local mv")

    def to(
        self,
        system: Union[str, "Cluster"],
        path: Optional[Union[str, Path]] = None,
        data_config: Optional[dict] = None,
    ):
        """Copy the folder to a new filesystem.
        Currently supported: ``here``, ``file``, ``gs``, ``s3``, or a cluster.

        Example:
            >>> local_folder = rh.folder(path="/my/local/folder")
            >>> s3_folder = local_folder.to("s3")
        """
        if system == "here":
            current_cluster_config = _current_cluster(key="config")
            if current_cluster_config:
                system = Cluster.from_config(current_cluster_config)
            else:
                system = "file"
            path = str(Path.cwd() / self.path.split("/")[-1]) if path is None else path

        if isinstance(system, Cluster):
            # Make sure the top level directory exists on the cluster before creating the module on the cluster

            if self.path.startswith("/") or self.path.startswith("~"):
                relative_path = os.path.relpath(self.path, str(Path.home()))
                path = f"~/{relative_path}"

            # rsync the folder contents to the cluster
            self._to_cluster(system, path=path)

            # Note: setting `force_install` to ensure the module gets installed the cluster
            # the folder's system may already be a cluster, which would skip the install
            return super().to(system=system, force_install=True)

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
        elif system in ["s3", "gs"]:
            return self._to_data_store(
                system=system, data_store_path=path, data_config=data_config
            )
        else:
            raise ValueError(
                f"System '{system}' not currently supported as a destination system."
            )

    def _fsspec_copy(self, system: str, path: str, data_config: dict):
        """Copy the fsspec filesystem to the given new filesystem and path."""
        raise NotImplementedError

    def _destination_folder(
        self,
        dest_path: str,
        dest_system: Optional[str] = "file",
        data_config: Optional[dict] = None,
    ):
        """Returns a new Folder object pointing to the destination folder."""
        folder_config = self.config()
        folder_config["system"] = dest_system
        folder_config["path"] = dest_path
        folder_config["data_config"] = data_config
        new_folder = Folder.from_config(folder_config)

        return new_folder

    def _to_local(self, dest_path: str, data_config: dict):
        """Copies folder to local. Only relevant for the base Folder if its system is a cluster."""
        if isinstance(self.system, Cluster):
            # Cluster --> local copying
            logger.debug(
                f"Copying folder from cluster {self.system.name} to local path: {dest_path}"
            )
            self._cluster_to_local(self.system, dest_path)
            return self

        if self.system == self.DEFAULT_FS:
            # Local --> local copying
            logger.debug(f"Copying folder to local path: {dest_path}")
            self.mv(system=self.system, path=dest_path, data_config=data_config)
            return self

        raise TypeError(f"Cannot copy from {self.system} to local.")

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
        path = Path(self.path)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Folder created in path: {self.path}")

    def mount(self, path: Optional[str] = None, tmp: bool = False) -> str:
        """Mount the folder locally.

        Example:
            remote_folder = rh.folder("folder/path", system="s3")
            local_mount = remote_folder.mount()
        """
        if tmp:
            local_mount_path = tempfile.mkdtemp()
        else:
            local_mount_path = path or os.path.join(
                tempfile.gettempdir(), "local_mount"
            )

        if not os.path.exists(local_mount_path):
            os.makedirs(local_mount_path)

        # Copy the contents to the local directory
        src_path = Path(self.path)
        dest_path = Path(local_mount_path)

        if src_path.is_dir():
            shutil.copytree(src_path, dest_path, dirs_exist_ok=True)
        else:
            shutil.copy2(src_path, dest_path)

        return local_mount_path

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

    def _upload_folder_to_bucket(self, command: str):
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
        rh_workdir = Path(locate_working_dir())
        try:
            return str(Path(path).relative_to(rh_workdir))
        except ValueError:
            return path

    @staticmethod
    def _path_absolute_to_rh_workdir(path):
        return (
            path
            if Path(path).expanduser().is_absolute()
            else str(Path(locate_working_dir()) / path)
        )

    @property
    def fsspec_url(self):
        """Generate the FSSpec style URL using the file system and path of the folder"""
        if self.path.startswith("/") and self._fs_str not in [
            rns_client.DEFAULT_FS,
            self.CLUSTER_FS,
        ]:
            return f"{self._fs_str}:/{self.path}"
        else:
            # For local, ssh / sftp filesystems we need both slashes
            # e.g.: 'ssh:///home/ubuntu/.cache/runhouse/tables/dede71ef83ce45ffa8cb27d746f97ee8'
            return f"{self._fs_str}://{self.path}"

    @property
    def _bucket_name(self):
        return self.path.lstrip("/").split("/")[0]

    @property
    def _key(self):
        filtered_parts = self.path.split("/")[2:]
        return "/".join(filtered_parts) + "/"

    def ls(self, full_paths: bool = True, sort: bool = False) -> List:
        """List the contents of the folder.

        Args:
            full_paths (Optional[bool]): Whether to list the full paths of the folder contents.
                Defaults to ``True``.
            sort (Optional[bool]): Whether to sort the folder contents by time modified.
                Defaults to ``False``.
        """
        path = Path(self.path).expanduser()
        paths = [p for p in path.iterdir()]

        # Sort the paths by modification time if sort is True
        if sort:
            paths.sort(key=lambda p: p.stat().st_mtime, reverse=True)

        # Convert paths to strings and format them based on full_paths
        if full_paths:
            return [str(p.resolve()) for p in paths]
        else:
            return [p.name for p in paths]

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
        """Returns the specified file as a stream (`botocore.response.StreamingBody`), which must be used as a
        content manager to be opened.

        Example:
            >>> with my_folder.open('obj_name') as my_file:
            >>>        pickle.load(my_file)
        """
        file_path = Path(self.path) / name
        valid_modes = {"r", "w", "a", "rb", "wb", "ab", "r+", "w+", "a+"}

        if mode not in valid_modes:
            raise NotImplementedError(
                f"{mode} mode is not implemented yet for local files"
            )

        return open(file_path, mode=mode, encoding=encoding)

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
        raise NotImplementedError

    def exists_in_system(self):
        """Whether the folder exists in the filesystem.

        Example:
            >>> exists_on_system = my_folder.exists_in_system()
        """
        return Path(self.path).exists() and Path(self.path).is_dir()

    def rm(self, contents: list = None, recursive: bool = True):
        """Delete a folder from the file system. Optionally provide a list of folder contents to delete.

        Args:
            contents (Optional[List]): Specific contents to delete in the folder.
            recursive (bool): Delete the folder itself (including all its contents).
                Defaults to ``True``.

        Example:
            >>> my_folder.rm()
        """
        folder_path = Path(self.path)

        if contents:
            for content in contents:
                content_path = folder_path / content
                if content_path.exists():
                    if content_path.is_file():
                        content_path.unlink()
                    elif content_path.is_dir() and recursive:
                        shutil.rmtree(content_path)
                    else:
                        raise ValueError(
                            f"Path {content_path} is a directory and recursive is set to False"
                        )
        else:
            if recursive:
                shutil.rmtree(folder_path)
            else:
                if folder_path.is_dir():
                    for item in folder_path.iterdir():
                        if item.is_file():
                            item.unlink()
                        else:
                            raise ValueError(
                                f"Folder {item} found in {folder_path}, recursive is set to False"
                            )
                else:
                    folder_path.unlink()

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
                Example: ``write_fn = lambda f, data: json.dump(data, f)``

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
                    f"Cannot put files into non-writable folder {self.path}"
                )
            if contents.path is None:  # Should only be the case when Folder is created
                contents.path = os.path.join(self.path, contents.name)
            return

        if not isinstance(contents, dict):
            raise TypeError(
                "`contents` argument must be a dict mapping filenames to file-like objects"
            )

        if overwrite is False:
            # Check if files exist and raise an error if they do
            existing_files = set(os.listdir(self.path))
            intersection = existing_files.intersection(set(contents.keys()))
            if intersection:
                raise FileExistsError(
                    f"File(s) {intersection} already exist(s) at path: {self.path}. "
                    f"Cannot save them with overwrite={overwrite}."
                )

        for filename, file_obj in contents.items():
            file_obj = self._serialize_file_obj(file_obj)
            file_path = Path(self.path) / filename
            if not overwrite and file_path.exists():
                raise FileExistsError(f"File {file_path} already exists.")

            try:
                with open(file_path, mode) as f:
                    if write_fn:
                        write_fn(f, file_obj)
                    else:
                        f.write(file_obj)

            except Exception as e:
                raise RuntimeError(f"Failed to write {filename} to {file_path}: {e}")

    @staticmethod
    def _serialize_file_obj(file_obj):
        if not isinstance(file_obj, bytes):
            try:
                file_obj = pickle.dumps(file_obj)
            except (pickle.PicklingError, TypeError) as e:
                raise ValueError(f"Cannot serialize file contents: {e}")

        return file_obj

    @staticmethod
    def _bucket_name_from_path(path: str) -> str:
        """Extract the bucket name from a path (e.g. '/my-bucket/my-folder/my-file.txt' -> 'my-bucket')"""
        return Path(path).parts[1]
