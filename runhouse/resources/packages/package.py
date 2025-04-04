import copy
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Union

from runhouse.constants import INSUFFICIENT_DISK_MSG
from runhouse.exceptions import InsufficientDiskError

from runhouse.globals import obj_store

from runhouse.logger import get_logger
from runhouse.resources.hardware.cluster import Cluster
from runhouse.resources.hardware.utils import (
    _get_cluster_from,
    detect_cuda_version_or_cpu,
)
from runhouse.resources.resource import Resource

from runhouse.utils import (
    conda_env_cmd,
    find_locally_installed_version,
    get_local_install_path,
    install_conda,
    is_python_package_string,
    locate_working_dir,
    run_setup_command,
    split_pip_extras,
    venv_cmd,
)


INSTALL_METHODS = {"local", "pip", "uv", "conda"}

logger = get_logger(__name__)


class CodeSyncError(Exception):
    pass


@dataclass
class InstallTarget:
    local_path: str
    _path_to_sync_to_on_cluster: Optional[str] = None

    @property
    def path_to_sync_to_on_cluster(self) -> str:
        return (
            self._path_to_sync_to_on_cluster
            if self._path_to_sync_to_on_cluster
            else f"~/{Path(self.full_local_path_str()).name}"
        )

    def full_local_path_str(self) -> str:
        return str(Path(self.local_path).expanduser().resolve())

    def __str__(self):
        return f"InstallTarget(local_path={self.local_path}, path_to_sync_to_on_cluster={self._path_to_sync_to_on_cluster})"


class Package(Resource):
    RESOURCE_TYPE = "package"

    # https://pytorch.org/get-started/locally/
    # Note: no binaries exist for 11.4 (https://github.com/pytorch/pytorch/issues/75992)
    TORCH_INDEX_URLS = {
        "11.3": "https://download.pytorch.org/whl/cu113",
        "11.5": "https://download.pytorch.org/whl/cu115",
        "11.6": "https://download.pytorch.org/whl/cu116",
        "11.7": "https://download.pytorch.org/whl/cu117",
        "11.8": "https://download.pytorch.org/whl/cu118",
        "12.1": "https://download.pytorch.org/whl/cu121",
        "cpu": "https://download.pytorch.org/whl/cpu",
    }

    def __init__(
        self,
        name: Optional[str] = None,
        install_method: Optional[str] = None,
        install_target: Optional[Union[str, "Folder"]] = None,
        install_args: Optional[str] = None,
        install_extras: Optional[str] = None,
        preferred_version: Optional[str] = None,
        dryrun: bool = False,
        **kwargs,  # We have this here to ignore extra arguments when calling from from_config
    ):
        """
        Runhouse Package resource.

        .. note::
            To create a git package, please use the factory method :func:`package`.
        """
        super().__init__(
            name=name,
            dryrun=dryrun,
        )
        self.install_method = install_method
        self.install_target = install_target
        self.install_args = install_args
        self.install_extras = install_extras
        self.preferred_version = preferred_version

    def config(self, condensed: bool = True):
        # If the package is just a simple Package.from_string string, no
        # need to store it in rns, just give back the string.
        # if self.install_method in ['pip', 'conda', 'git']:
        #     return f'{self.install_method}:{self.name}'
        config = super().config(condensed)
        config["install_method"] = self.install_method
        config["install_target"] = (
            (
                self.install_target.local_path,
                self.install_target._path_to_sync_to_on_cluster,
            )
            if isinstance(self.install_target, InstallTarget)
            else self.install_target
        )
        config["install_args"] = self.install_args
        config["install_extras"] = self.install_extras
        config["preferred_version"] = self.preferred_version
        return config

    def __str__(self):
        if self.name:
            return f"Package: {self.name}"
        return f"Package: {self.install_target}"

    @staticmethod
    def _prepend_python_executable(
        install_cmd: str,
        conda_env_name: Optional[str] = None,
        venv_path: Optional[str] = None,
        cluster: "Cluster" = None,
    ):
        if venv_path:
            return install_cmd
        if cluster or conda_env_name:
            return f"python3 -m {install_cmd}"
        return f"{sys.executable} -m {install_cmd}"

    @staticmethod
    def _prepend_env_command(
        install_cmd: str,
        conda_env_name: Optional[str] = None,
        venv_path: Optional[str] = None,
    ):
        if conda_env_name:
            install_cmd = conda_env_cmd(cmd=install_cmd, conda_env_name=conda_env_name)
        if venv_path:
            install_cmd = venv_cmd(cmd=install_cmd, venv_path=venv_path)

        return install_cmd

    def _validate_folder_path(self):
        # If self.path is the same as the user's home directory, raise an error.
        # Check this with Path and expanduser to handle both relative and absolute paths.
        if isinstance(
            self.install_target, InstallTarget
        ) and self.install_target.full_local_path_str() in [
            str(Path("~").expanduser()),
            str(Path("/")),
        ]:
            raise CodeSyncError(
                "Cannot sync the home directory. Please include a Python configuration file in a subdirectory."
            )

    def _install_cmd_with_extras(self, install_target: str, install_extras: str):
        install_specific_version_of_target = "==" in install_target
        if install_specific_version_of_target:
            split_target = install_target.split(
                "=="
            )  # split the target package name from its version
            return f'"{split_target[0]}{install_extras}=={split_target[1]}"'
        return f'"{self.install_target}{install_extras}"'

    def _pip_install_cmd(
        self,
        conda_env_name: Optional[str] = None,
        venv_path: Optional[str] = None,
        cluster: "Cluster" = None,
        uv: bool = None,
    ):
        if uv is None:
            uv = self.install_method == "uv"
        install_args = f" {self.install_args}" if self.install_args else ""
        install_extras = f"[{self.install_extras}]" if self.install_extras else ""
        if isinstance(self.install_target, InstallTarget):
            if cluster:
                install_cmd = (
                    self.install_target.path_to_sync_to_on_cluster
                    + install_extras
                    + install_args
                )
            else:
                install_cmd = (
                    self.install_target.full_local_path_str()
                    + install_extras
                    + install_args
                )
        else:
            install_target = (
                f'"{self.install_target}"'
                if not install_extras
                else self._install_cmd_with_extras(
                    install_target=self.install_target, install_extras=install_extras
                )
            )
            install_cmd = install_target + install_args

        install_cmd = f"pip install {self._install_cmd_for_torch(install_cmd, cluster)}"
        if uv:
            install_cmd = f"uv {install_cmd}"
        else:
            # uv doesn't need python executable
            install_cmd = self._prepend_python_executable(
                install_cmd,
                cluster=cluster,
                conda_env_name=conda_env_name,
                venv_path=venv_path,
            )
        install_cmd = self._prepend_env_command(
            install_cmd, conda_env_name=conda_env_name, venv_path=venv_path
        )
        return install_cmd

    def _conda_install_cmd(
        self, conda_env_name: Optional[str] = None, cluster: "Cluster" = None
    ):
        install_args = f" {self.install_args}" if self.install_args else ""
        if isinstance(self.install_target, InstallTarget):
            install_cmd = f"{self.install_target.local_path}" + install_args
        else:
            install_cmd = self.install_target + install_args

        install_cmd = f"conda install -y {install_cmd}"
        install_cmd = self._prepend_env_command(
            install_cmd, conda_env_name=conda_env_name
        )
        install_conda(cluster=cluster)
        return install_cmd

    def _install_and_validate_output(
        self, install_cmd: str, cluster: "Cluster" = None, node: Optional[str] = None
    ):
        install_res = run_setup_command(install_cmd, cluster=cluster, node=node)
        retcode = install_res[0]
        if retcode != 0:
            stdout, stderr = install_res[1], install_res[2]
            if INSUFFICIENT_DISK_MSG in stdout or INSUFFICIENT_DISK_MSG in stderr:
                raise InsufficientDiskError(command=install_cmd)
            raise RuntimeError(
                f"{self.install_method} install '{install_cmd}' failed, check that the package exists and is available for your platform."
            )

    def _install(
        self,
        cluster: "Cluster" = None,
        node: Optional[str] = None,
        conda_env_name: Optional[str] = None,
        venv_path: Optional[str] = None,
        override_remote_version: bool = False,
    ):
        """Install package.

        Args:
            cluster (Optional[Cluster]): If provided, will install package on cluster using SSH. Otherwise, the
                assumption is that we are installing locally. (Default: ``None``)
            node (Optional[str]): Node on the cluster to install the package on, if using SSH. If ``cluster`` is
                provided without a ``node``, package will be installed on the head node. (Default: ``None``)
            conda_env_name (Optional[str]): Name of the conda environment to install the package on, if using SSH and
                installing in a specific conda env that is not activated by default.
            override_remote_version (bool, optional): If the package exists both locally and remotely, whether to override
                the remote version with the local version. By default, the local version will be installed only if
                the package does not already exist on the cluster. (Default: ``False``)
        """
        logger.info(f"Installing {str(self)} with method {self.install_method}.")

        if self.install_method in ["pip", "uv"]:

            # If this is a generic pip package, with no version pinned, we want to check if there is a version
            # already installed. If there is, and ``override_remote_version`` is not set to ``True``, then we ignore
            # preferred version and leave the existing version.  The user can also force a version install by
            # doing `numpy==2.0.0`. Else, we install the preferred version, that matches their local.
            if is_python_package_string(self.install_target):
                exists_remotely = (
                    run_setup_command(
                        f"[ -d ~/{self.install_target} ]",
                        cluster=cluster,
                        conda_env_name=conda_env_name,
                        venv_path=venv_path,
                        node=node,
                        stream_logs=False,
                    )[0]
                    == 0
                )
                if exists_remotely:
                    self.install_target = InstallTarget(
                        local_path=self.install_target,
                        _path_to_sync_to_on_cluster=f"~/{self.install_target}",
                    )
                elif self.preferred_version is not None:
                    if not override_remote_version:
                        # Check if this is installed
                        retcode = run_setup_command(
                            f"python3 -c \"import importlib.util; exit(0) if importlib.util.find_spec('{self.install_target}') else exit(1)\"",
                            cluster=cluster,
                            conda_env_name=conda_env_name,
                            venv_path=venv_path,
                            node=node,
                        )[0]

                        if retcode == 0:
                            logger.info(
                                f"{self.install_target} already installed. Skipping."
                            )
                            return
                        else:
                            override_remote_version = True
                    if override_remote_version:
                        self.install_target = (
                            f"{self.install_target}=={self.preferred_version}"
                        )

            install_cmd = self._pip_install_cmd(
                conda_env_name=conda_env_name,
                venv_path=venv_path,
                cluster=cluster,
                uv=(self.install_method == "uv"),
            )
            logger.info(f"Running via install_method pip: {install_cmd}")
            self._install_and_validate_output(
                install_cmd=install_cmd, cluster=cluster, node=node
            )

        elif self.install_method == "conda":
            install_cmd = self._conda_install_cmd(
                conda_env_name=conda_env_name, cluster=cluster
            )
            logger.info(f"Running via install_method conda: {install_cmd}")
            self._install_and_validate_output(
                install_cmd=install_cmd, cluster=cluster, node=node
            )

        else:
            if self.install_method != "local":
                raise ValueError(
                    f"Unknown install method {self.install_method}. Must be one of {INSTALL_METHODS}"
                )

        # Need to append to path
        if self.install_method == "local":
            if isinstance(self.install_target, InstallTarget):
                obj_store.add_sys_path_to_all_processes(
                    self.install_target.full_local_path_str()
                ) if not cluster else run_setup_command(
                    f'export PATH="$PATH;{self.install_target.path_to_sync_to_on_cluster}"',
                    cluster=cluster,
                    venv_path=venv_path,
                    node=node,
                )
            elif not cluster:
                if Path(self.install_target).resolve().expanduser().exists():
                    obj_store.add_sys_path_to_all_processes(
                        str(Path(self.install_target).resolve().expanduser())
                    )
                else:
                    raise ValueError(
                        f"install_target {self.install_target} must be a Folder or a path to a directory for install_method {self.install_method}"
                    )
            else:
                raise ValueError(
                    f"If cluster is provided, install_target must be a Folder for install_method {self.install_method}"
                )

    # ----------------------------------
    # Torch Install Helpers
    # ----------------------------------
    def _reqs_install_cmd_for_torch(
        self, reqs_path, reqs_list, install_args="", cluster=None
    ):
        """Read requirements from file, append --index-url and --extra-index-url where relevant for torch packages,
        and return list of formatted packages."""
        # if torch extra index url is already defined by the user or torch isn't a req, directly pip install reqs file
        if not any("torch" in req for req in reqs_list):
            return f"-r {reqs_path}" + install_args

        for req in reqs_list:
            if (
                "--index-url" in req or "--extra-index-url" in req
            ) and "pytorch.org" in req:
                return f"-r {reqs_path}" + install_args

        # add extra-index-url for torch if not found
        cuda_version_or_cpu = detect_cuda_version_or_cpu(cluster=cluster)
        return f"-r {reqs_path} --extra-index-url {self._torch_index_url(cuda_version_or_cpu)}"

    def _install_cmd_for_torch(self, install_cmd, cluster=None):
        """Return the correct formatted pip install command for the torch package(s) provided."""
        if install_cmd.startswith("#"):
            return None

        torch_source_packages = ["torch", "torchvision", "torchaudio"]
        if not any([x in install_cmd for x in torch_source_packages]):
            return install_cmd
        if "+" in install_cmd or "--extra-index-url" in install_cmd:
            return install_cmd

        packages_to_install: list = self._packages_to_install_from_cmd(install_cmd)
        final_install_cmd = ""
        cuda_version_or_cpu = detect_cuda_version_or_cpu(cluster=cluster)
        for package_install_cmd in packages_to_install:
            formatted_cmd = self._install_url_for_torch_package(
                package_install_cmd, cuda_version_or_cpu
            )
            if formatted_cmd:
                final_install_cmd += formatted_cmd + " "

        final_install_cmd = final_install_cmd.rstrip()
        return final_install_cmd if final_install_cmd != "" else None

    def _install_url_for_torch_package(self, install_cmd, cuda_version_or_cpu):
        """Build the full install command, adding a --index-url and --extra-index-url where applicable."""
        # Grab the relevant index url for torch based on the CUDA version provided
        if "," in install_cmd:
            # If installing a range of versions format the string to make it compatible with `pip_install` method
            install_cmd = install_cmd.replace(" ", "")

        index_url = self._torch_index_url(cuda_version_or_cpu)
        if index_url and not any(
            specifier in install_cmd for specifier in ["--index-url ", "-i "]
        ):
            install_cmd = f"{install_cmd} --index-url {index_url}"

        if "--extra-index-url" not in install_cmd:
            return f"{install_cmd} --extra-index-url https://pypi.python.org/simple/"

        return install_cmd

    def _torch_index_url(self, cuda_version_or_cpu: str):
        return self.TORCH_INDEX_URLS.get(cuda_version_or_cpu)

    @staticmethod
    def _packages_to_install_from_cmd(install_cmd: str):
        """Split a string of command(s) into a list of separate commands"""
        # Remove any --extra-index-url flags from the install command (to be added later by default)
        install_cmd = re.sub(r"--extra-index-url\s+\S+", "", install_cmd)
        install_cmd = install_cmd.strip()

        if ", " in install_cmd:
            # Ex: 'torch>=1.13.0,<2.0.0'
            return [install_cmd]

        matches = re.findall(r"(\S+(?:\s+(-i|--index-url)\s+\S+)?)", install_cmd)

        packages_to_install = [match[0] for match in matches]
        return packages_to_install

    def to(
        self,
        system: Union[str, Dict, "Cluster"],
    ):
        """Copy the package onto filesystem or cluster, and return the new Package object.

        Args:
            system (str, Dict, or Cluster): Cluster to send the package to.
        """
        if not isinstance(self.install_target, InstallTarget):
            return self

        system = _get_cluster_from(system)
        if isinstance(system, Cluster) and system.on_this_cluster():
            return self

        self._validate_folder_path()
        if isinstance(system, Cluster):
            system.rsync(
                source=str(self.install_target.full_local_path_str()),
                dest=str(self.install_target.path_to_sync_to_on_cluster),
                up=True,
                contents=True,
                node="all",
                parallel=True,
            )

            new_package = copy.copy(self)
            new_package.install_target = InstallTarget(
                local_path=self.install_target.path_to_sync_to_on_cluster,
                _path_to_sync_to_on_cluster=self.install_target.path_to_sync_to_on_cluster,
            )
            return new_package

        return self

    @staticmethod
    def split_req_install_method(req_str: str):
        """Split a requirements string into a install method and the rest of the string."""
        splat = req_str.split(":", 1)
        return (splat[0], splat[1]) if len(splat) > 1 else ("", splat[0])

    @staticmethod
    def from_config(config: Dict, dryrun: bool = False, _resolve_children: bool = True):
        if isinstance(config.get("install_target"), (tuple, list)):
            config["install_target"] = InstallTarget(
                local_path=config["install_target"][0],
                _path_to_sync_to_on_cluster=config["install_target"][1],
            )
        return Package(**config, dryrun=dryrun)

    @staticmethod
    def from_string(specifier: str, dryrun: bool = False):
        install_method, target_and_args = Package.split_req_install_method(specifier)

        # Handles a case like "torch --index-url https://download.pytorch.org/whl/cu113"
        target, args = (
            target_and_args.split(" ", 1)
            if " " in target_and_args
            else (target_and_args, "")
        )
        target, extras = split_pip_extras(target_and_args)

        # If the target is a path
        # If it doesn't have a /, we're assuming it's a pip installable thing first and foremost
        if "/" in target or "~" in target or install_method == "local":
            # We need to do this because relative paths are relative to the current working directory!
            abs_target = (
                Path(target).expanduser()
                if Path(target).expanduser().is_absolute()
                else Path(locate_working_dir()) / target
            )
            if abs_target.exists():
                target = InstallTarget(
                    local_path=str(abs_target), _path_to_sync_to_on_cluster=None
                )

        # If install method is not provided, default to pip
        if not install_method:
            if Path(specifier).expanduser().resolve().exists():
                install_method = "local"
            else:
                install_method = "pip"

        # Attempt to use the same version of the package installed locally, if pip_install (pip)
        # or a sync_package (local)
        preferred_version = None
        if is_python_package_string(target):
            locally_installed_version = find_locally_installed_version(target)
            if locally_installed_version:
                # Check if this is a package that was installed from local
                local_install_path = get_local_install_path(target)
                if local_install_path and Path(local_install_path).exists():
                    install_method = install_method or "local"
                    if install_method == "local":
                        target = InstallTarget(
                            local_path=local_install_path,
                            _path_to_sync_to_on_cluster=None,
                        )
                else:
                    # We want to preferrably install this version of the package server-side
                    install_method == install_method or "pip"
                    if install_method in ["pip", "uv"]:
                        preferred_version = locally_installed_version

        # If install method is still not set, default to pip
        install_method = install_method or "pip"

        # "Local" install method is a special case where we just copy a local folder and add to path
        if install_method == "local":
            return Package(
                install_target=target, install_method=install_method, dryrun=dryrun
            )
        elif install_method in ["pip", "uv", "conda"]:
            return Package(
                install_target=target,
                install_args=args,
                install_method=install_method,
                install_extras=extras,
                preferred_version=preferred_version,
                dryrun=dryrun,
            )
        else:
            raise ValueError(
                f"Unknown install method {install_method}. Must be one of {INSTALL_METHODS}"
            )


def package(
    name: str = None,
    install_method: str = None,
    install_str: str = None,
    path: str = None,
    system: str = None,
    load_from_den: bool = True,
    dryrun: bool = False,
) -> Package:
    """
    Builds an instance of :class:`Package`.

    Args:
        name (str, optional): Name to assign the package resource.
        install_method (str, optional): Method for installing the package.
            Options: [``pip``, ``conda``, ``local``]
        install_str (str, optional): Additional arguments to install.
        path (str, optional): URL of the package to install.
        system (str, optional): File system or cluster on which the package lives.
            Currently this must a cluster or one of: [``file``, ``s3``, ``gs``].
        load_from_den (bool, optional): Whether to try loading the Package from Den. (Default: ``True``)
        dryrun (bool, optional): Whether to create the Package if it doesn't exist, or load the Package
            object as a dryrun. (Default: ``False``)

    Returns:
        Package: The resulting package.

    Example:
        >>> reloaded_package = rh.package(name="my-package")
        >>> local_package = rh.package(path="local/folder/path", install_method="local")
    """
    if name and not any([install_method, install_str, path, system]):
        # If only the name is provided and dryrun is set to True
        return Package.from_name(name, load_from_den=load_from_den, dryrun=dryrun)

    install_target = None
    install_args = None
    if path is not None:
        install_target = (path, None)
        install_args = install_str
    elif install_str is not None:
        install_target, install_args = install_str.split(" ", 1)

    return Package(
        install_method=install_method,
        install_target=install_target,
        install_args=install_args,
        name=name,
        dryrun=dryrun,
    )
