import copy
import re
import sys
from pathlib import Path
from typing import Dict, Optional, Union

from runhouse.resources.envs.utils import install_conda, run_setup_command
from runhouse.resources.folders import Folder, folder
from runhouse.resources.hardware.cluster import Cluster
from runhouse.resources.hardware.utils import (
    _get_cluster_from,
    detect_cuda_version_or_cpu,
)
from runhouse.resources.resource import Resource
from runhouse.utils import (
    find_locally_installed_version,
    get_local_install_path,
    is_python_package_string,
    locate_working_dir,
)


INSTALL_METHODS = {"local", "reqs", "pip", "conda", "rh"}

from runhouse.logger import logger


class CodeSyncError(Exception):
    pass


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
        name: str = None,
        install_method: str = None,
        install_target: Union[str, Folder] = None,
        install_args: str = None,
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

    def config(self, condensed=True):
        # If the package is just a simple Package.from_string string, no
        # need to store it in rns, just give back the string.
        # if self.install_method in ['pip', 'conda', 'git']:
        #     return f'{self.install_method}:{self.name}'
        config = super().config(condensed)
        config["install_method"] = self.install_method
        config["install_target"] = self._resource_string_for_subconfig(
            self.install_target, condensed
        )
        config["install_args"] = self.install_args
        return config

    def __str__(self):
        if self.name:
            return f"Package: {self.name}"
        if isinstance(self.install_target, Folder):
            # if self.install_target.name:
            #     return f'Package: {self.install_target.name}'
            return f"Package: {self.install_target.path}"
        return f"Package: {self.install_target}"

    @staticmethod
    def _prepend_python_executable(
        install_cmd: str, env: Union[str, "Env"] = None, cluster: "Cluster" = None
    ):
        return (
            f"python3 -m {install_cmd}"
            if cluster or env
            else f"{sys.executable} -m {install_cmd}"
        )

    @staticmethod
    def _prepend_env_command(install_cmd: str, env: Union[str, "Env"] = None):
        if env:
            from runhouse.resources.envs.utils import _get_env_from

            env = _get_env_from(env)
            install_cmd = env._full_command(install_cmd)

        return install_cmd

    def _validate_folder_path(self):
        # If self.path is the same as the user's home directory, raise an error.
        # Check this with Path and expanduser to handle both relative and absolute paths.
        if Path(self.install_target.path).expanduser() in [
            Path("~").expanduser(),
            Path("/"),
        ]:
            raise CodeSyncError(
                "Cannot sync the home directory. Please include a Python configuration file in a subdirectory."
            )

    def _pip_install_cmd(
        self, env: Union[str, "Env"] = None, cluster: "Cluster" = None
    ):
        install_args = f" {self.install_args}" if self.install_args else ""
        if isinstance(self.install_target, Folder):
            install_cmd = (
                f"{str(Path(self.install_target.local_path).absolute())}" + install_args
            )
        else:
            install_target = f'"{self.install_target}"'
            install_cmd = install_target + install_args

        install_cmd = f"pip install {self._install_cmd_for_torch(install_cmd, cluster)}"
        install_cmd = self._prepend_python_executable(
            install_cmd, cluster=cluster, env=env
        )
        install_cmd = self._prepend_env_command(install_cmd, env=env)
        return install_cmd

    def _conda_install_cmd(
        self, env: Union[str, "Env"] = None, cluster: "Cluster" = None
    ):
        install_args = f" {self.install_args}" if self.install_args else ""
        if isinstance(self.install_target, Folder):
            install_cmd = f"{self.install_target.local_path}" + install_args
        else:
            install_cmd = self.install_target + install_args

        install_cmd = f"conda install -y {install_cmd}"
        install_cmd = self._prepend_env_command(install_cmd, env=env)
        install_conda(cluster=cluster)
        return install_cmd

    def _reqs_install_cmd(
        self, env: Union[str, "Env"] = None, cluster: "Cluster" = None
    ):
        install_args = f" {self.install_args}" if self.install_args else ""
        if not isinstance(self.install_target, Folder):
            install_cmd = self.install_target + install_args
        else:
            path = self.install_target.local_path

            # Read reqs_path and reqs from local if not cluster
            if not cluster:
                reqs_path = f"{path}/requirements.txt"
                if not Path(reqs_path).expanduser().exists():
                    return None

                with open(reqs_path) as f:
                    reqs_list = f.readlines()

            # Otherwise, make sure the target folder is on the cluster,
            # and read reqs from the cluster
            else:
                if not self.install_target.system == cluster:
                    install_target = self.to(cluster).install_target
                else:
                    install_target = self.install_target

                if not install_target.exists_in_system():
                    return None
                elif "requirements.txt" not in install_target.ls(full_paths=False):
                    return None

                reqs_list = (
                    install_target.get("requirements.txt", mode="r")
                    .strip("\n")
                    .split("\n")
                )
                reqs_path = f"{install_target.path}/requirements.txt"

            install_cmd = self._reqs_install_cmd_for_torch(
                reqs_path, reqs_list, install_args, cluster=cluster
            )

        install_cmd = f"pip install {install_cmd}"
        install_cmd = self._prepend_python_executable(
            install_cmd, env=env, cluster=cluster
        )
        install_cmd = self._prepend_env_command(install_cmd, env=env)
        return install_cmd

    def _install(self, env: Union[str, "Env"] = None, cluster: "Cluster" = None):
        """Install package.

        Args:
            env (Env or str): Environment to install package on. If left empty, defaults to base environment.
                (Default: ``None``)
            cluster (Optional[Cluster]): If provided, will install package on cluster using SSH.
        """

        logger.info(f"Installing {str(self)} with method {self.install_method}.")

        if isinstance(self.install_target, Folder):
            if not cluster:
                path = self.install_target.local_path
            elif self.install_target.exists_in_system():
                path = self.install_target.path
            else:
                path = self.to(cluster).install_target.path

            if not path:
                return

        if self.install_method == "pip":
            install_cmd = self._pip_install_cmd(env=env, cluster=cluster)
            logger.info(f"Running via install_method pip: {install_cmd}")
            retcode = run_setup_command(install_cmd, cluster=cluster)[0]
            if retcode != 0:
                raise RuntimeError(
                    f"Pip install {install_cmd} failed, check that the package exists and is available for your platform."
                )

        elif self.install_method == "conda":
            install_cmd = self._conda_install_cmd(env=env, cluster=cluster)
            logger.info(f"Running via install_method conda: {install_cmd}")
            retcode = run_setup_command(install_cmd, cluster=cluster)[0]
            if retcode != 0:
                raise RuntimeError(
                    f"Conda install {install_cmd} failed, check that the package exists and is "
                    "available for your platform."
                )

        elif self.install_method == "reqs":
            install_cmd = self._reqs_install_cmd(env=env, cluster=cluster)
            if install_cmd:
                logger.info(f"Running via install_method reqs: {install_cmd}")
                retcode = run_setup_command(install_cmd, cluster=cluster)[0]
                if retcode != 0:
                    raise RuntimeError(
                        f"Reqs install {install_cmd} failed, check that the package exists and is available for your platform."
                    )
            else:
                logger.info(f"{path}/requirements.txt not found, skipping reqs install")

        else:
            if self.install_method != "local":
                raise ValueError(
                    f"Unknown install method {self.install_method}. Must be one of {INSTALL_METHODS}"
                )

        # Need to append to path
        if self.install_method in ["local", "reqs"]:
            if isinstance(self.install_target, Folder):
                sys.path.insert(0, path) if not cluster else run_setup_command(
                    f"export PATH=$PATH;{path}", cluster=cluster
                )
            elif not cluster:
                if Path(self.install_target).resolve().expanduser().exists():
                    sys.path.insert(
                        0, str(Path(self.install_target).resolve().expanduser())
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

        cuda_version_or_cpu = detect_cuda_version_or_cpu(cluster=cluster)
        for req in reqs_list:
            if (
                "--index-url" in req or "--extra-index-url" in req
            ) and "pytorch.org" in req:
                return f"-r {reqs_path}" + install_args

        # add extra-index-url for torch if not found
        return f"-r {reqs_path} --extra-index-url {self._torch_index_url(cuda_version_or_cpu)}"

    def _install_cmd_for_torch(self, install_cmd, cluster=None):
        """Return the correct formatted pip install command for the torch package(s) provided."""
        if install_cmd.startswith("#"):
            return None

        torch_source_packages = ["torch", "torchvision", "torchaudio"]
        if not any([x in install_cmd for x in torch_source_packages]):
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
        path: Optional[str] = None,
        mount: bool = False,
    ):
        """Copy the package onto filesystem or cluster, and return the new Package object."""
        if not isinstance(self.install_target, Folder):
            raise TypeError(
                "`install_target` must be a Folder in order to copy the package to a system."
            )

        if (
            isinstance(self.install_target.system, str)
            and not self.install_target.system == "file"
        ):
            self.install_target.system = _get_cluster_from(self.install_target.system)

        install_system_name = (
            self.install_target.system.name
            if isinstance(self.install_target.system, Cluster)
            else self.install_target.system
        )
        system = _get_cluster_from(system)
        system_name = system.name if isinstance(system, Cluster) else system

        if system_name == install_system_name:
            return self

        if isinstance(system, Resource):
            if isinstance(self.install_target.system, Resource):
                # if these are both clusters, check if they're pointing to the same address.
                # We use endpoint instead of address here because if address is localhost, we need to port too
                if self.install_target.system.endpoint(
                    external=False
                ) == system.endpoint(external=False):
                    # If we're on the target system, just make sure the package is in the Python path
                    sys.path.insert(0, self.install_target.local_path)
                    return self
            logger.info(
                f"Copying package from {self.install_target.fsspec_url} to: {getattr(system, 'name', system)}"
            )
            self._validate_folder_path()
            new_folder = self.install_target._to_cluster(system, path=path, mount=mount)
        else:  # to fs
            self._validate_folder_path()
            new_folder = self.install_target.to(system, path=path)
        new_folder.system = system
        new_package = copy.copy(self)
        new_package.install_target = new_folder
        return new_package

    @staticmethod
    def split_req_install_method(req_str: str):
        """Split a requirements string into a install method and the rest of the string."""
        splat = req_str.split(":", 1)
        return (splat[0], splat[1]) if len(splat) > 1 else ("", splat[0])

    @staticmethod
    def from_config(config: dict, dryrun=False, _resolve_children=True):
        if isinstance(config.get("install_target"), dict):
            config["install_target"] = Folder.from_config(
                config["install_target"],
                dryrun=dryrun,
                _resolve_children=_resolve_children,
            )
        if config.get("resource_subtype") == "GitPackage":
            from runhouse import GitPackage

            return GitPackage.from_config(
                config, dryrun=dryrun, _resolve_children=_resolve_children
            )
        return Package(**config, dryrun=dryrun)

    @staticmethod
    def from_string(specifier: str, dryrun=False):
        if specifier == "requirements.txt":
            specifier = "reqs:./"

        # Use regex to check if specifier matches '<method>:https://github.com/<path>' or 'https://github.com/<path>'
        match = re.search(
            r"^(?:(?P<method>[^:]+):)?(?P<path>https://github.com/.+)", specifier
        )
        if match:
            install_method = match.group("method")
            url = match.group("path")
            from runhouse.resources.packages.git_package import git_package

            return git_package(
                git_url=url, install_method=install_method, dryrun=dryrun
            )

        install_method, target_and_args = Package.split_req_install_method(specifier)

        # Handles a case like "torch --index-url https://download.pytorch.org/whl/cu113"
        rel_target, args = (
            target_and_args.split(" ", 1)
            if " " in target_and_args
            else (target_and_args, "")
        )

        # We need to do this because relative paths are relative to the current working directory!
        abs_target = (
            Path(rel_target).expanduser()
            if Path(rel_target).expanduser().is_absolute()
            else Path(locate_working_dir()) / rel_target
        )
        if abs_target.exists():
            target = Folder(
                path=abs_target, dryrun=True
            )  # No need to create the folder here
        else:
            target = rel_target

        # If install method is not provided, we need to infer it
        if not install_method:
            if Path(specifier).resolve().exists():
                install_method = "reqs"
            else:
                install_method = "pip"

        # If we are just defaulting to pip, attempt to install the same version of the package
        # that is already installed locally
        # Check if the target is only letters, nothing else. This means its a string like 'numpy'.
        if install_method == "pip" and is_python_package_string(target):
            locally_installed_version = find_locally_installed_version(target)
            if locally_installed_version:
                # Check if this is a package that was installed from local
                local_install_path = get_local_install_path(target)
                if local_install_path and Path(local_install_path).exists():
                    target = Folder(path=local_install_path, dryrun=True)

        # "Local" install method is a special case where we just copy a local folder and add to path
        if install_method == "local":
            return Package(
                install_target=target, install_method=install_method, dryrun=dryrun
            )

        elif install_method in ["reqs", "pip", "conda"]:
            return Package(
                install_target=target,
                install_args=args,
                install_method=install_method,
                dryrun=dryrun,
            )
        elif install_method == "rh":
            # Calling the factory method below
            return package(name=specifier[len("rh:") :], dryrun=dryrun)
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
    dryrun: bool = False,
    local_mount: bool = False,
    data_config: Optional[Dict] = None,
) -> Package:
    """
    Builds an instance of :class:`Package`.

    Args:
        name (str): Name to assign the package resource.
        install_method (str): Method for installing the package. Options: [``pip``, ``conda``, ``reqs``, ``local``]
        install_str (str): Additional arguments to install.
        path (str): URL of the package to install.
        system (str): File system or cluster on which the package lives. Currently this must a cluster or one of:
            [``file``, ``github``, ``sftp``, ``ssh``, ``s3``, ``gs``, ``azure``].
        dryrun (bool): Whether to create the Package if it doesn't exist, or load the Package object as a dryrun.
            (Default: ``False``)
        local_mount (bool): Whether to locally mount the installed package. (Default: ``False``)
        data_config (Optional[Dict]): The data config to pass to the underlying fsspec handler.

    Returns:
        Package: The resulting package.

    Example:
        >>> import runhouse as rh
        >>> reloaded_package = rh.package(name="my-package")
        >>> local_package = rh.package(path="local/folder/path", install_method="local")
    """
    if name and not any(
        [install_method, install_str, path, system, data_config, local_mount]
    ):
        # If only the name is provided and dryrun is set to True
        return Package.from_name(name, dryrun)

    install_target = None
    install_args = None
    if path is not None:
        system = system or Folder.DEFAULT_FS
        install_target = folder(
            path=path, system=system, local_mount=local_mount, data_config=data_config
        )
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
