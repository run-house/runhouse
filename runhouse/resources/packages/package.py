import copy
import logging
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional, Union

from runhouse import globals
from runhouse.resources.folders import Folder, folder
from runhouse.resources.hardware.utils import _get_cluster_from
from runhouse.resources.resource import Resource
from runhouse.utils import install_conda, run_with_logs

INSTALL_METHODS = {"local", "reqs", "pip", "conda"}

logger = logging.getLogger(__name__)


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

    def _install(self, env: Union[str, "Env"] = None):
        """Install package.

        Args:
            env (Env or str): Environment to install package on. If left empty, defaults to base environment.
                (Default: ``None``)
        """
        logging.info(f"Installing {str(self)} with method {self.install_method}.")
        install_cmd = ""
        install_args = f" {self.install_args}" if self.install_args else ""
        cuda_version_or_cpu = self._detect_cuda_version_or_cpu()

        if isinstance(self.install_target, Folder):
            local_path = self.install_target.local_path
            if self.install_method == "pip":
                # TODO [DG] Revisit: Would be nice if we could use -e by default, but importlib on the rpc server
                #  isn't finding the package right after its installed.
                # if (Path(local_path) / 'setup.py').exists():
                #     install_cmd = f'-e {local_path}' + install_args
                # else:
                install_cmd = f"{local_path}" + install_args
            elif self.install_method == "conda":
                install_cmd = f"{local_path}" + install_args
            elif self.install_method == "reqs":
                reqs_path = f"{local_path}/requirements.txt"
                logging.info(f"reqs path: {reqs_path}")
                if Path(reqs_path).expanduser().exists():
                    install_cmd = self._requirements_txt_install_cmd(
                        path=reqs_path,
                        cuda_version_or_cpu=cuda_version_or_cpu,
                        args=install_args,
                    )
                    logging.info(
                        f"pip installing requirements from {reqs_path} with: {install_cmd}"
                    )
                    self._pip_install(install_cmd)
                else:
                    logging.info(f"{local_path}/requirements.txt not found, skipping")
        else:
            install_cmd = self.install_target + install_args

        if self.install_method == "pip":
            install_cmd = self._install_cmd_for_torch(install_cmd, cuda_version_or_cpu)
            if not install_cmd:
                raise ValueError("Invalid install command")

            self._pip_install(install_cmd, env)
        elif self.install_method == "conda":
            self._conda_install(install_cmd, env)
        elif self.install_method in ["local", "reqs"]:
            if isinstance(self.install_target, Folder):
                sys.path.append(local_path)
            elif Path(self.install_target).resolve().expanduser().exists():
                sys.path.append(str(Path(self.install_target).resolve().expanduser()))
            else:
                raise ValueError(
                    f"install_target must be a Folder or a path to a directory for "
                    f"install_method {self.install_method}"
                )
        # elif self.install_method == 'unpickle':
        #     # unpickle the functions and make them importable
        #     with self.get('functions.pickle') as f:
        #         sys.modules[self.name] = pickle.load(f)
        else:
            raise ValueError(
                f"Unknown install_method {self.install_method}. Try using cluster.run() or to install instead."
            )

    # ----------------------------------
    # Torch Install Helpers
    # ----------------------------------
    def _requirements_txt_install_cmd(self, path, cuda_version_or_cpu="", args=""):
        """Read requirements from file, append --index-url and --extra-index-url where relevant for torch packages,
        and return list of formatted packages."""
        with open(path) as f:
            reqs = f.readlines()

        # if torch extra index url is already defined by the user or torch isn't a req, directly pip install reqs file
        if not [req for req in reqs if "torch" in req]:
            return f"-r {path}" + args
        for req in reqs:
            if (
                "--index-url" in req or "--extra-index-url" in req
            ) and "pytorch.org" in req:
                return f"-r {path}" + args

        # add extra-index-url for torch if not found
        return (
            f"-r {path} --extra-index-url {self._torch_index_url(cuda_version_or_cpu)}"
        )

    def _install_cmd_for_torch(self, install_cmd, cuda_version_or_cpu):
        """Return the correct formatted pip install command for the torch package(s) provided."""
        if install_cmd.startswith("#"):
            return None

        torch_source_packages = ["torch", "torchvision", "torchaudio"]
        if not any([x in install_cmd for x in torch_source_packages]):
            return install_cmd

        packages_to_install: list = self._packages_to_install_from_cmd(install_cmd)
        final_install_cmd = ""
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
    def _detect_cuda_version_or_cpu():
        """Return the CUDA version on the cluster. If we are on a CPU-only cluster return 'cpu'.

        Note: A cpu-only machine may have the CUDA toolkit installed, which means nvcc will still return
        a valid version. Also check if the NVIDIA driver is installed to confirm we are on a GPU."""
        try:
            cuda_version_info: str = subprocess.check_output(
                "nvcc --version", shell=True
            ).decode("utf-8")
            cuda_version = cuda_version_info.split("release ")[1].split(",")[0]
        except subprocess.CalledProcessError:
            return "cpu"

        try:
            subprocess.run(["nvidia-smi"], check=True)
            return cuda_version
        except (subprocess.CalledProcessError, FileNotFoundError):
            return "cpu"

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

    # ----------------------------------

    @staticmethod
    def _pip_install(install_cmd: str, env: Union[str, "Env"] = ""):
        """Run pip install."""
        pip_cmd = f"pip install {install_cmd}"
        if env:
            from runhouse.resources.envs.utils import _get_env_from

            env = _get_env_from(env)
            env._run_command(pip_cmd)
        else:
            cmd = f"{sys.executable} -m {pip_cmd}"
            retcode = run_with_logs(cmd)
            if retcode != 0:
                raise RuntimeError(
                    "Pip install failed, check that the package exists and is available for your platform."
                )

    @staticmethod
    def _conda_install(install_cmd: str, env: Union[str, "Env"] = ""):
        """Run conda install."""
        cmd = f"conda install -y {install_cmd}"
        if env:
            if isinstance(env, str):
                from runhouse.resources.envs import Env

                env = Env.from_name(env)
            cmd = f"{env._run_cmd} {cmd}"

        install_conda()

        retcode = run_with_logs(cmd)
        if retcode != 0:
            raise RuntimeError(
                "Conda install failed, check that the package exists and is "
                "available for your platform."
            )

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

        system = _get_cluster_from(system)
        if self.install_target.system == system:
            return self

        if isinstance(system, Resource):
            if isinstance(self.install_target.system, Resource):
                # if these are both clusters, check if they're pointing to the same address.
                # We use endpoint instead of address here because if address is localhost, we need to port too
                if self.install_target.system.endpoint(
                    external=False
                ) == system.endpoint(external=False):
                    # If we're on the target system, just make sure the package is in the Python path
                    sys.path.append(self.install_target.local_path)
                    return self
            logger.info(
                f"Copying package from {self.install_target.fsspec_url} to: {getattr(system, 'name', system)}"
            )
            new_folder = self.install_target._to_cluster(system, path=path, mount=mount)
        else:  # to fs
            new_folder = self.install_target.to(system, path=path)
        new_folder.system = system
        new_package = copy.copy(self)
        new_package.install_target = new_folder
        return new_package

    @staticmethod
    def from_config(config: dict, dryrun=False):
        if isinstance(config.get("install_target"), dict):
            config["install_target"] = Folder.from_config(
                config["install_target"], dryrun=dryrun
            )
        if config.get("resource_subtype") == "GitPackage":
            from runhouse import GitPackage

            return GitPackage.from_config(config, dryrun=dryrun)
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

        target_and_args = specifier
        if specifier.split(":")[0] in INSTALL_METHODS:
            target_and_args = specifier.split(":", 1)[1]
        rel_target, args = (
            target_and_args.split(" ", 1)
            if " " in target_and_args
            else (target_and_args, "")
        )

        # We need to do this because relative paths are relative to the current working directory!
        abs_target = (
            Path(rel_target).expanduser()
            if Path(rel_target).expanduser().is_absolute()
            else Path(globals.rns_client.locate_working_dir()) / rel_target
        )
        if abs_target.exists():
            target = Folder(
                path=abs_target, dryrun=True
            )  # No need to create the folder here
        else:
            target = rel_target

        if specifier.startswith("local:"):
            return Package(install_target=target, install_method="local", dryrun=dryrun)
        elif specifier.startswith("reqs:"):
            return Package(
                install_target=target,
                install_args=args,
                install_method="reqs",
                dryrun=dryrun,
            )
        elif specifier.startswith("pip:"):
            return Package(
                install_target=target,
                install_args=args,
                install_method="pip",
                dryrun=dryrun,
            )
        elif specifier.startswith("conda:"):
            return Package(
                install_target=target,
                install_args=args,
                install_method="conda",
                dryrun=dryrun,
            )
        elif specifier.startswith("rh:"):
            # Calling the factory method below
            return package(name=specifier[3:], dryrun=dryrun)
        else:
            if Path(specifier).resolve().exists():
                return Package(
                    install_target=target,
                    install_args=args,
                    install_method="reqs",
                    dryrun=dryrun,
                )
            else:
                return Package(
                    install_target=target,
                    install_args=args,
                    install_method="pip",
                    dryrun=dryrun,
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
