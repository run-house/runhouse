import copy
import logging
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional, Union

from runhouse import rh_config
from runhouse.rns.folders.folder import Folder
from runhouse.rns.resource import Resource

INSTALL_METHODS = {"local", "reqs", "pip", "conda"}


class Package(Resource):
    RESOURCE_TYPE = "package"

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
        """
        super().__init__(
            name=name,
            dryrun=dryrun,
        )
        self.install_method = install_method
        self.install_target = install_target
        self.install_args = install_args

    @property
    def config_for_rns(self):
        # If the package is just a simple Package.from_string string, no
        # need to store it in rns, just give back the string.
        # if self.install_method in ['pip', 'conda', 'git']:
        #     return f'{self.install_method}:{self.name}'
        config = super().config_for_rns
        config["install_method"] = self.install_method
        config["install_target"] = self._resource_string_for_subconfig(
            self.install_target
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

    def install(self):
        """Install package."""
        logging.info(
            f"Installing package {str(self)} with method {self.install_method}."
        )
        install_cmd = ""
        install_args = f" {self.install_args}" if self.install_args else ""
        if isinstance(self.install_target, Folder):
            local_path = self.install_target.local_path
            if not local_path:
                local_path = "~/" + self.name
            elif not self.install_target.is_local():
                # TODO [DG] replace this with empty mount() call to be put in tmp folder by Folder
                local_path = self.install_target.mount(
                    path=f"~/{Path(self.install_target.path).stem}"
                )

            if self.install_method == "pip":
                # TODO [DG] Revisit: Would be nice if we could use -e by default, but importlib on the grpc server
                #  isn't finding the package right after its installed.
                # if (Path(local_path) / 'setup.py').exists():
                #     install_cmd = f'-e {local_path}' + install_args
                # else:
                install_cmd = f"{local_path}" + install_args
            elif self.install_method == "conda":
                install_cmd = f"{local_path}" + install_args
            elif self.install_method == "reqs":
                if (Path(local_path) / "requirements.txt").exists():
                    logging.info(
                        f"Attempting to install requirements from {local_path}/requirements.txt"
                    )
                    self.pip_install(
                        f"-r {Path(local_path)}/requirements.txt" + install_args
                    )
                else:
                    logging.info(f"{local_path}/requirements.txt not found, skipping")
        else:
            install_cmd = self.install_target + install_args

        if self.install_method == "pip":
            self.pip_install(install_cmd)
        elif self.install_method == "conda":
            self.conda_install(install_cmd)
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
                f"Unknown install_method {self.install_method}. Try using cluster.run() or "
                f"function.run_setup() to install instead."
            )

    @staticmethod
    def pip_install(install_cmd: str):
        """Run pip install."""
        logging.info(f"Running: pip install {install_cmd}")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install"] + install_cmd.split(" ")
        )

    @staticmethod
    def conda_install(install_cmd: str):
        """Run conda install."""
        logging.info(f"Running: conda install {install_cmd}")
        # check if conda is installed, and if not, install it
        try:
            subprocess.check_call(["conda", "--version"])
            subprocess.run(["conda", "install", "-y"] + install_cmd.split(" "))
        except FileNotFoundError:
            logging.info("Conda not found, installing...")
            subprocess.check_call(
                "wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh "
                "-O ~/miniconda.sh".split(" ")
            )
            subprocess.check_call(["bash", "~/miniconda.sh", "-b", "-p", "~/miniconda"])
            subprocess.check_call("source $HOME/miniconda3/bin/activate".split(" "))
            status = subprocess.check_call(
                ["conda", "install", "-y"] + install_cmd.split(" ")
            )
            if not status == 0:
                raise RuntimeError(
                    "Conda install failed, check that the package exists and is "
                    "available for your platform."
                )

    def to_cluster(self, dest_cluster: "Cluster", path=None, mount=False):
        """Returns a copy of the package on the destination cluster."""
        if not isinstance(self.install_target, Folder):
            raise TypeError(
                "`install_target` must be a Folder in order to copy the package to a cluster"
            )

        new_folder = self.install_target.to_cluster(
            dest_cluster,
            path=path,
            mount=mount,
        )
        new_folder.system = "file"
        new_package = copy.copy(self)
        new_package.install_target = new_folder
        return new_package

    @staticmethod
    def from_config(config: dict, dryrun=False):
        if config.get("resource_subtype") == "GitPackage":

            from runhouse import GitPackage

            return GitPackage.from_config(config, dryrun=dryrun)
        return Package(**config, dryrun=dryrun)

    @staticmethod
    def from_string(specifier: str, dryrun=False):
        # Use regex to check if specifier matches '<method>:https://github.com/<path>' or 'https://github.com/<path>'
        match = re.search(
            r"^(?:(?P<method>[^:]+):)?(?P<path>https://github.com/.+)", specifier
        )
        if match:
            install_method = match.group("method")
            url = match.group("path")
            from runhouse.rns.packages.git_package import git_package

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
            else Path(rh_config.rns_client.locate_working_dir()) / rel_target
        )
        if abs_target.exists():
            target = Folder(
                path=rel_target, dryrun=True
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
    url: str = None,
    system: str = Folder.DEFAULT_FS,
    dryrun: bool = False,
    local_mount: bool = False,
    data_config: Optional[Dict] = None,
    load: bool = True,
) -> Package:
    """
    Builds an instance of :class:`Package`.

    Args:
        name (str): Name to assign the pacakge.
        install_method (str): Method for installing the package. Options: [``pip``, ``conda``, ``reqs``, ``local``]
        install_str (str): Additional arguments to install.  # TODO: not too sure about this
        url (str): URL of the package to install.
        system (str): File system. Currently this must be one of:
            [``file``, ``github``, ``sftp``, ``ssh``,``s3``, ``gs``, ``azure``].
            We are working to add additional file system support.
        dryrun (bool): Whether to create the Package if it doesn't exist, or load the Package object as a dryrun.
            (Default: ``False``)
        local_mount (bool): Whether to locally mount the installed package. (Default: ``False``)
        data_config (Optional[Dict]): The data config to pass to the underlying fsspec handler.
        load (bool): Whether to load an existing config for the Package. (Default: ``True``)

    Returns:
        Package: The resulting package.
    """
    config = rh_config.rns_client.load_config(name) if load else {}
    config["name"] = name or config.get("rns_address", None) or config.get("name")

    config["install_method"] = install_method or config.get("install_method")
    if url is not None:
        config["install_target"] = Folder(
            path=url, system=system, local_mount=local_mount, data_config=data_config
        )
        config["install_args"] = install_str
    elif install_str is not None:
        config["install_target"], config["install_args"] = install_str.split(" ", 1)
    elif "install_target" in config and isinstance(config["install_target"], dict):
        config["install_target"] = Folder.from_config(config["install_target"])

    new_package = Package.from_config(config, dryrun=dryrun)

    return new_package
