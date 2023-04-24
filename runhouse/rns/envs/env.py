import copy
from pathlib import Path
from typing import List, Optional, Union

from runhouse.rh_config import rns_client
from runhouse.rns.folders.folder import Folder
from runhouse.rns.hardware import Cluster
from runhouse.rns.packages import Package
from runhouse.rns.resource import Resource


def _process_reqs(reqs):
    preprocessed_reqs = []
    for package in reqs:
        # TODO [DG] the following is wrong. RNS address doesn't have to start with '/'. However if we check if each
        #  string exists in RNS this will be incredibly slow, so leave it for now.
        if (
            isinstance(package, str)
            and package[0] == "/"
            and rns_client.exists(package)
        ):
            # If package is an rns address
            package = rns_client.load_config(package)
        elif (
            isinstance(package, str)
            and Path(package.split(":")[-1]).expanduser().exists()
        ):
            # if package refers to a local path package
            package = Package.from_string(package)
        elif isinstance(package, dict):
            package = Package.from_config(package, dryrun=True)
        preprocessed_reqs.append(package)
    return preprocessed_reqs


class Env(Resource):
    RESOURCE_TYPE = "env"

    def __init__(
        self,
        name: Optional[str] = None,
        reqs: List[Union[str, Package]] = None,
        setup_cmds: List[str] = None,
        dryrun: bool = True,
        **kwargs,  # We have this here to ignore extra arguments when calling from_config
    ):
        """
        Runhouse Env object.
        .. note::
            To create an Env, please use the factory method :func:`env`.
        """
        super().__init__(name=name, dryrun=dryrun)
        self.reqs = _process_reqs(reqs)
        self.setup_cmds = setup_cmds

    @staticmethod
    def from_config(config: dict, dryrun: bool = True):
        """Create an Env object from a config dict"""
        return Env(**config, dryrun=dryrun)

    @property
    def config_for_rns(self):
        config = super().config_for_rns
        config.update(
            {
                "reqs": [
                    self._resource_string_for_subconfig(package)
                    for package in self.reqs
                ],
                "setup_cmds": self.setup_cmds,
            }
        )
        return config

    def to(self, system: Union[str, Cluster], path=None, mount=False):
        # env doesn't have a concept of system, so this just sets up the environment on the given cluster
        new_reqs = []
        for req in self.reqs:
            if isinstance(req, str):
                req = Package.from_string(req)
            if isinstance(req.install_target, Folder):
                req = req.to(system, path=path, mount=mount)
            new_reqs.append(req)
        new_env = copy.deepcopy(self)
        new_env.reqs = new_reqs

        system.install_packages(new_env.reqs)

        if new_env.setup_cmds:
            system.run(new_env.setup_cmds)

        return new_env


def env(
    reqs: List[Union[str, Package]] = None,
    name: Optional[str] = None,
    setup_cmds: List[str] = None,
    dryrun: bool = True,
    load: bool = True,
):
    """Factory method for creating a Runhouse Env object.
    Args:
        reqs (List[str]): List of package names to install in this environment.
        conda_env (Union[str, Dict], optional): Path to a conda environment file or Dict representing conda env.
        name (Optional[str], optional): Name of the environment.
        dryrun (bool, optional): Whether to run in dryrun mode. (Default: ``True``)
        load (bool, optional): Whether to load the environment. (Default: ``True``)
    Returns:
        Env: The resulting Env object.
    Example:
        >>> # regular python env
        >>> env = rh.env(reqs=["pytorch", "pip"])
        >>> env = rh.env(reqs=["requirements.txt"], name="myenv")
    """

    config = rns_client.load_config(name) if load else {}
    config["name"] = name or config.get("rns_address", None) or config.get("name")

    reqs = reqs if reqs is not None else config.get("reqs", [])
    config["reqs"] = reqs

    config["setup_cmds"] = (
        setup_cmds if setup_cmds is not None else config.get("setup_cmds")
    )

    return Env.from_config(config, dryrun=dryrun)
