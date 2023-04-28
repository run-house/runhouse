import copy
from pathlib import Path
from typing import Dict, List, Optional, Union

from runhouse.rh_config import rns_client
from runhouse.rns.folders.folder import Folder
from runhouse.rns.hardware import Cluster
from runhouse.rns.packages import Package

from runhouse.rns.packages.package import _get_cluster_from
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


def _get_env_from(env):
    if isinstance(env, Resource):
        return env
    elif isinstance(env, List):
        return Env(reqs=env)
    elif isinstance(env, Dict):
        return Env.from_config(env)
    elif isinstance(env, str) and rns_client.exists(env, resource_type="env"):
        return Env.from_name(env)
    return env


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

    def _reqs_to(self, system: Union[str, Cluster], path=None, mount=False):
        """Send self.reqs to the system (cluster or file system)"""
        new_reqs = []
        for req in self.reqs:
            if isinstance(req, str):
                new_req = Package.from_string(req)
                if isinstance(new_req.install_target, Folder):
                    req = new_req

            if isinstance(req, Package) and isinstance(req.install_target, Folder):
                req = (
                    req.to(system, path=path, mount=mount)
                    if isinstance(system, Cluster)
                    else req.to(system, path=path)
                )
            new_reqs.append(req)
        return new_reqs

    def _setup_env(self, system: Union[str, Cluster]):
        """Install packages and run setup commands on the cluster."""
        if self.reqs:
            system.install_packages(self.reqs)
        if self.setup_cmds:
            system.run(self.setup_cmds)

    def to(self, system: Union[str, Cluster], path=None, mount=False):
        """
        Send environment to the system (Cluster or file system).
        This includes installing packages and running setup commands if system is a cluster
        """
        system = _get_cluster_from(system)
        new_env = copy.deepcopy(self)
        new_env.reqs = self._reqs_to(system, path, mount)

        if isinstance(system, Cluster):
            new_env._setup_env(system)

        return new_env

    def install_packages(
        self, pkgs: List[str or Package], system: Union[Cluster, Dict, str]
    ):
        """
        Additionally install given packages on the environment on the system.
        Note that this does not update the runhouse Env object or metadata, but simply installs
        packages on the existing env on the system.
        """
        system = _get_cluster_from(system)
        assert isinstance(system, Cluster)
        system.install_packages(pkgs, self._run_cmd)

    @property
    def _activate_cmd(self):
        return ""

    @property
    def _run_cmd(self):
        return ""


def env(
    name: Optional[str] = None,
    reqs: List[Union[str, Package]] = None,
    conda_env: Union[str, Dict] = None,
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
    config["conda_env"] = conda_env or config.get("conda_env")

    if config["conda_env"]:
        from runhouse.rns.envs.conda_env import CondaEnv

        return CondaEnv.from_config(config, dryrun=dryrun)

    return Env.from_config(config, dryrun=dryrun)
