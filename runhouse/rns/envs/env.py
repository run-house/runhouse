import copy
from pathlib import Path
from typing import Dict, List, Optional, Union

from runhouse.rns.folders import Folder
from runhouse.rns.hardware import Cluster
from runhouse.rns.packages import Package
from runhouse.rns.resource import Resource

from runhouse.rns.utils.hardware import _get_cluster_from


class Env(Resource):
    RESOURCE_TYPE = "env"

    def __init__(
        self,
        name: Optional[str] = None,
        reqs: List[Union[str, Package]] = [],
        setup_cmds: List[str] = None,
        env_vars: Union[Dict, str] = {},
        working_dir: Optional[Union[str, Path]] = None,
        dryrun: bool = True,
        **kwargs,  # We have this here to ignore extra arguments when calling from_config
    ):
        """
        Runhouse Env object.

        .. note::
            To create an Env, please use the factory method :func:`env`.
        """
        super().__init__(name=name, dryrun=dryrun)
        self._reqs = reqs
        self.setup_cmds = setup_cmds
        self.env_vars = env_vars
        self.working_dir = working_dir

    @staticmethod
    def from_config(config: dict, dryrun: bool = False):
        """Create an Env object from a config dict"""
        config["reqs"] = [
            Package.from_config(req, dryrun=True) if isinstance(req, dict) else req
            for req in config.get("reqs", [])
        ]
        config["working_dir"] = (
            Package.from_config(config["working_dir"], dryrun=True)
            if isinstance(config["working_dir"], dict)
            else config["working_dir"]
        )

        resource_subtype = config.get("resource_subtype")
        if resource_subtype == "CondaEnv":
            from runhouse import CondaEnv

            return CondaEnv(**config, dryrun=dryrun)

        return Env(**config, dryrun=dryrun)

    @property
    def config_for_rns(self):
        config = super().config_for_rns
        config.update(
            {
                "reqs": [
                    self._resource_string_for_subconfig(package)
                    for package in self._reqs
                ],
                "setup_cmds": self.setup_cmds,
                "env_vars": self.env_vars,
                "working_dir": self._resource_string_for_subconfig(self.working_dir),
            }
        )
        return config

    @property
    def reqs(self):
        return (self._reqs or []) + ([self.working_dir] if self.working_dir else [])

    @reqs.setter
    def reqs(self, reqs):
        self._reqs = reqs

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
        if self.working_dir:
            return new_reqs[:-1], new_reqs[-1]
        return new_reqs, None

    def _setup_env(self, system: Cluster):
        """Install packages and run setup commands on the cluster."""
        if self.reqs:
            system.install_packages(self.reqs)
        if self.setup_cmds:
            system.run(self.setup_cmds)

    def to(self, system: Union[str, Cluster], path=None, mount=False):
        """
        Send environment to the system (Cluster or file system).
        This includes installing packages and running setup commands if system is a cluster.

        Example:
            >>> env = rh.env(reqs=["numpy", "pip"])
            >>> cluster_env = env.to(my_cluster)
            >>> s3_env = env.to("s3", path="s3_bucket/my_env")
        """
        system = _get_cluster_from(system)
        new_env = copy.deepcopy(self)
        new_env.reqs, new_env.working_dir = self._reqs_to(system, path, mount)

        if isinstance(system, Cluster):
            system.check_server()
            new_env._setup_env(system)

        return new_env

    @property
    def _activate_cmd(self):
        return ""

    @property
    def _run_cmd(self):
        return ""
