import copy
from typing import List, Optional, Union

from runhouse.rns.folders.folder import Folder
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
        dryrun: bool = True,
        **kwargs,  # We have this here to ignore extra arguments when calling from_config
    ):
        """
        Runhouse Env object.
        .. note::
            To create an Env, please use the factory method :func:`env`.
        """
        super().__init__(name=name, dryrun=dryrun)
        self.reqs = reqs
        self.setup_cmds = setup_cmds

    @staticmethod
    def from_config(config: dict, dryrun: bool = True):
        """Create an Env object from a config dict"""
        config["reqs"] = [
            Package.from_config(req) if isinstance(req, dict) else req
            for req in config.get("reqs", [])
        ]

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

    def _setup_env(self, system: Cluster):
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
            system.check_server()
            new_env._setup_env(system)

        return new_env

    @property
    def _activate_cmd(self):
        return ""

    @property
    def _run_cmd(self):
        return ""
