import copy
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

from runhouse.globals import obj_store
from runhouse.resources.folders import Folder
from runhouse.resources.hardware import _get_cluster_from, Cluster
from runhouse.resources.packages import Package
from runhouse.resources.resource import Resource

from runhouse.utils import run_with_logs

from .utils import _env_vars_from_file


logger = logging.getLogger(__name__)


class Env(Resource):
    RESOURCE_TYPE = "env"
    DEFAULT_NAME = "base_env"

    def __init__(
        self,
        name: Optional[str] = None,
        reqs: List[Union[str, Package]] = [],
        setup_cmds: List[str] = None,
        env_vars: Union[Dict, str] = {},
        working_dir: Optional[Union[str, Path]] = None,
        secrets: Optional[Union[str, "Secret"]] = [],
        compute: Optional[Dict] = {},
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
        self.secrets = secrets
        self.compute = compute

    @property
    def env_name(self):
        return self.name or "base"

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

    @staticmethod
    def _set_env_vars(env_vars):
        for k, v in env_vars.items():
            os.environ[k] = v

    def config(self, condensed=True):
        config = super().config(condensed)
        self.save_attrs_to_config(
            config, ["setup_cmds", "env_vars", "env_name", "compute"]
        )
        config.update(
            {
                "reqs": [
                    self._resource_string_for_subconfig(package, condensed)
                    for package in self._reqs
                ],
                "working_dir": self._resource_string_for_subconfig(
                    self.working_dir, condensed
                ),
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

    def _secrets_to(self, system: Union[str, Cluster]):
        from runhouse.resources.secrets import Secret

        new_secrets = []
        for secret in self.secrets:
            if isinstance(secret, str):
                secret = Secret.from_name(secret)
            if hasattr(secret, "path") and secret.path:
                new_secrets.append(secret.to(system=system))
            else:
                new_secrets.append(secret.to(system=system, env=self))
        return new_secrets

    def install(self, force=False):
        """Locally install packages and run setup commands."""
        # Hash the config_for_rns to check if we need to install
        env_config = self.config()
        # Remove the name because auto-generated names will be different, but the installed components are the same
        env_config.pop("name")
        install_hash = hash(str(env_config))
        # Check the existing hash
        if install_hash in obj_store.installed_envs and not force:
            logger.debug("Env already installed, skipping")
            return
        obj_store.installed_envs[install_hash] = self.name

        for package in self.reqs:
            if isinstance(package, str):
                pkg = Package.from_string(package)
            elif hasattr(package, "_install"):
                pkg = package
            else:
                raise ValueError(f"package {package} not recognized")

            logger.debug(f"Installing package: {str(pkg)}")
            pkg._install(self)
        if self.setup_cmds:
            for cmd in self.setup_cmds:
                self._run_command(cmd)

    def _run_command(self, command: str, **kwargs):
        """Run command locally inside the environment"""
        if self._run_cmd:
            command = f"{self._run_cmd} {command}"
        logging.info(f"Running command in {self.name}: {command}")
        return run_with_logs(command, **kwargs)

    def to(
        self, system: Union[str, Cluster], path=None, mount=False, force_install=False
    ):
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
        new_env.secrets = self._secrets_to(system)

        if isinstance(system, Cluster):
            key = system.put_resource(new_env)
            env_vars = (
                _env_vars_from_file(self.env_vars)
                if isinstance(self.env_vars, str)
                else self.env_vars
            )
            if env_vars:
                system.call(key, "_set_env_vars", env_vars)
            system.call(key, "install", force=force_install)

        return new_env

    @property
    def _activate_cmd(self):
        return ""

    @property
    def _run_cmd(self):
        return ""
