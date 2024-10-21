import copy
import shlex
from pathlib import Path
from typing import Dict, List, Optional, Union

from runhouse.globals import obj_store
from runhouse.logger import get_logger
from runhouse.resources.hardware import _get_cluster_from, Cluster
from runhouse.resources.packages import InstallTarget, Package
from runhouse.resources.resource import Resource

from runhouse.utils import _process_env_vars, run_setup_command, run_with_logs

logger = get_logger(__name__)


def install_reqs_on_cluster(
    system: Union[str, Cluster], reqs: List[Union[str, Package]], path=None
):
    new_reqs = []
    for req in reqs:
        if isinstance(req, str):
            new_req = Package.from_string(req)
            req = new_req

        if isinstance(req, Package) and isinstance(req.install_target, InstallTarget):
            req = req.to(system, path=path)
        new_reqs.append(req)

    return new_reqs


class Env(Resource):
    RESOURCE_TYPE = "env"

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
    def from_config(config: dict, dryrun: bool = False, _resolve_children: bool = True):
        config["reqs"] = [
            Package.from_config(req, dryrun=True, _resolve_children=_resolve_children)
            if isinstance(req, dict)
            else req
            for req in config.get("reqs", [])
        ]
        config["working_dir"] = (
            Package.from_config(
                config["working_dir"], dryrun=True, _resolve_children=_resolve_children
            )
            if isinstance(config.get("working_dir"), dict)
            else config.get("working_dir")
        )

        resource_subtype = config.get("resource_subtype")
        if resource_subtype == "CondaEnv":
            from runhouse import CondaEnv

            return CondaEnv(**config, dryrun=dryrun)

        return Env(**config, dryrun=dryrun)

    def add_env_var(self, key: str, value: str):
        """Add an env var to the environment. Environment must be re-installed to propagate new
        environment variables if it already lives on a cluster."""
        self.env_vars.update({key: value})

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

    def _secrets_to(self, system: Union[str, Cluster]):
        from runhouse.resources.secrets import Secret

        new_secrets = []
        for secret in self.secrets:
            if isinstance(secret, str):
                secret = Secret.from_name(secret)
            new_secrets.append(secret.to(system=system, env=self))
        return new_secrets

    def _run_setup_cmds(
        self, cluster: Cluster = None, setup_cmds: List = None, node: str = "all"
    ):
        setup_cmds = setup_cmds or self.setup_cmds

        if not setup_cmds:
            return

        for cmd in setup_cmds:
            cmd = self._full_command(cmd)
            run_setup_command(
                cmd,
                cluster=cluster,
                env_vars=_process_env_vars(self.env_vars),
                node=node,
            )

    def install(self, force: bool = False, cluster: Cluster = None, node: str = "all"):
        """Locally install packages and run setup commands.

        Args:
            force (bool, optional): Whether to setup the installation again if the env already exists
                on the cluster. (Default: ``False``)
            cluster (Cluster, optional): Cluster to install the env on. If not provided, env is installed
                on the current cluster. (Default: ``None``)
            node (str, optional): Node to install the env on. (Default: ``"all"``)
        """
        # If we're doing the install remotely via SSH (e.g. for default_env), there is no cache
        if not cluster:
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

        self._install_reqs(cluster=cluster, node=node)
        self._run_setup_cmds(cluster=cluster, node=node)

    def _full_command(self, command: str):
        if self._run_cmd:
            return f"{self._run_cmd} ${{SHELL:-/bin/bash}} -c {shlex.quote(command)}"
        return command

    def _run_command(self, command: str, **kwargs):
        """Run command locally inside the environment"""
        command = self._full_command(command)
        logger.info(f"Running command in {self.name}: {command}")
        return run_with_logs(command, **kwargs)

    def to(
        self,
        system: Union[str, Cluster],
        node_idx: Optional[int] = None,
        path: str = None,
        force_install: bool = False,
    ):
        """
        Send environment to the system, and set it up if on a cluster.

        Args:
            system (str or Cluster): Cluster or file system to send the env to.
            node_idx (int, optional): Node index of the cluster to send the env to. If not specified,
                uses the head node. (Default: ``None``)
            path (str, optional): Path on the cluster to sync the env's working dir to. Uses a default
                path if not specified. (Default: ``None``)
            force_install (bool, optional): Whether to setup the installation again if the env already
                exists on the cluster. (Default: ``False``)

        Example:
            >>> env = rh.env(reqs=["numpy", "pip"])
            >>> cluster_env = env.to(my_cluster)
            >>> s3_env = env.to("s3", path="s3_bucket/my_env")
        """
        system = _get_cluster_from(system)
        if (
            isinstance(system, Cluster)
            and node_idx is not None
            and node_idx >= len(system.ips)
        ):
            raise ValueError(
                f"Cluster {system.name} has only {len(system.ips)} nodes. Requested node index {node_idx} is out of bounds."
            )

        new_env = copy.deepcopy(self)

        if isinstance(system, Cluster):
            if node_idx is not None:
                new_env.compute = new_env.compute or {}
                new_env.compute["node_idx"] = node_idx

            key = (
                system.put_resource(new_env)
                if new_env.name
                else system.default_env.name
            )

            env_vars = _process_env_vars(self.env_vars)
            if env_vars:
                system.set_process_env_vars(process_name=key, env_vars=env_vars)

            conda_env_name = new_env.env_name if hasattr(self, "conda_yaml") else None
            system.install_packages(reqs=new_env.reqs, conda_env_name=conda_env_name)
            system.call(key, "_run_setup_cmds", setup_cmds=new_env.setup_cmds)

            # Secrets are resources that go in the env, so put them in after the env is created
            new_env.secrets = self._secrets_to(system)

        return new_env

    @property
    def _activate_cmd(self):
        return ""

    @property
    def _run_cmd(self):
        return ""
