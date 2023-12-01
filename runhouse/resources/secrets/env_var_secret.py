import copy
import os
from pathlib import Path

from typing import Dict, List, Optional, Union

from runhouse.globals import rns_client
from runhouse.resources.blobs.file import File

from runhouse.resources.envs.env import Env
from runhouse.resources.hardware.cluster import Cluster
from runhouse.resources.hardware.utils import _get_cluster_from
from runhouse.resources.secrets.secret import Secret
from runhouse.resources.secrets.utils import _get_env_file_values
from runhouse.rns.utils.names import _generate_default_name


class EnvVarSecret(Secret):
    def __init__(
        self,
        name: Optional[str],
        values: Dict = {},
        env_vars: List[str] = None,
        path: Union[str, Path] = None,
        dryrun: bool = False,
        **kwargs,
    ):
        """
        Env Var Secret object.

        .. note::
            To create an EnvVarSecret, please use the factory method :func:`env_var_secret`.
        """
        super().__init__(name=name, values=values, dryrun=dryrun)

        if not (values or env_vars):
            values_from_path = self._from_path(path)
        self._env_vars = (
            env_vars or list(values.keys()) or list(values_from_path.keys())
        )

        if path and (values or env_vars):
            self._write_to_file(path)
        self.path = path

    @property
    def values(self):
        "Env var secret values. Dict mapping the env var to it's value."
        if self._values:
            return self._values
        if self._cached_values:
            return self._cached_values
        self._cached_values = self._from_path() or self._from_env(self.env_vars)
        return self._cached_values

    def save(self, values: bool = True, headers: str = rns_client.request_headers):
        if not self.name:
            self.name = _generate_default_name(prefix="env_var_secret")
        super().save(values=values, headers=headers)

    def to(
        self,
        system: Union[Cluster, str],
        env: Union[Env, str] = None,
        values: bool = True,
        path: Union[str, Path] = None,
        file: bool = True,
        set_env: bool = True,
        name: Optional[str] = None,
    ):
        """
        Send the secret resource to the corresponding cluster and env, and optionally set the environment
        variables. (True by default)

        If secret has a corresponding valid (.env) file associated with it, the file will be synced
        over (unless setting file=False).

        Args:
            system (str or Cluster): Cluster to send the secret to
            env (str of Env): Env on the cluster to send the secret to
            values (bool, optional): Whether to save down the values in the resource config.
            path (str, Path): Path to write the .env file to. If the secret has a valid .env
                file associated with it, the default path will be the same as the original path. (Default: None)
            file (bool): Whether to write down the .env file of the secret. Only applicable for env var secrets
                associated with a path. To write an in-memory env secret to a path, please set the path variable
                to the target remote path. (Default: True)
            set_env (bool): Whether to set the env vars of the corresponding env. (Default: True)
            name (str, ooptional): Name to assign the resource on the cluster.
        """
        new_secret = copy.deepcopy(self)
        new_secret.name = name or self.name or _generate_default_name(prefix="secret")

        system = _get_cluster_from(system)
        valid_file = bool(self._from_path())

        if system.on_this_cluster():
            if values:
                new_secret._values = self.values
                new_secret.pin()
            if path and path != self.path:
                self._write_to_file(path)
            if set_env:
                system.call(env, "_set_env_vars", self.values)
            return new_secret

        if values:
            new_secret._values = self.values

        # handle remote env secrets file
        if path or (valid_file and file):
            new_secret.path = (
                file(path, system=system)
                if path
                else file(path=self.path, system=system)
            )
        key = system.put_resource(new_secret)
        if new_secret.path:
            system.call(key, "_write_to_file", path=path)

        # handle setting remote env vars
        if set_env:
            env = env.name if isinstance(env, Env) else (env or "base_env")
            system.call(env, "_set_env_vars", self.values)

        return new_secret

    def _write_to_file(self, path):
        if isinstance(path, File):
            with path.open(mode="w") as f:
                for key, val in self.values:
                    f.write(f"{key}={val}\n")
        else:
            with open(path, "w") as f:
                for key, val in self.values:
                    f.write(f"{key}={val}\n")

    def _from_env(self, env_vars: List = None):
        env_vars = env_vars or self.env_vars
        if not env_vars:
            return {}

        values = {}
        for env_var in env_vars:
            try:
                values[env_var] = os.environ[env_var]
            except KeyError:
                raise KeyError(
                    f"Could not determine value for {env_var} from the os environment."
                )
        return values

    def _from_path(self):
        if not self.path:
            return {}

        if isinstance(self.path, File):
            if not self.path.exists_in_system():
                return {}

            from runhouse.resources.function import function

            get_remote_dotenv = function(_get_env_file_values).to(
                system=self.path.system, env=["python-dotenv"]
            )
            return get_remote_dotenv(self.path.path)
        else:
            path = os.path.expanduser(self.path)
            if not os.path.exists(path):
                return {}

            return _get_env_file_values(path)
