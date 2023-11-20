import copy
import os

from typing import Dict, List, Optional, Union

from runhouse.globals import rns_client

from runhouse.resources.envs.env import Env
from runhouse.resources.hardware.cluster import Cluster
from runhouse.resources.hardware.utils import _get_cluster_from
from runhouse.resources.secrets.secret import Secret
from runhouse.rns.utils.names import _generate_default_name


class EnvVarSecret(Secret):
    def __init__(
        self,
        name: Optional[str],
        values: Dict = None,
        env_vars: List[str] = None,
        dryrun: bool = False,
        **kwargs,
    ):
        """
        Env Var Secret object.

        .. note::
            To create an EnvVarSecret, please use the factory method :func:`env_var_secret`.
        """
        self.env_vars = env_vars or list(values.keys())
        super().__init__(name=name, values=values, dryrun=dryrun)

    @property
    def values(self):
        "Env var secret values. Dict mapping the env var to it's value."
        if self._values:
            return self._values
        return self._from_env(self.env_vars)

    # TODO: rename to set?
    def set(self, values: Dict = None, env_vars: List[str] = None):
        """Set env var values in the environment."""
        if values and env_vars and list(values.keys()) != env_vars:
            raise ValueError("env_vars list and values keys do not match.")
        if env_vars:
            if not self.values or not set(env_vars).issubset(self.values.keys()):
                raise KeyError(
                    "Could not determine all the values to be written into the env vars."
                    "You can pass them in to the `values` field."
                )
        elif values or self.values:
            values = values or self.values
            for key, val in values.items():
                os.environ[key] = val
        return

    def save(self, values: bool = True, headers: str = rns_client.request_headers):
        if not self.name:
            self.name = _generate_default_name(prefix="env_var")
        super().save(values=values, headers=headers)

    def to(
        self,
        system: Union[Cluster, str],
        values: bool = True,
        env: Union[Env, str] = None,
        name: Optional[str] = None,
    ):
        """Set the env var values on the corresponding system and env. If ``values`` is set to True,
        also save the secret to the cluster object store.

        Args:
            system (str or Cluster): Cluster to send the secret to
            values (bool, optional): Whether to save the secret and values as a resource on the cluster,
                or just have the env vars be set.
            name (str, ooptional): Name to assign the resource on the cluster.
        """
        new_secret = copy.deepcopy(self)
        new_secret.name = name or self.name or _generate_default_name(prefix="secret")

        system = _get_cluster_from(system)
        if system.on_this_cluster():
            if values:
                new_secret._values = self.values
                new_secret.pin()
        else:
            if values:
                new_secret._values = self.values
                system.put_resource(new_secret)

        env = env.name if isinstance(env, Env) else (env or "base_env")
        if not system.get(env):
            raise ValueError("Env {env} does not exist on the cluster.")
        system.call(env, "_set_env_vars", self.values)
        return new_secret

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
