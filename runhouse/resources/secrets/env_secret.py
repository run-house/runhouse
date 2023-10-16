import copy
import os
from pathlib import Path

from typing import Dict, List, Optional, Union

from runhouse.resources.hardware.cluster import Cluster
from runhouse.resources.hardware.utils import _get_cluster_from

from runhouse.resources.secrets.secret import Secret
from runhouse.resources.secrets.utils import load_config


class EnvSecret(Secret):
    DEFAULT_ENV_DIR = "~/.rh/secrets/envs"
    DEFAULT_NAME = "env_vars"

    def __init__(
        self,
        name: Optional[str] = None,
        values: Dict = {},
        path: Union[str, Path] = None,
        env_vars: List = None,
        dryrun: bool = False,
        **kwargs,
    ):
        """
        Runhouse EnvSecret object.

        .. note::
            To create an EnvSecret, please use the factory method :func:`env_secret`.
        """
        name = name or self.DEFAULT_NAME
        super().__init__(name, values, path, dryrun)
        self.env_vars = env_vars or {key: key for key in self.values.keys()}

    @staticmethod
    def from_config(config: dict, dryrun: bool = False):
        """Create a EnvSecret object from a config dictionary."""
        return EnvSecret(**config, dryrun=dryrun)

    def _from_path(self, path: Union[str, Path] = None, env_vars: List[str] = None):
        from dotenv import dotenv_values

        path = path or self.path
        path = os.path.expanduser(path)
        config = dotenv_values(path)
        if not config:
            return {}
        env_vars = env_vars or self.env_vars
        if not env_vars:
            return config
        if not set(env_vars).issubset(set(config.keys())):
            return {}
        return {key: config[key] for key in env_vars}

    def _from_env(self, keys: List[str] = None):
        keys = keys or self.env_vars
        if not keys:
            return {}
        return dict((key, os.environ[key]) for key in keys)

    def save(self):
        """Save the config, into Vault if the user is logged in,
        or to local if not or if the resource is a local resource."""
        if not self.rns_address.startswith("/"):
            config = load_config(self.rns_address, self.USER_ENDPOINT)
        else:
            config = load_config(self.name, self.USER_ENDPOINT)

        if config:
            new_secret = EnvSecret(**config, dryrun=True)
            new_secret._values = {**new_secret._values, **self._values}
            new_secret.env_vars = new_secret.env_vars + [
                env_var
                for env_var in self.env_vars
                if env_var not in new_secret.env_vars
            ]
            new_secret._basic_save()
            return new_secret

        super().save()
        return self

    def _basic_save(self):
        return super().save()

    def write(
        self,
        path: Union[str, Path] = None,
        python_env: bool = False,
        keys: List[str] = None,
        overwrite: bool = False,
    ):
        """Write down the env var secret values.

        Args:
            python_env (bool, optional): Whether to set Python os environment variables. (Default: False)
            path (str or Path, optional): File to write down the environment variables to (e.g. .env), if any.
            keys (List[str], optional): List of keys to write down. If none is provided, will write down all.
            overwrite (bool, optional): Whether to overwrite existing env vars if they have already been previously set.
                (Default: False)
        """
        keys = keys or self.env_vars
        keys_dict = {key: key for key in keys}

        new_secret = copy.deepcopy(self)
        if keys:
            new_secret.env_vars = keys

        if not python_env:
            path = path or self.path or os.path.join(self.DEFAULT_ENV_DIR, self.name)
        if path:
            new_secret.path = path

        super().write_env_vars(
            python_env=python_env, path=path, env_vars=keys_dict, overwrite=overwrite
        )
        return new_secret

    def delete(
        self,
        python_env: bool = False,
        file: bool = False,
        keys: List[str] = None,
    ):
        """Delete the resource and it's corresponding environment variables from where they are stored.

        Args:
            python_env (bool, optional): Whether to unset the environment variables from Python os environment.
                (Default: False)
            file (bool, optional): Whether to delete the env variables from the file. (Default: False)
            keys (List[str], optional): List of env var keys to delete. If none is provided, all of the env vars
                associated with the secret will be deleted from the path and/or env.
        """
        self.remove(python_env=python_env, file=file, keys=keys)
        super().delete(file=False)

    def remove(
        self,
        python_env: bool = False,
        file: bool = False,
        keys: List[str] = None,
    ):
        """Remove the environment variables from where they are stored.

        Args:
            python_env (bool, optional): Whether to unset the environment variables from Python os environment.
                (Default: False)
            file (bool, optional): Whether to delete the file of environment variables. (Default: False)
            keys (List[str], optional): List of env var keys to delete. If none is provided, all of the env vars
                associated with the secret will be deleted from the path and/or env.
        """
        to_delete = keys or self.env_vars
        if python_env:
            for key in to_delete:
                del os.environ[key]

        if file and self.path:
            super().delete_env_vars(
                path=self.path, keys=to_delete, env_vars={key: key for key in to_delete}
            )

        if not keys:
            return self

        new_secret = copy.deepcopy(self)
        for key in keys:
            del new_secret._values[key]

        return new_secret

    def to(
        self,
        system: Union[str, Cluster],
        path: Union[str, Path] = None,
        python_env: bool = False,
    ):
        system = _get_cluster_from(system)
        new_secret = super().to(system=system, path=path)

        if system.on_this_cluster() and path:
            new_secret.write(path=path, python_env=python_env)
            new_secret.path = path

        if python_env:
            new_secret.write(python_env=python_env)
        return new_secret
