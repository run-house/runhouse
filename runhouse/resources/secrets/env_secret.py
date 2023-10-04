import copy
import os
from pathlib import Path

from typing import Dict, List, Optional, Union

from runhouse.resources.secrets.secret import Secret
from runhouse.resources.secrets.utils import load_config


class EnvSecret(Secret):
    DEFAULT_ENV_PATH = "~/.rh/.env"
    DEFAULT_NAME = "env_vars"

    def __init__(
        self,
        name: Optional[str] = None,
        secrets: Dict = {},
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
        path = path or self.DEFAULT_ENV_PATH
        super().__init__(name, secrets, path, dryrun)
        self.env_vars = env_vars

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
        return dict((key, config[key]) for key in env_vars)

    def _from_env(self, keys: List[str] = None):
        keys = keys or self.env_vars
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
            new_secret._secrets = {**new_secret._secrets, **self._secrets}
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
        python_env: bool = False,
        path: Union[str, Path] = None,
        keys: List[str] = None,
        override: bool = False,
    ):
        """Write down the env var secrets.

        Args:
            python_env (bool, optional): Whether to set Python os environment variables. (Default: False)
            path (str or Path, optional): File to write down the environment variables to (e.g. .env), if any.
            keys (List[str], optional): List of keys to write down. If none is provided, will write down all.
            override (bool, optional): Whether to override existing env vars if they have already been previously set.
                (Default: False)
        """
        secrets = self.secrets
        keys = keys or self.env_vars

        new_secret = copy.deepcopy(self)
        new_secret.env_vars = keys

        if python_env:
            for key in keys:
                os.environ[key] = secrets[key]

        if path or not python_env:
            from dotenv import dotenv_values

            path = path or self.path
            full_path = os.path.expanduser(path)

            # TODO: handle duplicates / overriding if file exists
            existing_keys = dotenv_values(full_path).keys()
            with open(full_path, "a+") as f:
                for key in keys:
                    if key not in existing_keys:
                        f.write(f"\n{key}={secrets[key]}")
                    elif override:
                        pass
            new_secret.path = path

        return new_secret

    def delete(
        self,
        python_env: bool = False,
        path: Union[str, Path] = None,
        keys: List[str] = None,
    ):
        """Delete the environment variables from where they are stored.

        Args:
            python_env (bool, optional): Whether to unset the environment variables from Python os environment.
                (Default: False)
            path (str or Path, optional): File with env variables to delete. If neither python_env nor path is
                provided, the secrets path will be deleted.
            keys (List[str], optional): List of env var keys to delete. If none is provided, all of the env vars
                associated with the secret will be deleted from the path and/or env.
        """
        if python_env:
            to_delete = keys or self.env_vars
            for key in to_delete:
                del os.environ[key]

        if path or not python_env:
            path = path or self.path
            if not keys and os.path.exists(os.path.expanduser(path)):
                os.remove(os.path.expanduser(path))
            # TODO: handle removing specific keys from a file

        if not keys:
            super().delete()
            return self

        new_secret = copy.deepcopy(self)
        for key in keys:
            del new_secret._secrets[key]
        super().delete()
        new_secret.save()
        return new_secret
