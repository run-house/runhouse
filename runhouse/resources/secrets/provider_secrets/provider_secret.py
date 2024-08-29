import copy
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

from runhouse.globals import configs, rns_client
from runhouse.resources.envs.env import Env
from runhouse.resources.hardware.cluster import Cluster
from runhouse.resources.hardware.utils import _get_cluster_from
from runhouse.resources.secrets.secret import Secret
from runhouse.resources.secrets.utils import _check_file_for_mismatches
from runhouse.utils import create_local_dir


class ProviderSecret(Secret):
    _PROVIDER = None
    _DEFAULT_CREDENTIALS_PATH = None
    _DEFAULT_ENV_VARS = {}

    def __init__(
        self,
        name: Optional[str] = None,
        provider: Optional[str] = None,
        values: Dict = None,
        path: str = None,
        env_vars: Dict = None,
        dryrun: bool = False,
        **kwargs,
    ):
        """
        Provider Secret class. Built-in provider classes contain default path and/or environment variable mappings,
        based on it's expected usage.

        Currently supported built-in providers:
        anthropic, aws, azure, gcp, github, huggingface, lambda, langchain, openai, pinecone, ssh, sky, wandb.

        .. note::
            To create a ProviderSecret, please use the factory method :func:`provider_secret`.
        """
        super().__init__(name=name, values=values, dryrun=dryrun)
        self.provider = provider or self._PROVIDER
        self.path = path
        self.env_vars = env_vars

        if not any([values, path, env_vars]):
            if self._from_path(self._DEFAULT_CREDENTIALS_PATH):
                self.path = self._DEFAULT_CREDENTIALS_PATH
            elif self._from_env(self._DEFAULT_ENV_VARS):
                self.env_vars = self._DEFAULT_ENV_VARS
            else:
                raise ValueError(
                    "Secrets values not provided and could not be extracted from default file "
                    f"({self._DEFAULT_CREDENTIALS_PATH}) or env vars ({self._DEFAULT_ENV_VARS.values()}) locations."
                )

    @property
    def values(self):
        if self._values:
            return self._values
        elif self.path:
            return self._from_path(self.path)
        elif self.env_vars:
            return self._from_env(self.env_vars)
        return {}

    def config(self, condensed=True):
        config = super().config(condensed)
        config.update({"provider": self.provider})
        if self.path:
            config.update({"path": self.path})
        if self.env_vars:
            config.update({"env_vars": self.env_vars})
        return config

    @staticmethod
    def from_config(config: dict, dryrun: bool = False, _resolve_children: bool = True):
        """Create a ProviderSecret object from a config dictionary."""
        return ProviderSecret(**config, dryrun=dryrun)

    def save(
        self,
        name: str = None,
        save_values: bool = True,
        headers: Optional[Dict] = None,
        folder: str = None,
    ):
        name = name or self.name or self.rns_address or self.provider
        return super().save(
            name=name, save_values=save_values, headers=headers, folder=folder
        )

    def delete(self, headers: Optional[Dict] = None, contents: bool = False):
        """Delete the secret config from Den and from Vault/local. Optionally also delete contents of secret file
        or env vars."""
        headers = headers or rns_client.request_headers()
        if self.path and contents and os.path.exists(os.path.expanduser(self.path)):
            os.remove(os.path.expanduser(self.path))
        elif self.env_vars and contents:
            for (_, env_var) in self.env_vars.keys():
                if env_var in os.environ:
                    del os.environ[env_var]
        super().delete(headers=headers)

    def write(
        self,
        path: str = None,
        env_vars: Dict = None,
        file: bool = False,
        env: bool = False,
        overwrite: bool = False,
    ):
        if not self.values:
            raise ValueError("Could not determine values to write down.")
        if (file or path) and (env or env_vars):
            raise ValueError("Can only save to one of file or env at a given time.")
        if not any([file, env, path, env_vars]):
            file = True  # default write to file

        if file or path:
            path = path or self.path or self._DEFAULT_CREDENTIALS_PATH
            return self._write_to_file(path, values=self.values, overwrite=overwrite)
        elif env or env_vars:
            env_vars = env_vars or self.env_vars or self._DEFAULT_ENV_VARS
            return self._write_to_env(env_vars, values=self.values, overwrite=overwrite)

    def to(
        self,
        system: Union[str, Cluster],
        path: str = None,
        env: Union[str, Env] = None,
        values: bool = None,
        name: Optional[str] = None,
    ):
        """Return a copy of the secret on a system.

        Args:
            system (str or Cluster): Cluster to send the secret to
            path (str or Path, optional): Path on cluster to write down the secret values to.
                If not provided and secret is not already associated with a path, the secret values
                will not be written down on the cluster.
            env (str or Env, optional): Env to send the secret to. This will save down the secrets
                as env vars in the env.
            values (bool, optional): Whether to save down the values in the resource config. By default,
                save down values if the secret is not being written down to a file or environment variable.
                Otherwise, values are not written down. (Default: None)
            name (str, ooptional): Name to assign the resource on the cluster.

        Example:
            >>> secret.to(my_cluster, path=secret.path)
        """
        from runhouse import Env

        system = _get_cluster_from(system)
        path = path or self.path

        if not env or (env and isinstance(env, Env) and not env.name):
            env = system.default_env

        if system.on_this_cluster():
            if not env and not path == self.path:
                if name and not self.name == name:
                    self.rename(name)
                return self
            self.write(path=path, env=env)
            new_secret = copy.deepcopy(self)
            new_secret._values = None
            new_secret.path = path
            new_secret.name = name or self.name
            return new_secret

        new_secret = copy.deepcopy(self)
        new_secret.name = name or self.name or self.provider

        if values:
            new_secret._values = self.values
        elif values is None and not (path or env or self.env_vars):
            new_secret._values = self.values
        elif values is False:
            new_secret._values = None

        key = system.put_resource(new_secret, env=env)
        if path:
            new_secret.path = self._file_to(
                key=key, system=system, path=path, values=self.values
            )

        if env or self.env_vars:
            env_key = env if isinstance(env, str) else env.name
            if not system.get(env_key):
                env = env if isinstance(env, Env) else Env(name=env_key)
                env_key = system.put_resource(env)
            env_vars = self.env_vars or self._DEFAULT_ENV_VARS
            if env_vars:
                env_vars = {env_vars[k]: self.values[k] for k in self.values}
                system.call(env_key, "_set_env_vars", env_vars)
        return new_secret

    def _file_to(
        self,
        key: str,
        system: Union[str, Cluster],
        path: str = None,
        values: Any = None,
    ):
        system.call(key, "_write_to_file", path=path, values=values)
        return path

    def _write_to_file(self, path: str, values: Any, overwrite: bool = False):
        new_secret = copy.deepcopy(self)
        if not _check_file_for_mismatches(
            path, self._from_path(path), values, overwrite
        ):
            full_path = create_local_dir(path)
            with open(full_path, "w") as f:
                json.dump(values, f, indent=4)
            self._add_to_rh_config(path)

        new_secret._values = None
        new_secret.path = path
        return new_secret

    def _write_to_env(self, env_vars: Dict, values: Any, overwrite: bool):
        existing_keys = dict(os.environ).keys()
        added_keys = []
        for key in env_vars.keys():
            if env_vars[key] not in existing_keys or overwrite:
                os.environ[env_vars[key]] = values[key]
                added_keys.append(env_vars[key])

        if added_keys:
            self._add_to_rh_config(added_keys)

        new_secret = copy.deepcopy(self)
        new_secret._values = None
        new_secret.env_vars = env_vars
        return new_secret

    def _from_env(self, env_vars: Dict = None):
        env_vars = env_vars or self.env_vars
        if not env_vars:
            return {}

        values = {}

        for key in env_vars.keys():
            try:
                values[key] = os.environ[env_vars[key]]
            except KeyError:
                return {}
        return values

    def _from_path(self, path: str = None):
        path = path or self.path
        if not path:
            return ""

        path = os.path.expanduser(path)
        if os.path.exists(path):
            with open(path) as f:
                try:
                    contents = json.load(f)
                except json.decoder.JSONDecodeError:
                    contents = f.read()
                return contents
        return {}

    @staticmethod
    def extract_secrets_from_path(path: str) -> Union[str, None]:
        secret_path = os.path.expanduser(path)

        if not os.path.exists(secret_path):
            return None

        provider_secret = Path(secret_path).read_text()

        return provider_secret

    def _add_to_rh_config(self, val):
        if not self.name:
            self.name = self.provider
        configs.set_nested(key="secrets", value={self.name: val})
