import copy
import json
import logging
import os
from pathlib import Path

from typing import Dict, List, Optional, Tuple, Union

import requests

from runhouse.globals import rns_client
from runhouse.resources.blobs.file import File
from runhouse.resources.hardware import _get_cluster_from, Cluster
from runhouse.resources.resource import Resource
from runhouse.resources.secrets.utils import load_config
from runhouse.rns.utils.api import load_resp_content, read_resp_data, ResourceAccess


logger = logging.getLogger(__name__)


class Secret(Resource):
    RESOURCE_TYPE = "secret"

    USER_ENDPOINT = "user/secret"
    GROUP_ENDPOINT = "group/secret"

    DEFAULT_DIR = "~/.rh/secrets"

    def __init__(
        self,
        name: Optional[str],
        secrets: Dict = {},
        path: str = None,
        env_vars: List = None,
        dryrun: bool = False,
        **kwargs,
    ):
        """
        Runhouse Secret object.

        .. note::
            To create a Secret, please use one of the factory methods.
        """
        super().__init__(name=name, dryrun=dryrun)
        self._secrets = secrets
        self.path = path
        self.env_vars = env_vars

        # TODO: if secrets and path, write it down to the path?

    @staticmethod
    def from_config(config: dict, dryrun: bool = False):
        """Create a Secret object from a config dictionary."""
        resource_subtype = config.get("resource_subtype")
        if "provider" in config:
            from runhouse.resources.secrets.provider_secrets.providers import (
                _get_provider_class,
            )

            provider_class = _get_provider_class(config["provider"])
            return provider_class(**config, dryrun=dryrun)
        if resource_subtype == "ProviderSecret":
            from .provider_secrets.provider_secret import ProviderSecret

            return ProviderSecret(**config, dryrun=dryrun)
        elif resource_subtype == "EnvSecret":
            from .env_secret import EnvSecret

            return EnvSecret(**config, dryrun=dryrun)

        return Secret(**config, dryrun=dryrun)

    @classmethod
    def from_name(cls, name, dryrun=False):
        """Load existing Secret via its name."""
        config = load_config(name, cls.USER_ENDPOINT)
        config["name"] = name
        return cls.from_config(config=config, dryrun=dryrun)

    @property
    def config_for_rns(self):
        config = super().config_for_rns
        config.update(
            {
                "secrets": self._secrets,
                "path": self.path,
                "env_vars": self.env_vars,
            }
        )
        return config

    @property
    def secrets(self):
        """
        Extract secrets key-value pairs from the Secret object.
        The order of operations for retrieving the secrets:

        - Secrets values if they were provided upon object instantiation
        - Extracted from the Secret path, if exists locally
        - Extracted from environment variables
        """
        if self._secrets:
            return self._secrets
        if self.path:
            secrets = self._from_path(self.path)
            if secrets:
                return secrets
        try:
            return self._from_env()
        except KeyError:
            return {}

    # TODO: would be nice to add support for different format types here -- json, yaml [configparser]
    # and same for write function
    def _from_path(self, path: Optional[str] = None):
        path = path or self.path or f"{self.DEFAULT_DIR}/{self.name}.json"
        if isinstance(path, File):
            try:
                secrets = json.loads(path.fetch(mode="r"))
            except json.decoder.JSONDecodeError as e:
                logger.error(
                    f"Error loading config from {path.path} on {path.system.name}: {e}"
                )
                return {}
            return secrets
        elif path and os.path.exists(os.path.expanduser(path)):
            with open(os.path.expanduser(path), "r") as f:
                try:
                    secrets = json.load(f)
                except json.decoder.JSONDecodeError as e:
                    logger.error(f"Error loading config from {path}: {e}")
                    return {}
            return secrets
        return {}

    # TODO: still need to add support for env path
    def _from_env(self, keys: List = None, env_vars: Dict = None):
        secrets = {}
        keys = keys or self.env_vars.keys()
        env_vars = env_vars or self.env_vars or {key: key for key in keys}

        if not keys:
            return {}

        for key in keys:
            secrets[key] = os.environ[env_vars[key]]
        return secrets

    # TODO: refactor this code to reuse rns_client save_config code instead of rewriting
    def save(self, secrets: bool = None):
        """Save the secret config, into Vault if the user is logged in,
        or to local if not or if the resource is a local resource.

        Args:
            secrets (bool, optional): Whether to save the secret values into the config.
                By default, will save secrets only if the Secret was explicitly constructed
                with secrets values passed in. If set to True, will extract secrets values
                (from path, env, etc) and save them in the config. If set to False, will
                not save any secrets values, even if constructed with secrets values passed.
        """
        config = self.config_for_rns
        config["name"] = self.rns_address
        if secrets and not config["secrets"]:
            config["secrets"] = self.secrets
        elif secrets is False:
            config["secrets"] = {}

        if self.rns_address.startswith("/"):
            logger.info(f"Saving config for {self.name} to Vault")
            payload = rns_client.resource_request_payload(config)
            # resource_uri = rns_client.resource_uri(self.rns_address)
            resp = requests.put(
                f"{rns_client.api_server_url}/{self.USER_ENDPOINT}/{self.name}",
                data=json.dumps(payload),
                headers=rns_client.request_headers,
            )
            if resp.status_code != 200:
                raise Exception(
                    f"Failed to upload secrets in Vault: {load_resp_content(resp)}"
                )
        else:
            config_path = os.path.expanduser(f"~/.rh/secrets/{self.name}.json")
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, "w") as f:
                json.dump(config, f, indent=4)
            logger.info(f"Saving config for {self.rns_address} to: {config_path}")

        return self

    def write(
        self,
        path: Union[str, Path] = None,
        keys: List[str] = None,
    ):
        """Write the secrets to local filepath.

        Args:
            path (Path or str, optional): Path to write down the secret to. If not provided, defaults
                to the secrets path variable (if exists), or to a default location in the Runhouse directory.
            keys (List[str], optional): List of keys corresponding to the secrets to write down.
                If left empty, all secrets will be written down.

        Returns:
            Secret object consisting of the given keys at the path.

        Example:
            >>> secret.write()  # writes down secrets to secret.path

            >>> # writes down api_key key-value pair to "new/secrets/file"
            >>> secret.write(path="new/secrets/file", keys="api_key")
        """
        path = path or self.path or f"{self.DEFAULT_DIR}/{self.name}.json"
        if not path:
            raise Exception(
                f"Secret {self.name} was not constructed with a path. "
                "Please pass in a path to this function to save/write down"
                "the secret locally."
            )

        secrets = {key: self.secrets[key] for key in keys} if keys else self.secrets

        full_path = os.path.expanduser(path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "w") as f:
            json.dump(secrets, f)

        if not self.path:
            self.path = path
            return self

        new_secret = copy.deepcopy(self)
        new_secret._secrets = secrets
        new_secret.path = path
        return new_secret

    def write_env(
        self,
        path: str = None,
        keys: List[str] = None,
        env_vars: Dict = None,
    ):
        """Write down the env into the path if provided, otherwise into Python os environment.

        Args:
            path (str, optional): The path to write down the env variables to (such as .env file path).
                If none is provided, secrets are saved into the Python os environment variables instead
                of a file.
            keys (List[str], optional): The keys corresponding to the secrets to write down.
            env_vars (Dict, optional): The mapping of secret key to the corresponding environment
                variable name.

        Returns:
            Secret with the given keys and path, if provided.

        Example:
            >>> secret.write_env(path="secret.env")
            >>> secret.write_env(keys="api_key", env_vars={"api_key": "MY_API_KEY"})
        """
        pass

    def delete_file(
        self,
        path: Union[str, Path, File] = None,
    ):
        """Delete the secrets file.

        Args:
            path (str, optional): Path to delete the secrets file from. If none is provided,
                deletes the path corresponding to the secret class.

        Example:
            >>> secret.delete_file()
        """
        path = path or self.path
        if isinstance(path, File):
            path.rm()
        else:
            os.remove(os.path.expanduser(path))

    def _delete_local_config(self):
        config_path = f"~/.rh/secrets/{self.name}.json"
        os.remove(os.path.expanduser(config_path))

    def _delete_vault_config(self):
        # resource_uri = rns_client.resource_uri(self.rns_address)
        resp = requests.delete(
            f"{rns_client.api_server_url}/{self.USER_ENDPOINT}/{self.name}",
            headers=rns_client.request_headers,
        )
        if resp.status_code != 200:
            logger.error(
                f"Failed to delete secret {self.name} from Vault: {load_resp_content(resp)}"
            )

    def delete(self, file: bool = False):
        """Delete the secret config from Vault/local. Optionally also delete secrets file.

        Args:
            file (bool): Whether to also delete the file containing secrets values. (Default: False)
        """
        if self.rns_address.startswith("/"):
            self._delete_vault_config()
        else:
            self._delete_local_config()
        if file:
            self.delete_file()

    # TODO: handle env -- vars or file
    def to(
        self,
        system: Union[str, Cluster],
        path: Union[str, Path] = None,
    ):
        """Return a copy of the secret on a system.

        Args:
            system (str or Cluster): Cluster to send the secrets to
            path (str or Path): path on cluster to write down the secrets to. If not provided,
                secrets are not written down.

        Example:
            >>> secret.to(my_cluster, path=secret.path)
        """
        system = _get_cluster_from(system)
        key = system.put_resource(self)

        new_secret = copy.deepcopy(self)

        if path:
            from runhouse.resources.blobs.file import file

            if self._secrets:
                system.call(key, "write", path)
                remote_file = file(path=path, system=system)
            else:
                remote_file = file(path=self.path).to(system, path=path)
            new_secret.path = remote_file
        return new_secret

    def is_local(self):
        """Whether the secret config is stored locally (as opposed to Vault)."""
        path = os.path.expanduser(f"~/.rh/secrets/{self.name}.json")
        if os.path.exists(os.path.expanduser(path)):
            return True
        return False

    def in_vault(self):
        """Whether the secret is stored in Vault"""
        resp = requests.get(
            f"{rns_client.api_server_url}/{self.USER_ENDPOINT}/{self.name}",
            headers=rns_client.request_headers,
        )
        if read_resp_data(resp):
            return True
        return False

    def is_enabled(self):
        """Whether the secret is enabled locally."""
        path = self.path or f"{self.DEFAULT_DIR}/{self.name}.json"
        if os.path.exists(os.path.expanduser(path)):
            return True
        return False

    def share(
        self,
        users: list,
        access_type: Union[ResourceAccess, str] = ResourceAccess.READ,
        notify_users: bool = True,
        headers: Optional[Dict] = None,
    ) -> Tuple[Dict[str, ResourceAccess], Dict[str, ResourceAccess]]:
        pass
