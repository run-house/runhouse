import copy
import json
import logging
import os
from pathlib import Path

from typing import Dict, List, Optional, Union

import requests

from runhouse.globals import configs, rns_client
from runhouse.resources.hardware import _get_cluster_from, Cluster
from runhouse.resources.resource import Resource
from runhouse.resources.secrets.utils import load_config
from runhouse.rns.utils.api import load_resp_content, read_resp_data
from runhouse.rns.utils.names import _generate_default_name


logger = logging.getLogger(__name__)


class Secret(Resource):
    RESOURCE_TYPE = "secret"

    USER_ENDPOINT = "user/secret"
    GROUP_ENDPOINT = "group/secret"

    DEFAULT_DIR = "~/.rh/secrets"

    def __init__(
        self,
        name: Optional[str],
        values: Dict = None,
        dryrun: bool = False,
        **kwargs,
    ):
        """
        Runhouse Secret object.

        .. note::
            To create a Secret, please use one of the factory methods.
        """
        super().__init__(name=name, dryrun=dryrun)
        self._values = values

    @property
    def values(self):
        return self._values

    @staticmethod
    def from_config(config: dict, dryrun: bool = False):
        """Create a Secret object from a config dictionary."""
        if "provider" in config:
            from runhouse.resources.secrets.provider_secrets.providers import (
                _get_provider_class,
            )

            provider_class = _get_provider_class(config["provider"])
            return provider_class(**config, dryrun=dryrun)
        return Secret(**config, dryrun=dryrun)

    @property
    def config_for_rns(self):
        config = super().config_for_rns
        if self._values:
            config.update(
                {
                    "values": self._values,
                }
            )
        return config

    @classmethod
    def from_name(cls, name, dryrun=False):
        """Load existing Secret via its name."""
        config = load_config(name, cls.USER_ENDPOINT)
        config["name"] = name
        return cls.from_config(config=config, dryrun=dryrun)

    @classmethod
    def builtin_providers(cls, as_str: bool = False) -> list:
        """Return list of all Runhouse providers (as class objects) supported out of the box."""
        from runhouse.resources.secrets.provider_secrets.providers import (
            _str_to_provider_class,
        )

        if as_str:
            return list(_str_to_provider_class.keys())
        return list(_str_to_provider_class.values())

    @classmethod
    def vault_secrets(
        cls, names: List[str] = None, headers: str = rns_client.request_headers
    ) -> Dict[str, "Secret"]:
        from runhouse.resources.secrets import provider_secret, Secret, secret

        resp = requests.get(
            f"{rns_client.api_server_url}/user/secret",
            headers=headers,
        )

        if resp.status_code != 200:
            raise Exception("Failed to download secrets from Vault")

        secrets = {}
        response = read_resp_data(resp)
        if names is not None:
            response = {name: response[name] for name in names if name in response}
        for name, config in response.items():
            if config.get("name", None):
                if config.get("data", None):
                    config.update(config["data"])
                    del config["data"]
                secrets[name] = Secret.from_config(config)
            else:
                # handle converting previous type of secrets saving format to new resource format
                if name in cls.builtin_providers():
                    new_secret = provider_secret(provider=name, values=config)
                else:
                    new_secret = secret(name=name, values=config)

                secrets[name] = new_secret
                new_secret._delete_vault_config()
                new_secret.save()

        return secrets

    @classmethod
    def local_secrets(cls, names: List[str] = None) -> Dict[str, "Secret"]:
        if not os.path.exists(os.path.expanduser("~/.rh/secrets")):
            return {}

        all_names = [
            Path(file).stem
            for file in os.listdir(os.path.expanduser("~/.rh/secrets"))
            if file.endswith("json")
        ]
        names = [name for name in names if name in all_names] if names else all_names

        secrets = {}
        for name in names:
            path = os.path.expanduser(f"~/.rh/secrets/{name}.json")
            with open(path, "r") as f:
                config = json.load(f)
            if config["name"].startswith("~") or config["name"].startswith("^"):
                config["name"] = config["name"][2:]
            secrets[name] = Secret.from_config(config)
        return secrets

    @classmethod
    def extract_provider_secrets(cls, names: List[str] = None) -> Dict[str, "Secret"]:
        from runhouse.resources.secrets.provider_secrets.providers import (
            _str_to_provider_class,
        )
        from runhouse.resources.secrets.secret_factory import provider_secret

        secrets = {}

        # locally configured non-ssh provider secrets
        for provider in _str_to_provider_class.keys():
            if provider == "ssh":
                continue
            try:
                secret = provider_secret(provider=provider)
                secrets[provider] = secret
            except ValueError:
                continue

        # locally configured ssh secrets
        default_ssh_folder = "~/.ssh"
        ssh_files = os.listdir(os.path.expanduser(default_ssh_folder))
        for file in ssh_files:
            if file != "sky-key" and f"{file}.pub" in ssh_files:
                name = f"ssh-{file}"
                secret = provider_secret(
                    provider="ssh",
                    name=name,
                    path=os.path.join(default_ssh_folder, file),
                )
                secrets[name] = secret

        return secrets

    # TODO: refactor this code to reuse rns_client save_config code instead of rewriting
    def save(self, values: bool = True, headers: str = rns_client.request_headers):
        """
        Save the secret config, into Vault if the user is logged in,
        or to local if not or if the resource is a local resource.
        """
        config = self.config_for_rns
        config["name"] = self.rns_address
        if values:
            config["values"] = self.values

        if self.rns_address.startswith("/"):
            logger.info(f"Saving config for {self.name} to Vault")
            payload = rns_client.resource_request_payload(config)
            resp = requests.put(
                f"{rns_client.api_server_url}/{self.USER_ENDPOINT}/{self.name}",
                data=json.dumps(payload),
                headers=headers,
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

    def delete(self, headers: str = rns_client.request_headers):
        """Delete the secret config from Vault/local."""
        if not self.in_vault() or self.is_local():
            logger.warning(
                "Can not delete a secret that has not been saved down to Vault or local."
            )

        if self.rns_address.startswith("/"):
            self._delete_vault_config(headers)
        else:
            self._delete_local_config()
        configs.delete_provider(self.name)

    def _delete_local_config(self):
        config_path = os.path.expanduser(f"~/.rh/secrets/{self.name}.json")
        if os.path.exists(config_path):
            os.remove(config_path)

    def _delete_vault_config(self, headers: str = rns_client.request_headers):
        resp = requests.delete(
            f"{rns_client.api_server_url}/{self.USER_ENDPOINT}/{self.name}",
            headers=headers,
        )
        if resp.status_code != 200:
            logger.error(
                f"Failed to delete secret {self.name} from Vault: {load_resp_content(resp)}"
            )

    def to(
        self,
        system: Union[str, Cluster],
        name: Optional[str] = None,
    ):
        """Return a copy of the secret on a system.

        Args:
            system (str or Cluster): Cluster to send the secret to
            name (str, ooptional): Name to assign the resource on the cluster.

        Example:
            >>> secret.to(my_cluster, path=secret.path)
        """

        new_secret = copy.deepcopy(self)
        new_secret.name = name or self.name or _generate_default_name(prefix="secret")

        system = _get_cluster_from(system)
        if system.on_this_cluster():
            new_secret.pin()
        else:
            system.put_resource(new_secret)

        return new_secret

    def is_local(self):
        """Whether the secret config is stored locally (as opposed to Vault)."""
        path = os.path.expanduser(f"~/.rh/secrets/{self.name}.json")
        if os.path.exists(os.path.expanduser(path)):
            return True
        return False

    def in_vault(self, headers=rns_client.request_headers):
        """Whether the secret is stored in Vault"""
        resp = requests.get(
            f"{rns_client.api_server_url}/{self.USER_ENDPOINT}/{self.name}",
            headers=headers,
        )
        if resp.status_code != 200:
            return False
        response = read_resp_data(resp)
        if response and response[self.name]:
            return True
        return False

    def is_present(self):
        if self.values:
            return True
        return False
