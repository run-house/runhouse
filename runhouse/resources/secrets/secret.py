import copy
import json
import logging
import os

from typing import Dict, Optional, Union

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
        return Secret(**config, dryrun=dryrun)

    @property
    def config_for_rns(self):
        config = super().config_for_rns
        config.update(
            {
                "values": self.values,
            }
        )
        return config

    @classmethod
    def from_name(cls, name, dryrun=False):
        """Load existing Secret via its name."""
        config = load_config(name, cls.USER_ENDPOINT)
        config["name"] = name
        return cls.from_config(config=config, dryrun=dryrun)

    # TODO: refactor this code to reuse rns_client save_config code instead of rewriting
    def save(self, headers: str = rns_client.request_headers):
        """
        Save the secret config, into Vault if the user is logged in,
        or to local if not or if the resource is a local resource.
        """
        config = self.config_for_rns
        config["name"] = self.rns_address

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
        """Delete the secret config from Vault/local. Optionally also delete secret file."""
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

    # Q: is the way we send .to(cluster) secure?
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
        if read_resp_data(resp)[self.name]:
            return True
        return False

    def is_present(self):
        if self.values:
            return True
        return False

    def share(self):
        pass
