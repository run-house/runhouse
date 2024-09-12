import copy
import json
import os
from pathlib import Path

from typing import Dict, List, Optional, Union

from runhouse.globals import configs, rns_client
from runhouse.logger import get_logger

from runhouse.resources.hardware import _get_cluster_from, Cluster
from runhouse.resources.resource import Resource
from runhouse.resources.secrets.utils import _delete_vault_secrets, load_config
from runhouse.rns.utils.api import load_resp_content, read_resp_data
from runhouse.utils import generate_default_name

logger = get_logger(__name__)


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

    def config(self, condensed=True):
        config = super().config(condensed)
        if self._values:
            config.update(
                {
                    "values": self._values,
                }
            )
        return config

    @staticmethod
    def _write_shared_secret_to_local(config):
        import runhouse as rh

        new_creds_values = config["values"]
        folder_name = config["name"].replace("/", "_")
        path = f"{Secret.DEFAULT_DIR}/{folder_name}"
        private_key_value, public_key_value = new_creds_values.get(
            "private_key"
        ), new_creds_values.get("public_key")
        private_key_path = public_key_path = Path(f"{path}").expanduser()
        if private_key_value:
            if not private_key_path.exists():
                os.makedirs(str(private_key_path))
            private_file_path = private_key_path / "ssh-key"
            with open(str(private_file_path), "w") as f:
                f.write(private_key_value)
            private_file_path.chmod(0o600)
        if public_key_value:
            public_file_path = public_key_path / "ssh-key.pub"
            with open(str(public_key_path / "ssh-key.pub"), "w") as f:
                f.write(public_key_value)
            public_file_path.chmod(0o600)
        if private_key_value and public_key_value:
            new_creds_values = {
                "ssh_private_key": str(private_key_path / "ssh-key"),
                "ssh_public_key": str(public_key_path / "ssh-key.pub"),
            }
        if private_key_value and new_creds_values.get("ssh_user"):
            new_creds_values = {
                "ssh_private_key": str(private_key_path / "ssh-key"),
                "ssh_user": new_creds_values.get("ssh_user"),
            }
        return rh.secret(
            values=new_creds_values, name=f"loaded_secret_{config['name']}"
        )

    @staticmethod
    def from_config(config: dict, dryrun: bool = False, _resolve_children: bool = True):
        if "provider" in config:
            from runhouse.resources.secrets.provider_secrets.providers import (
                _get_provider_class,
            )

            provider_class = _get_provider_class(config["provider"])
            return provider_class.from_config(config, dryrun=dryrun)

        # checks if the config is a of a shared secret
        current_user = configs.username
        owner_user = config["owner"]["username"] if "owner" in config.keys() else None

        if owner_user and current_user != owner_user and config["values"]:
            return Secret._write_shared_secret_to_local(config)

        return Secret(**config, dryrun=dryrun)

    @classmethod
    def from_name(
        cls,
        name,
        load_from_den: bool = True,
        dryrun: bool = False,
        _alt_options: Dict = None,
        _resolve_children: bool = True,
    ):
        try:
            config = load_config(name, cls.USER_ENDPOINT)
            if config:
                return cls.from_config(config=config, dryrun=dryrun)
        except ValueError:
            pass
        if name in cls.builtin_providers(as_str=True):
            from runhouse.resources.secrets.provider_secrets.providers import (
                _get_provider_class,
            )

            provider_class = _get_provider_class(name)
            return provider_class(provider=name, dryrun=dryrun)
        raise ValueError(f"Could not locate secret {name}")

    @classmethod
    def builtin_providers(cls, as_str: bool = False) -> List:
        """Return list of all Runhouse providers (as class objects) supported out of the box.

        Args:
            as_str (bool, optional): Whether to return the providers as a string or as a class.
                (Default: ``False``)
        """
        from runhouse.resources.secrets.provider_secrets.providers import (
            _str_to_provider_class,
        )

        if as_str:
            return list(_str_to_provider_class.keys())
        return list(_str_to_provider_class.values())

    @classmethod
    def vault_secrets(cls, headers: Optional[Dict] = None) -> List[str]:
        """Get secret names that are stored in Vault"""
        uri = f"{rns_client.api_server_url}/{cls.USER_ENDPOINT}"
        resp = rns_client.session.get(
            uri,
            headers=headers or rns_client.request_headers(),
        )

        if resp.status_code not in [200, 404]:
            raise Exception(
                f"Received [{resp.status_code}] from Den GET '{uri}': Failed to download secrets from Vault."
            )

        response = read_resp_data(resp)
        return list(response.keys())

    @classmethod
    def local_secrets(cls, names: List[str] = None) -> Dict[str, "Secret"]:
        """Get list of local secrets.

        Args:
            names (List[str], optional): Specific names of local secrets to retrieve. If ``None``, returns all
                locally detected secrets. (Default: ``None``)
        """
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
            try:
                with open(path, "r") as f:
                    config = json.load(f)
                if config["name"].startswith("~") or config["name"].startswith("^"):
                    config["name"] = config["name"][2:]
                secrets[name] = Secret.from_config(config)
            except json.JSONDecodeError:
                # Ignore any empty / corrupted files
                continue

        return secrets

    @classmethod
    def extract_provider_secrets(cls, names: List[str] = None) -> Dict[str, "Secret"]:
        """Extract secret values from providers. Returns a Dict mapping the provider name to Secret.

        Args:
            names (List[str]): List of provider names to extract secrets for. If ``None``, returns
                secrets for all detected providers. (Default: ``None``)
        """
        from runhouse.resources.secrets.provider_secrets.providers import (
            _str_to_provider_class,
        )
        from runhouse.resources.secrets.secret_factory import provider_secret

        secrets = {}

        names = names or _str_to_provider_class.keys()
        for provider in names:
            if provider == "ssh":
                continue
            try:
                secret = provider_secret(provider=provider)
                secrets[provider] = secret
            except ValueError:
                continue

        # locally configured ssh secrets
        if "ssh" in names:
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
    def save(
        self,
        name: str = None,
        save_values: bool = True,
        headers: Optional[Dict] = None,
        folder: str = None,
    ):
        """
        Save the secret config to Den. Save the secret values into Vault if the user is logged in,
        or to local if not or if the resource is a local resource.

        Args:
            name (str, optional): Name to save the secret resource as.
            save_values (str, optional): Whether to save the values of the secret to Vault in addition
                to saving the metadata to Den. (Default: ``True``)
            headers (Dict, optional): Request headers to provide for the request to RNS. Contains the
                user's auth token. Example: ``{"Authorization": f"Bearer {token}"}`` (Default: ``None``)
            folder (str, optional): If specified, save the secret to that folder in Den (e.g. saving secrets
                for a cluster associated with an organization). (Default: ``None``)
        """
        if name:
            self.name = name
        elif not self.name:
            raise ValueError("A resource must have a name to be saved.")

        self._rns_folder = folder or self._rns_folder or rns_client.current_folder

        config = self.config()
        config["name"] = self.rns_address
        if "values" in config:
            # don't save values into Den config
            del config["values"]

        headers = headers or rns_client.request_headers()

        # Save metadata to Den
        if self.rns_address.startswith("/"):
            logger.info(f"Saving config for {self.name} to Den")
            payload = rns_client.resource_request_payload(config)
            uri = f"{rns_client.api_server_url}/resource"
            resp = rns_client.session.post(
                uri,
                data=json.dumps(payload),
                headers=headers,
            )

            # If resource config hasn't changed (i.e. nothing to update) will return a 422
            if resp.status_code not in [200, 422]:
                raise Exception(
                    f"Received [{resp.status_code}] from Den POST '{uri}': Failed to save metadata to Den: {load_resp_content(resp)}"
                )

            if save_values and self.values:
                logger.info(f"Saving secrets for {self.name} to Vault")
                resource_uri = rns_client.resource_uri(self.rns_address)
                uri = f"{rns_client.api_server_url}/{self.USER_ENDPOINT}/{resource_uri}"
                resp = rns_client.session.put(
                    uri,
                    data=json.dumps(
                        {"name": self.rns_address, "data": {"values": self.values}}
                    ),
                    headers=headers,
                )
                if resp.status_code != 200:
                    raise Exception(
                        f"Received [{resp.status_code}] from Den PUT '{uri}': Failed to put resources in Vault: {load_resp_content(resp)}"
                    )

        else:
            config_path = os.path.expanduser(f"~/.rh/secrets/{self.name}.json")
            os.makedirs(os.path.dirname(config_path), exist_ok=True)

            if save_values:
                config["values"] = self.values

            with open(config_path, "w") as f:
                json.dump(config, f, indent=4)
            logger.info(f"Saving config for {self.rns_address} to: {config_path}")

        return self

    def delete(self, headers: Optional[Dict] = None):
        """Delete the secret config from Den and from Vault/local."""
        if not (self.in_vault() or self.in_local()):
            logger.warning(
                "Can not delete a secret that has not been saved down to Vault or local."
            )

        else:
            if self.rns_address and self.rns_address.startswith("/"):
                self._delete_secret_configs(headers)
            else:
                self._delete_local_config()
        configs.delete_provider(self.name)

    def _delete_local_config(self):
        config_path = os.path.expanduser(f"~/.rh/secrets/{self.name}.json")
        if os.path.exists(config_path):
            os.remove(config_path)

    def _delete_secret_configs(self, headers: Optional[Dict] = None):
        headers = headers or rns_client.request_headers()

        # Delete secrets in Vault
        resource_uri = rns_client.resource_uri(self.rns_address)
        _delete_vault_secrets(resource_uri, self.USER_ENDPOINT, headers=headers)

        # Delete RNS data for resource
        uri = f"{rns_client.api_server_url}/resource/{resource_uri}"
        resp = rns_client.session.delete(
            uri,
            headers=headers,
        )
        if resp.status_code != 200:
            logger.error(
                f"Received [{resp.status_code}] from Den DELETE '{uri}': Failed to delete secret resource from Den: {load_resp_content(resp)}"
            )

    def to(
        self,
        system: Union[str, Cluster],
        name: Optional[str] = None,
        env: Optional["Env"] = None,
    ):
        """Return a copy of the secret on a system.

        Args:
            system (str or Cluster): Cluster to send the secret to
            name (str, optional): Name to assign the resource on the cluster.
            env (Env, optional): Env to send the secret to.

        Example:
            >>> secret.to(my_cluster, path=secret.path)
        """

        new_secret = copy.deepcopy(self)
        new_secret.name = name or self.name or generate_default_name(prefix="secret")

        system = _get_cluster_from(system)
        if system.on_this_cluster():
            new_secret.pin()
        else:
            env = env or system.default_env
            system.put_resource(new_secret, env=env)

        return new_secret

    def in_local(self):
        """Whether the secret config is stored locally (as opposed to Vault)."""
        path = os.path.expanduser(f"~/.rh/secrets/{self.name}.json")
        if os.path.exists(os.path.expanduser(path)):
            return True
        return False

    def in_vault(self, headers=None):
        """Whether the secret is stored in Vault"""
        if not self.rns_address:
            return False
        resource_uri = rns_client.resource_uri(self.rns_address)
        resp = rns_client.session.get(
            f"{rns_client.api_server_url}/{self.USER_ENDPOINT}/{resource_uri}",
            headers=headers or rns_client.request_headers(),
        )
        if resp.status_code != 200:
            return False
        response = read_resp_data(resp)
        # TODO: switch this to use self.name once vault updates
        if response and response[list(response.keys())[0]]:
            return True
        return False

    def is_present(self):
        if self.values:
            return True
        return False
