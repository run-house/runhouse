import configparser
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import requests

import sky
import typer
import yaml

from runhouse.rh_config import configs, rns_client
from runhouse.rns.api_utils.utils import load_resp_content, read_resp_data

logger = logging.getLogger(__name__)


class Secrets:
    """Handles cluster secrets management (reading and writing) across all major cloud providers.
    Secrets are stored in Vault.
    Checks for locally configured secrets before pulling down or saving to Vault"""

    PROVIDER_NAME = None
    CREDENTIALS_FILE = None

    USER_ENDPOINT = "user/secret"
    GROUP_ENDPOINT = "group/secret"

    def __str__(self):
        return str(self.__class__.__name__)

    @classmethod
    def read_secrets(
        cls, from_env: bool = False, file_path: Optional[str] = None
    ) -> Dict:
        raise NotImplementedError

    @classmethod
    def save_secrets(cls, secrets: Dict, overwrite: bool = False) -> Dict:
        """Save secrets for providers to their respective configs. If overwrite is set to `False` will check for
        potential clashes with existing secrets that may already be saved, otherwise will overwrite whatever exists."""
        raise NotImplementedError

    @classmethod
    def has_secrets_file(cls) -> bool:
        file_path = cls.default_credentials_path()
        if not file_path:
            return False
        return cls.file_exists(file_path)

    @classmethod
    def default_credentials_path(cls):
        return cls.CREDENTIALS_FILE

    @classmethod
    def extract_and_upload(
        cls,
        headers: Optional[Dict] = None,
        interactive=False,
        providers: Optional[List[str]] = None,
    ):
        """Upload all locally configured secrets into Vault. Secrets are loaded from their local config files.
        (ex: ~/.aws/credentials). To upload custom secrets for custom providers, see Secrets.put()"""
        secrets: dict = cls.load_provider_secrets(providers=providers)
        for provider_name, provider_secrets in secrets.items():
            if interactive:
                upload_secrets = typer.confirm(f"Upload secrets for {provider_name}?")
                if not upload_secrets:
                    secrets.pop(provider_name, None)

        resp = requests.put(
            f"{rns_client.api_server_url}/{cls.USER_ENDPOINT}",
            data=json.dumps(secrets),
            headers=headers or rns_client.request_headers,
        )
        if resp.status_code != 200:
            raise Exception(
                f"Failed to update secrets in Vault: {load_resp_content(resp)}"
            )

        logging.info(f"Uploaded secrets for to Vault for: {list(secrets)}")

    @classmethod
    def download_into_env(
        cls,
        save_locally: bool = True,
        providers: Optional[List[str]] = None,
        headers: Optional[Dict] = None,
        check_enabled: bool = True,
    ) -> Dict:
        """Get all user secrets from Vault. Optionally save them down to local config files (where relevant)."""
        logger.info("Getting secrets from Vault.")
        resp = requests.get(
            f"{rns_client.api_server_url}/{cls.USER_ENDPOINT}",
            headers=headers or rns_client.request_headers,
        )
        if resp.status_code != 200:
            raise Exception("Failed to download secrets from Vault")

        secrets = read_resp_data(resp)

        if providers is not None:
            secrets = {p: secrets[p] for p in providers}

        if save_locally and secrets:
            cls.save_provider_secrets(secrets, check_enabled=check_enabled)
            logger.info("Saved secrets from Vault to local config files")
        else:
            return secrets

    @classmethod
    def put(
        cls,
        provider: str,
        from_env: bool = False,
        file_path: Optional[str] = None,
        secret: Optional[dict] = None,
        group: Optional[str] = None,
        headers: Optional[dict] = None,
    ):
        """Upload locally configured secrets for a specified provider into Vault.
        To upload secrets for a custom provider (i.e. not AWS, GCP or Azure), include the secret param and specify
        the keys and values to upload.
        If from_env is True, will read secrets from environment variables instead of local config files.
        If file_path is provided, will read the secrets directly from the file
        If group is provided, will attribute the secrets to the specified group"""
        provider_name = provider.lower()
        if not secret and provider_name in cls.enabled_providers(as_str=True):
            # if a supported cloud provider is given and no secret is provided, extract it from its default location
            p = cls.builtin_provider_class_from_name(provider_name)
            if p is not None:
                secret = p.read_secrets(from_env=from_env, file_path=file_path)

        if not secret and not isinstance(secret, dict):
            raise Exception(
                f"No secrets dict found or provided for {provider}. Please make sure the credentials "
                f"file exists in its default location, or provide credentials with the `secret` param"
            )

        endpoint = cls.set_endpoint(group)
        resp = requests.put(
            f"{rns_client.api_server_url}/{endpoint}/{provider}",
            data=json.dumps(secret),
            headers=headers or rns_client.request_headers,
        )
        if resp.status_code != 200:
            raise Exception(f"Failed to update {provider} secrets in Vault")

    @classmethod
    def get(
        cls, provider: str, save_to_env: bool = False, group: Optional[str] = None
    ) -> dict:
        """Read secrets from the Vault service for a given provider and optionally save them to their local config.
        If group is provided will read secrets for the specified group."""
        provider_name = provider.lower()
        endpoint = cls.set_endpoint(group)
        url = f"{rns_client.api_server_url}/{endpoint}/{provider_name}"
        resp = requests.get(url, headers=rns_client.request_headers)
        if resp.status_code != 200:
            raise Exception(
                f"Failed to get secrets from Vault for {provider_name}: {load_resp_content(resp)}"
            )

        secrets = read_resp_data(resp).get(provider_name, {})
        if not secrets:
            logger.info(f"No secrets found in Vault for {provider_name}")
            return {}

        p = cls.builtin_provider_class_from_name(provider_name)

        if save_to_env and p is not None:
            logger.info(f"Saving secrets for {provider_name} to local config")
            p.save_secrets(secrets)

        return secrets

    @classmethod
    def to(cls, hardware: Union[str, "Cluster"], providers: Optional[List] = None):
        """Copy secrets to the desired hardware for a list of builtin providers. If no providers are specified
        will load all builtin providers that are already enabled."""
        if isinstance(hardware, str):
            from runhouse import Cluster

            hardware = Cluster.from_name(name=hardware)

        hardware_name = hardware.name
        if not hardware.is_up():
            raise RuntimeError(
                f"Hardware {hardware_name} is not up. Run `hardware_obj.up()` to re-up the cluster."
            )

        enabled_providers: list = cls.enabled_providers(as_str=True)

        # Extract secrets from default paths
        configured_secrets: dict = cls.load_provider_secrets(providers=providers)

        if not configured_secrets or len(configured_secrets) < len(enabled_providers):
            # If no secrets found in the enabled providers' credentials files check if they exist in Vault
            missing_providers = list(
                set(enabled_providers) - set(list(configured_secrets))
            )
            secrets_for_missing_providers: dict = cls.download_into_env(
                save_locally=False, providers=missing_providers
            )

            # Add the missing provider secrets from Vault to the configured secrets
            configured_secrets.update(secrets_for_missing_providers)

            # Confirm all enabled providers are either configured locally or have secrets stored in Vault
            if len(configured_secrets) < len(enabled_providers):
                raise Exception(
                    f"Failed to find secrets locally or in Vault for providers: {missing_providers}. "
                    f"For enabling locally save the secrets to the provider's default credentials file, "
                    f"or upload the secrets directly to Vault (e.g: `rh.Secrets.put({missing_providers[0]})`)"
                )

        # Send provider secrets over RPC to the cluster, then save each provider's secrets into their default
        # file paths on the cluster
        failed_to_add_secrets: dict = hardware.add_secrets(configured_secrets)
        if len(failed_to_add_secrets) == len(configured_secrets):
            raise RuntimeError(
                f"Failed to copy all secrets onto the {hardware_name} cluster: {failed_to_add_secrets}"
            )
        elif failed_to_add_secrets:
            logger.warning(
                f"Failed to copy some secrets onto the {hardware_name} cluster: {failed_to_add_secrets}"
            )
        else:
            logger.info(
                f"Finished copying all secrets onto the {hardware_name} cluster"
            )

    @classmethod
    def update(cls, provider: str, secrets: dict):
        """Add new keys to existing secrets saved for a given provider in Vault."""
        existing_secrets = cls.get(provider=provider)
        if existing_secrets:
            existing_secrets.update(secrets)
            cls.put(provider, secret=existing_secrets)

    @classmethod
    def delete_from_local_env(cls, providers: Optional[List[str]] = None):
        """Delete secrets credential files and use in Runhouse configs for list of specified providers.
        If none are provided, will delete secrets for all providers which have been enabled in the local environment."""
        providers = providers or cls.enabled_providers(as_str=True)
        for provider in providers:
            p = cls.builtin_provider_class_from_name(provider)
            if p is not None:
                # Use the default credentials path defined in the builtin provider's class
                creds_file_path = p.default_credentials_path()
            else:
                # See if we have the provider's path saved in the rh config
                creds_file_path = configs.get("secrets", {}).get(provider)
                if creds_file_path is None:
                    logger.warning(
                        f"Unable to delete credentials file for {provider}. Please delete the file manually."
                    )
                    continue

            configs.delete(provider)

            # Delete the local creds file
            cls.delete_secrets_file(creds_file_path)

    @classmethod
    def delete_from_vault(cls, providers: Optional[List[str]] = None):
        """Delete secrets from Vault for specified providers.
        If none are provided, will delete secrets for all providers which have been enabled in the local environment."""
        providers = providers or cls.enabled_providers(as_str=True)
        for provider in providers:
            url = f"{rns_client.api_server_url}/{cls.USER_ENDPOINT}/{provider}"
            resp = requests.delete(url, headers=rns_client.request_headers)
            if resp.status_code != 200:
                logger.error(
                    f"Failed to delete secrets from Vault: {load_resp_content(resp)}"
                )

    @classmethod
    def load_provider_secrets(
        cls, from_env: bool = False, providers: Optional[List] = None
    ) -> Dict[str, Dict]:
        """Load secret credentials for all the providers which have been configured locally, or optionally
        provide a list of specific providers to load. Returns a dictionary with provider name as the key and
        secrets dictionary as value."""
        secrets = {}
        providers = providers or cls.enabled_providers()
        for provider in providers:
            if isinstance(provider, str):
                provider = cls.builtin_provider_class_from_name(provider)
                if not provider:
                    continue

            if not from_env and not provider.has_secrets_file():
                # no secrets file configured for this provider
                continue

            provider_secrets = provider.read_secrets(from_env=from_env)
            if provider_secrets:
                secrets[provider.PROVIDER_NAME] = provider_secrets

        return secrets

    @classmethod
    def save_provider_secrets(cls, secrets: dict, check_enabled=True):
        """Save secrets for each provider to their respective local configs"""
        for provider_name, provider_secrets in secrets.items():
            provider_cls = cls.builtin_provider_class_from_name(provider_name)
            if provider_cls is not None:
                try:
                    provider_cls.save_secrets(provider_secrets, overwrite=True)
                except Exception as e:
                    logger.error(
                        f"Failed to save {provider_name} secrets to config: {e}"
                    )
                    continue

        if check_enabled:
            enabled_providers = cls.enabled_providers(as_str=True)
            not_enabled = [
                p
                for p in secrets.keys()
                if p not in enabled_providers
                and p in cls.builtin_providers(as_str=True)
            ]
            if not_enabled:
                logger.warning(
                    f"Received secrets {not_enabled} which Runhouse did not auto-detect as configured. "
                    f"For cloud providers, you may want to run `sky check` to double check that they're "
                    f"enabled and to see instructions on how to enable them."
                )

    @classmethod
    def enabled_providers(cls, as_str: bool = False) -> List:
        """Returns a list of cloud provider classes which Runhouse supports out of the box.
        If as_str is True, return the names of the providers as strings"""
        sky.check.check(quiet=True)
        clouds = sky.global_user_state.get_enabled_clouds()
        cloud_names = [str(c).lower() for c in clouds]
        if "local" in cloud_names:
            cloud_names.remove("local")

        cloud_names.append("sky")

        try:
            import huggingface_hub  # noqa

            if configs.get("huggingface"):
                cloud_names.append("huggingface")
        except ModuleNotFoundError:
            pass

        # Add any SSH keys + GitHub token that were explicitly added
        config_secrets = configs.get("secrets", {})
        if config_secrets.get("ssh"):
            cloud_names.append("ssh")

        if config_secrets.get("github"):
            cloud_names.append("github")

        if as_str:
            return cloud_names

        return [cls.builtin_provider_class_from_name(c) for c in cloud_names]

    @classmethod
    def builtin_providers(cls, as_str=False) -> list:
        """Return list of all Runhouse providers (as class objects) supported out of the box."""
        from runhouse.rns.secrets.providers import Providers

        if as_str:
            return [e.name.lower() for e in Providers]
        return [e.value for e in Providers]

    @classmethod
    def check_secrets_for_mismatches(
        cls, secrets_to_save: dict, secrets_path: str, overwrite: bool
    ):
        """When overwrite is set to `False` and a secrets file already exists, check if new secrets clash with
        what may have already been saved."""
        if overwrite or not cls.has_secrets_file():
            # If explicitly overwriting or the secrets file does not exist we can ignore
            return

        existing_secrets: dict = cls.read_secrets(file_path=secrets_path)
        provider = existing_secrets.pop("provider", None)

        for existing_key, existing_val in existing_secrets.items():
            new_val = secrets_to_save.get(existing_key)
            if existing_key != new_val:
                raise ValueError(
                    f"Mismatch in {provider} secrets for key `{existing_key}`! Secrets in config file {secrets_path} "
                    f"do not match those provided. If you intend to overwrite a particular secret key, "
                    f"please do so manually."
                )

    @classmethod
    def delete_secrets_file(cls, file_path: Union[str, tuple] = None):
        """Delete local credentials file. If no path is provided will use the default path set for the provider."""
        file_path = file_path or cls.default_credentials_path()
        if isinstance(file_path, str):
            Path(file_path).unlink(missing_ok=True)
        if isinstance(file_path, tuple):
            for f in file_path:
                Secrets.delete_secrets_file(file_path=f)

    @classmethod
    def add_provider_to_rh_config(cls, secrets_for_config: Optional[dict] = None):
        """Save the loaded provider config path to the runhouse config saved in the file system."""
        config_secrets = secrets_for_config or {
            cls.PROVIDER_NAME: cls.default_credentials_path()
        }
        configs.set_nested(key="secrets", value=config_secrets)

    @classmethod
    def set_endpoint(cls, group: Optional[str] = None):
        return (
            f"{cls.GROUP_ENDPOINT}/{group}" if group is not None else cls.USER_ENDPOINT
        )

    @staticmethod
    def save_to_config_file(parser, file_path: str):
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w+") as f:
            parser.write(f)

    @staticmethod
    def save_to_json_file(data: dict, file_path: str):
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w+") as f:
            json.dump(data, f)

    @staticmethod
    def read_json_file(file_path: str) -> Dict:
        with open(file_path, "r") as config_file:
            config_data = json.load(config_file)
        return config_data

    @staticmethod
    def read_config_file(file_path: str):
        config = configparser.ConfigParser()
        config.read(file_path)
        return config

    @staticmethod
    def read_yaml_file(file_path: str):
        with open(file_path, "r") as stream:
            config = yaml.safe_load(stream)
        return config

    @staticmethod
    def save_to_yaml_file(data, file_path):
        with open(file_path, "w") as yaml_file:
            yaml.dump(data, yaml_file, default_flow_style=False)

    @staticmethod
    def builtin_provider_class_from_name(name: str):
        try:
            from runhouse.rns.secrets.providers import Providers

            return Providers[name.upper()].value
        except:
            # could be a custom provider, in which case there is no built-in class
            return None

    @staticmethod
    def file_exists(file_path: str) -> bool:
        if not Path(file_path).exists():
            return False
        return True


# TODO AWS secrets (use https://github.com/99designs/aws-vault ?)
# TODO Azure secrets
# TODO GCP secrets
# TODO custom vault secrets
