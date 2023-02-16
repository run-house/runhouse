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
    def save_secrets(
        cls, secrets: Dict, file_path: Optional[str] = None, overwrite: bool = False
    ) -> Dict:
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
        interactive=True,
        providers: Optional[List[str]] = None,
    ):
        """Upload all locally configured secrets into Vault. Secrets are loaded from their local config files.
        (ex: ~/.aws/credentials). To upload custom secrets for custom providers, see Secrets.put()"""
        secrets: list = cls.load_provider_secrets(providers=providers)
        for idx, provider_secrets in enumerate(secrets):
            provider = provider_secrets["provider"]
            if interactive:
                upload_secrets = typer.confirm(f"Upload secrets for {provider}?")
                if not upload_secrets:
                    secrets.pop(idx)

        resp = requests.put(
            f"{rns_client.api_server_url}/{cls.USER_ENDPOINT}",
            data=json.dumps(secrets),
            headers=headers or rns_client.request_headers,
        )
        if resp.status_code != 200:
            raise Exception(
                f"Failed to update secrets in Vault: {json.loads(resp.content)}"
            )
        found_providers = [secret.get("provider") for secret in secrets]
        logging.info(f"Uploaded secrets for providers {found_providers} to Vault")

    @classmethod
    def download_into_env(
        cls,
        save_locally: bool = True,
        providers: Optional[List[str]] = None,
        headers: Optional[Dict] = None,
    ) -> Dict:
        """Get all user secrets from Vault. Optionally save them down to local config files (where relevant)."""
        logger.info("Getting secrets from Vault.")
        resp = requests.get(
            f"{rns_client.api_server_url}/{cls.USER_ENDPOINT}",
            headers=headers or rns_client.request_headers,
        )
        if resp.status_code != 200:
            raise Exception("Failed to download secrets from Vault")

        secrets = cls.load_resp_data(resp)
        if providers is not None:
            secrets = {p: secrets[p] for p in providers}
        if save_locally and secrets:
            cls.save_provider_secrets(secrets)
            logger.info(
                f"Saved secrets from Vault to local config files for providers: {list(secrets)}"
            )
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
    ):
        """Upload locally configured secrets for a specified provider into Vault.
        To upload secrets for a custom provider (i.e. not AWS, GCP or Azure), include the secret param and specify
        the keys and values to upload.
        If from_env is True, will read secrets from environment variables instead of local config files.
        If file_path is provided, will read the secrets directly from the file
        If group is provided, will attribute the secrets to the specified group"""
        provider_name = provider.lower()
        if provider_name in cls.builtin_providers(as_str=True):
            # if a supported cloud provider is given, use the provider's built-in class
            p = cls.builtin_provider_class_from_name(provider_name)
            if p is not None:
                secret = p.read_secrets(from_env=from_env, file_path=file_path)

        if not secret and not isinstance(secret, dict):
            raise Exception(f"No secrets dict found or provided for {provider}")

        endpoint = cls.set_endpoint(group)
        resp = requests.put(
            f"{rns_client.api_server_url}/{endpoint}/{provider}",
            data=json.dumps(secret),
            headers=rns_client.request_headers,
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
            raise Exception(f"Failed to get secrets from Vault for {provider_name}")

        secrets = cls.load_resp_data(resp).get(provider_name, {})
        if not secrets:
            logger.error(f"Failed to load secrets for {provider_name}")
            return {}

        p = cls.builtin_provider_class_from_name(provider_name)

        if save_to_env and p is not None:
            logger.info(f"Saving secrets for {provider_name} to local config")
            p.save_secrets(secrets)

        return secrets

    @classmethod
    def delete_from_vault(cls, providers: Optional[List[str]] = None):
        """Delete secrets from Vault for specified providers (builtin or custom).
        If none are provided, will delete secrets for all providers which have been enabled in the local environment."""
        providers = providers or cls.builtin_providers(as_str=True)
        for provider in providers:
            url = f"{rns_client.api_server_url}/{cls.USER_ENDPOINT}/{provider}"
            resp = requests.delete(url, headers=rns_client.request_headers)
            if resp.status_code != 200:
                logger.error(
                    f"Failed to delete secrets from Vault: {json.loads(resp.content)}"
                )

    @classmethod
    def to(cls, hardware: Union[str, "Cluster"], providers: Optional[List] = None):
        """Copy secrets to the desired hardware for a list of builtin providers. If no providers are specified
        will load all builtin providers that are already enabled."""
        if isinstance(hardware, str):
            from runhouse import cluster

            hardware = cluster(name=hardware)

        hardware_name = hardware.name
        if not hardware.is_up():
            raise RuntimeError(
                f"Hardware {hardware_name} is not up. Run `hardware_obj.up()` to re-up the cluster."
            )

        configured_providers: list = cls.configured_providers(as_str=True)
        provider_secrets: list = cls.load_provider_secrets(providers=providers)
        if not provider_secrets or len(provider_secrets) < len(configured_providers):
            # If no secrets found in local config files or secrets are missing for some configured providers,
            # check if they exist in Vault
            vault_secrets: dict = cls.download_into_env(save_locally=False)
            # TODO [JL] change this API so we don't have to convert the list to a dict?
            provider_secrets: list = [
                {"provider": k, **v} for k, v in vault_secrets.items()
            ]
            missing_providers = list(set(configured_providers) - set(provider_secrets))
            if missing_providers:
                raise Exception(
                    f"Failed to find secrets for providers: {missing_providers}"
                )

        # Send provider secrets over RPC to the cluster, then save each provider's secrets into their default
        # file paths on the cluster
        failed_secrets: dict = json.loads(hardware.add_secrets(provider_secrets))
        if failed_secrets:
            logger.warning(
                f"Failed to copy some secrets to cluster {hardware_name}: {failed_secrets}"
            )
        else:
            logger.info(f"Finished copying secrets onto cluster {hardware_name}")

    @classmethod
    def load_provider_secrets(
        cls, from_env: bool = False, providers: Optional[List] = None
    ) -> List[Dict[str, str]]:
        """Load secret credentials for all the providers which have been configured locally, or optionally
        provide a list of specific providers to load."""
        secrets = []
        providers = providers or cls.configured_providers()
        for provider in providers:
            if isinstance(provider, str):
                provider = cls.builtin_provider_class_from_name(provider)
                if not provider:
                    continue

            if not from_env and not provider.has_secrets_file():
                # no secrets file configured for this provider
                continue

            configs.set(provider.PROVIDER_NAME, provider.default_credentials_path())
            provider_secrets = provider.read_secrets(from_env=from_env)
            if provider_secrets:
                secrets.append(provider_secrets)

        return secrets

    @classmethod
    def save_provider_secrets(cls, secrets: dict):
        """Save secrets for each provider to their respective local configs"""
        # configured_providers = cls.configured_providers(as_str=True) # TODO
        configured_providers = ["aws"]
        for provider_name, provider_data in secrets.items():
            if provider_name not in configured_providers:
                logger.warning(
                    f"Received secrets for {provider_name} which are not configured locally. Run `sky check`"
                    f" for instructions on how to configure. If the secret is for a custom provider, you "
                    f"can set the relevant environment variables or save them to their respective local files manually."
                )
                continue

            provider_cls = cls.builtin_provider_class_from_name(provider_name)
            if provider_cls is not None:
                try:
                    provider_cls.save_secrets(provider_data, overwrite=True)
                except Exception as e:
                    logger.error(
                        f"Failed to save {provider_name} secrets to local config: {e}"
                    )
                    continue

            # Make sure local config reflects this provider has been enabled
            configs.set(provider_name, provider_cls.default_credentials_path())

    @classmethod
    def builtin_providers(cls, as_str: bool = False) -> List:
        """Returns a list of cloud provider class objects which Runhouse supports out of the box.
        If as_str is True, return the names of the providers as strings"""
        sky.check.check(quiet=True)
        clouds = sky.global_user_state.get_enabled_clouds()
        cloud_names = [str(c).lower() for c in clouds]
        if "local" in cloud_names:
            cloud_names.remove("local")

        cloud_names.append("sky")

        # Check if the huggingface_hub package is installed
        try:
            import huggingface_hub  # noqa

            cloud_names.append("huggingface")
        except ModuleNotFoundError:
            pass

        if as_str:
            return [
                c.PROVIDER_NAME
                for c in cls.__subclasses__()
                if c.PROVIDER_NAME in cloud_names
            ]

        return [c for c in cls.__subclasses__() if c.PROVIDER_NAME in cloud_names]

    @classmethod
    def configured_providers(cls, as_str: bool = False) -> List:
        """Return list of builtin providers which have been configured in the local filesystem."""
        configured_providers = [
            p for p in cls.builtin_providers() if p.has_secrets_file()
        ]
        if as_str:
            return [c.PROVIDER_NAME for c in configured_providers]
        return configured_providers

    @classmethod
    def check_secrets_for_mismatches(cls, secrets_to_save: dict, file_path: str):
        """When overwrite is set to `False` check if new secrets clash with what may have already been saved in
        the Vault or existing file."""
        existing_secrets: dict = cls.read_secrets(file_path=file_path)
        existing_secrets.pop("provider", None)
        for existing_key, existing_val in existing_secrets.items():
            new_val = secrets_to_save.get(existing_key)
            if existing_key != new_val:
                raise ValueError(
                    f"Mismatch in secrets for key `{existing_key}`! Secrets in config file {file_path} "
                    f"do not match those provided. If you intend to overwrite a particular secret key, "
                    f"please manually rename or remove it from the secrets file"
                )

    @staticmethod
    def delete_secrets_file(file_path: str):
        Path(file_path).unlink(missing_ok=True)

    @classmethod
    def save_secret_to_config(cls):
        """Save the loaded provider config path to the runhouse config saved in the file system."""
        configs.set(cls.PROVIDER_NAME, cls.default_credentials_path())

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
    def load_resp_data(resp) -> Dict:
        return json.loads(resp.content).get("data", {})

    @staticmethod
    def file_exists(file_path: str) -> bool:
        if not Path(file_path).exists():
            return False
        return True


# TODO AWS secrets (use https://github.com/99designs/aws-vault ?)
# TODO Azure secrets
# TODO GCP secrets
# TODO custom vault secrets
