import configparser
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

import requests

import sky
import typer
import yaml

from runhouse.rh_config import rns_client

logger = logging.getLogger(__name__)


class Secrets:
    """Handles cluster secrets management (reading and writing) across all major cloud providers.
    Secrets are stored in Vault.
    Checks for locally configured secrets before pulling down or saving to Vault"""

    PROVIDER_NAME = None
    CREDENTIALS_FILE = None

    USER_ENDPOINT = "user/secret"
    GROUP_ENDPOINT = "group/secret"

    @classmethod
    def read_secrets(
        cls, from_env: bool = False, file_path: Optional[str] = None
    ) -> Dict:
        raise NotImplementedError

    @classmethod
    def save_secrets(cls, secrets: Dict, file_path: Optional[str] = None) -> Dict:
        raise NotImplementedError

    @classmethod
    def has_secrets_file(cls) -> bool:
        if not cls.CREDENTIALS_FILE:
            return False
        return cls.file_exists(cls.CREDENTIALS_FILE)

    @classmethod
    def extract_and_upload(
        cls,
        headers: Optional[Dict] = None,
        interactive=True,
        providers: Optional[List[str]] = None,
    ):
        """Upload all locally configured secrets into Vault. Secrets are loaded from their local config files.
        (ex: ~/.aws/credentials). We currently support AWS, Azure, and GCP. To upload custom secrets for
        additional providers, see Secrets.put()"""
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
        providers: Optional[List] = None,
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
            secrets = {provider: secrets[provider] for provider in providers}
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
        provider = provider.lower()
        if provider in cls.enabled_providers(as_str=True):
            # if a supported cloud provider is given, use the provider's built-in class
            provider_cls_name = cls.provider_cls_name(provider)
            p = cls.get_class_from_name(provider_cls_name)
            if not from_env and not p.has_secrets_file():
                # no secrets file configured for this provider
                raise Exception(f"No local secrets file found for {provider}")

            secret = p.read_secrets(from_env=from_env, file_path=file_path)

        if not secret and not isinstance(secret, dict):
            raise Exception("No secrets dict found or provided for {provider}")

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
        provider = provider.lower()
        endpoint = cls.set_endpoint(group)
        url = f"{rns_client.api_server_url}/{endpoint}/{provider}"
        resp = requests.get(url, headers=rns_client.request_headers)
        if resp.status_code != 200:
            raise Exception(f"Failed to get secrets from Vault for {provider}")

        secrets = cls.load_resp_data(resp).get(provider, {})
        if not secrets:
            raise Exception(f"Failed to load secrets for {provider}")

        cls_name = cls.provider_cls_name(provider)
        p = cls.get_class_from_name(cls_name)

        if save_to_env and p is not None:
            logger.info(f"Saving secrets for {provider} to local config")
            p.save_secrets(secrets)

        return secrets

    @classmethod
    def delete(cls, providers: List[str]):
        """Delete secrets from Vault for the specified providers"""
        for provider in providers:
            provider_cls_name = cls.provider_cls_name(provider)
            p = cls.get_class_from_name(provider_cls_name)
            if p is None:
                continue

            p.delete_secrets_from_vault()
            logger.info(f"Successfully deleted {provider} secrets from Vault")

    @classmethod
    def delete_secrets_from_vault(cls):
        resp = requests.delete(
            f"{rns_client.api_server_url}/{cls.USER_ENDPOINT}/{cls.PROVIDER_NAME}",
            headers=rns_client.request_headers,
        )
        if resp.status_code != 200:
            raise Exception(f"Failed to delete {cls.PROVIDER_NAME} secrets from Vault")

    @classmethod
    def load_provider_secrets(
        cls, from_env: bool = False, providers: Optional[List] = None
    ) -> List[Dict[str, str]]:
        """Load secret credentials for all the providers which have been configured locally, or optionally
        provide a list of specific providers to load"""
        secrets = []
        providers = providers or cls.enabled_providers()
        for provider in providers:
            if isinstance(provider, str):
                provider_cls_name = cls.provider_cls_name(provider)
                provider = cls.get_class_from_name(name=provider_cls_name)
                if not provider:
                    continue

            if not from_env and not provider.has_secrets_file():
                # no secrets file configured for this provider
                continue

            provider_secrets = provider.read_secrets(from_env=from_env)
            if provider_secrets:
                secrets.append(provider_secrets)
        return secrets

    @classmethod
    def save_provider_secrets(cls, secrets: dict):
        """Save secrets for each provider to their respective local configs"""
        for provider_name, provider_data in secrets.items():
            cls_name = cls.provider_cls_name(provider_name)
            provider_cls = cls.get_class_from_name(cls_name)
            # Save secrets to local config
            if provider_cls is not None:
                try:
                    provider_cls.save_secrets(provider_data)
                except Exception as e:
                    logger.error(
                        f"Failed to save {provider_name} secrets to local config: {e}"
                    )
                    continue

        enabled_providers = cls.enabled_providers(as_str=True)
        for provider_name in secrets.keys():
            if provider_name not in enabled_providers:
                logger.warning(
                    f"Received secrets for {provider_name} which are not configured locally. Run `sky check`"
                    f" for instructions on how to configure. If the secret is for a custom provider, you "
                    f"can set the relevant environment variables manually."
                )

    @classmethod
    def enabled_providers(cls, as_str: bool = False) -> List:
        """Returns a list of cloud provider class objects which have been enabled locally. If as_str is True,
        return the names of the providers as strings"""
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
    def get_class_from_name(name: str):
        try:
            return getattr(sys.modules[__name__], name)
        except:
            # could be a custom provider, in which case there is no built-in class
            return None

    @staticmethod
    def provider_cls_name(provider: str):
        return provider.upper() + "Secrets"

    @staticmethod
    def load_resp_data(resp) -> Dict:
        return json.loads(resp.content).get("data", {})

    @staticmethod
    def file_exists(file_path: str) -> bool:
        if not Path(file_path).exists():
            return False
        return True


class AWSSecrets(Secrets):
    PROVIDER_NAME = "aws"
    CREDENTIALS_FILE = os.path.expanduser("~/.aws/credentials")

    @classmethod
    def read_secrets(cls, from_env: bool = False, file_path: Optional[str] = None):
        if from_env:
            access_key = os.getenv("AWS_ACCESS_KEY_ID")
            secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
            if not access_key or not secret_key:
                raise Exception(
                    f"AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY must be set for {cls.PROVIDER_NAME}"
                )
        else:
            if file_path:
                # Read the credentials from the given file path
                config = cls.read_config_file(file_path)
                section_name = "default"
                access_key = config[section_name]["aws_access_key_id"]
                secret_key = config[section_name]["aws_secret_access_key"]
            else:
                # TODO check if user has boto installed, if not tell them to install with runhouse[aws]
                import boto3

                session = boto3.Session()
                credentials = session.get_credentials()

                # Credentials are refreshable, so accessing your access key / secret key
                # separately can lead to a race condition.
                credentials = credentials.get_frozen_credentials()
                access_key = credentials.access_key
                secret_key = credentials.secret_key

        return {
            "provider": cls.PROVIDER_NAME,
            "access_key": access_key,
            "secret_key": secret_key,
        }

    @classmethod
    def save_secrets(cls, secrets: dict, file_path: Optional[str] = None):
        dest_path = file_path or cls.CREDENTIALS_FILE
        parser = configparser.ConfigParser()
        section_name = "default"
        parser.add_section(section_name)
        parser.set(
            section=section_name,
            option="aws_access_key_id",
            value=secrets["access_key"],
        )
        parser.set(
            section=section_name,
            option="aws_secret_access_key",
            value=secrets["secret_key"],
        )

        cls.save_to_config_file(parser, dest_path)


class AZURESecrets(Secrets):
    PROVIDER_NAME = "azure"
    CREDENTIALS_FILE = os.path.expanduser("~/.azure/clouds.config")

    @classmethod
    def read_secrets(cls, from_env: bool = False, file_path: Optional[str] = None):
        if from_env:
            subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")
            if not subscription_id:
                raise Exception(
                    f"AZURE_SUBSCRIPTION_ID must is not set for {cls.PROVIDER_NAME}"
                )
        else:
            creds_file = file_path or cls.CREDENTIALS_FILE
            config = cls.read_config_file(creds_file)
            subscription_id = config["AzureCloud"]["subscription"]

        return {"provider": cls.PROVIDER_NAME, "subscription_id": subscription_id}

    @classmethod
    def save_secrets(cls, secrets: dict, file_path: Optional[str] = None):
        dest_path = file_path or cls.CREDENTIALS_FILE
        parser = configparser.ConfigParser()
        section_name = "AzureCloud"
        parser.add_section(section_name)
        parser.set(
            section=section_name,
            option="subscription",
            value=secrets["subscription_id"],
        )

        cls.save_to_config_file(parser, dest_path)


class GCPSecrets(Secrets):
    PROVIDER_NAME = "gcp"
    CREDENTIALS_FILE = os.path.expanduser(
        "~/.config/gcloud/application_default_credentials.json"
    )

    @classmethod
    def read_secrets(cls, from_env: bool = False, file_path: Optional[str] = None):
        if from_env:
            client_id = os.getenv("CLIENT_ID")
            client_secret = os.getenv("CLIENT_SECRET")
            if not client_id or not client_secret:
                raise Exception(
                    f"CLIENT_ID and CLIENT_SECRET must be set for {cls.PROVIDER_NAME}"
                )
        else:
            creds_file = file_path or cls.CREDENTIALS_FILE
            config_data = cls.read_json_file(creds_file)
            client_id = config_data["client_id"]
            client_secret = config_data["client_secret"]

        return {
            "provider": cls.PROVIDER_NAME,
            "client_id": client_id,
            "client_secret": client_secret,
        }

    @classmethod
    def save_secrets(cls, secrets: dict, file_path: Optional[str] = None):
        dest_path = file_path or cls.CREDENTIALS_FILE
        config = cls.read_json_file(dest_path) if cls.file_exists(dest_path) else {}
        config["client_id"] = secrets["client_id"]
        config["client_secret"] = secrets["client_secret"]

        # We need to do extra stuff if we're in a colab
        if "google.colab" in sys.modules:
            from rich.console import Console

            console = Console()
            console.print(
                "Please do the following to complete gcp secrets setup:",
                style="bold yellow",
            )
            console.print("!gcloud init", style="bold yellow")
            console.print("!gcloud auth application-default login", style="bold yellow")
            console.print(
                "!cp -r /content/.config/* ~/.config/gcloud", style="bold yellow"
            )

        cls.save_to_json_file(config, dest_path)


class HUGGINGFACESecrets(Secrets):
    PROVIDER_NAME = "huggingface"
    CREDENTIALS_FILE = os.path.expanduser("~/.huggingface/token")

    @classmethod
    def read_secrets(cls, from_env: bool = False, file_path: Optional[str] = None):
        if from_env:
            raise NotImplementedError(
                f"Reading secrets from env is not supported for {cls.PROVIDER_NAME}"
            )
        else:
            creds_path = file_path or cls.CREDENTIALS_FILE
            token = Path(creds_path).read_text()

        return {"provider": cls.PROVIDER_NAME, "token": token}

    @classmethod
    def save_secrets(cls, secrets: dict, file_path: Optional[str] = None):
        # TODO check properly if hf needs to be installed
        try:
            import huggingface_hub
        except ModuleNotFoundError:
            subprocess.run(["pip", "install", "--upgrade", "huggingface-hub"])
            import huggingface_hub
        huggingface_hub.login(token=secrets["token"])


class SKYSecrets(Secrets):
    PROVIDER_NAME = "sky"
    PRIVATE_KEY_FILE = os.path.expanduser(sky.authentication.PRIVATE_SSH_KEY_PATH)
    PUBLIC_KEY_FILE = os.path.expanduser(sky.authentication.PUBLIC_SSH_KEY_PATH)

    @classmethod
    def has_secrets_file(cls) -> bool:
        return cls.file_exists(cls.PRIVATE_KEY_FILE) and cls.file_exists(
            cls.PUBLIC_KEY_FILE
        )

    @classmethod
    def read_secrets(cls, from_env: bool = False, file_path: Optional[str] = None):
        if from_env:
            raise NotImplementedError(
                f"Reading secrets from env is not supported for {cls.PROVIDER_NAME}"
            )
        else:
            private_key_file = file_path or cls.PRIVATE_KEY_FILE
            public_key_file = file_path or cls.PUBLIC_KEY_FILE

            private_key = Path(private_key_file).read_text()
            public_key = Path(public_key_file).read_text()

        return {
            "provider": cls.PROVIDER_NAME,
            "ssh_private_key": private_key,
            "ssh_public_key": public_key,
        }

    @classmethod
    def save_secrets(cls, secrets: dict, file_path: Optional[str] = None):
        private_key_path = file_path or cls.PRIVATE_KEY_FILE
        if private_key_path.endswith(".pem"):
            public_key_path = private_key_path.rsplit(".", 1)[0] + ".pub"
        else:
            public_key_path = private_key_path + ".pub"

        sky.authentication._save_key_pair(
            private_key_path,
            public_key_path,
            secrets["ssh_private_key"],
            secrets["ssh_public_key"],
        )
        # TODO do we need to register the keys with cloud providers? Probably not, sky does this for us later.
        # backend_utils._add_auth_to_cluster_config(sky.clouds.CLOUD_REGISTRY.from_str(self.provider),
        #                                                   Path(yaml_path).expanduser())


# TODO [DG] untested, test this
class GHSecrets(Secrets):
    PROVIDER_NAME = "gh"
    CREDENTIALS_FILE = os.path.expanduser("~/.config/gh/hosts.yml")

    @classmethod
    def read_secrets(cls, from_env: bool = False, file_path: Optional[str] = None):
        if from_env:
            raise NotImplementedError(
                f"Reading secrets from env is not supported for {cls.PROVIDER_NAME}"
            )
        else:
            creds_file = file_path or cls.CREDENTIALS_FILE
            config_data = cls.read_yaml_file(creds_file)
            token = config_data["github.com"]["oauth_token"]

        return {"provider": cls.PROVIDER_NAME, "token": token}

    @classmethod
    def save_secrets(cls, secrets: dict, file_path: Optional[str] = None):
        dest_path = file_path or cls.CREDENTIALS_FILE
        config = cls.read_yaml_file(dest_path) if cls.file_exists(dest_path) else {}
        config["github.com"] = {"oauth_token": secrets["token"]}

        cls.save_to_yaml_file(config, dest_path)


# TODO AWS secrets (use https://github.com/99designs/aws-vault ?)
# TODO Azure secrets
# TODO GCP secrets
# TODO custom vault secrets
