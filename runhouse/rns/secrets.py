import json
import logging
from typing import Dict, List, Optional

import requests

import typer

from runhouse.globals import rns_client

from runhouse.resources.secrets import provider_secret, Secret
from runhouse.rns.utils.api import load_resp_content, read_resp_data

logger = logging.getLogger(__name__)


class Secrets:
    """WARNING: This class is DEPRECATED. Please use the new ``rh.Secret`` resource APIs instead."""

    @classmethod
    def extract_and_upload(
        cls,
        headers: Optional[Dict] = None,
        interactive=False,
        providers: Optional[List[str]] = None,
    ):
        """Upload all locally configured secrets into Vault. Secrets are loaded from their local config files.
        (ex: ``~/.aws/credentials)``. To upload custom secrets for custom providers, see Secrets.put()

        Example:
            >>> rh.Secrets.extract_and_upload(providers=["aws", "lambda"])
        """
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
        """Get all user secrets from Vault. Optionally save them down to local config files (where relevant).

        Example:
            >>> rh.Secrets.download_into_env(providers=["aws", "lambda"])
        """
        logger.info("Getting secrets from Vault.")
        resp = requests.get(
            f"{rns_client.api_server_url}/{cls.USER_ENDPOINT}",
            headers=headers or rns_client.request_headers,
        )
        if resp.status_code != 200:
            raise Exception("Failed to download secrets from Vault")

        secrets = read_resp_data(resp)

        if providers is not None:
            secrets = {p: secrets[p] for p in providers if p in secrets}

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
        logger.warning(
            "``rh.Secrets`` class is deprecated. Please use the ``rh.Secret`` APIs instead."
        )

        secret_resource = provider_secret(
            provider=provider, values=secret, path=file_path
        )
        secret_resource.save()

    @classmethod
    def get(
        cls, provider: str, save_to_env: bool = False, group: Optional[str] = None
    ) -> dict:
        logger.warning(
            "``rh.Secrets`` class is deprecated. Please use the ``rh.Secret`` APIs instead."
        )

        if provider not in Secret.vault_secrets():
            raise Exception(f"Failed to get secrets from Vault for {provider}")

        secret = Secret.from_name(provider)
        if save_to_env:
            secret.write()
        return secret.values

    @classmethod
    def load_provider_secrets(
        cls, from_env: bool = False, providers: Optional[List] = None
    ) -> Dict[str, Dict]:
        """ """
        logger.warning(
            "``rh.Secrets`` class is deprecated. Please use the ``rh.Secret.extract_provider_secrets()`` API instead."
        )

        extracted_secrets = Secret.extract_provider_secrets(names=providers)
        provider_secrets = {}
        for (name, resource) in extracted_secrets.items():
            provider_secrets[name] = resource.values
        return provider_secrets

    @classmethod
    def save_provider_secrets(cls, secrets: dict, check_enabled=True):
        logger.warning(
            "``rh.Secrets`` class is deprecated. Please use the ``rh.Secret`` APIs instead."
        )
        for provider_name, provider_secrets in secrets.items():
            secret = provider_secret(provider=provider_name, values=provider_secrets)
            secret.save()

    @classmethod
    def builtin_providers(cls, as_str: bool = False) -> list:
        logger.warning(
            "``rh.Secrets`` class is deprecated. Please use ``rh.Secret.builtin_providers()`` instead."
        )
        return Secret.builtin_providers(as_str=as_str)
