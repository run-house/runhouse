import logging

import typer

from runhouse.rh_config import configs, rns_client
from .secrets import Secrets

logger = logging.getLogger(__name__)


def is_interactive():
    import __main__ as main

    return not hasattr(main, "__file__")


def login(
    token: str = None,
    download_config: bool = None,
    upload_config: bool = None,
    download_secrets: bool = None,
    upload_secrets: bool = None,
    ret_token: bool = False,
    interactive: bool = False,
):
    """Login to Runhouse. Validates token provided, with options to upload or download stored secrets or config between
    local environment and Runhouse / Vault.
    """
    if is_interactive() or interactive:
        from getpass import getpass

        from rich.console import Console

        console = Console()
        console.print(
            """
            ____              __                             @ @ @
           / __ \__  ______  / /_  ____  __  __________     []___
          / /_/ / / / / __ \/ __ \/ __ \/ / / / ___/ _ \   /    /\____    @@
         / _, _/ /_/ / / / / / / / /_/ / /_/ (__  )  __/  /_/\_//____/\  @@@@
        /_/ |_|\__,_/_/ /_/_/ /_/\____/\__,_/____/\___/   | || |||__|||   ||
        """
        )
        link = (
            f'[link={configs.get("api_server_url")}/dashboard/?option=token]https://api.run.house[/link]'
            if is_interactive()
            else f'{configs.get("api_server_url")}/dashboard/?option=token'
        )
        console.print(
            f"Retrieve your token :key: here to use :person_running: :house: Runhouse for "
            f"secrets and artifact management: {link}",
            style="bold yellow",
        )
        if not token:
            token = getpass("Token: ")

        download_config = (
            download_config
            if download_config is not None
            else typer.confirm(
                "Download config from Runhouse to your local .rh folder?"
            )
        )
        download_secrets = (
            download_secrets
            if download_secrets is not None
            else typer.confirm(
                "Download secrets from Vault to your local Runhouse config?"
            )
        )
        upload_config = (
            upload_config
            if upload_config is not None
            else typer.confirm("Upload your local config to Runhouse?")
        )
        upload_secrets = (
            upload_secrets
            if upload_secrets is not None
            else typer.confirm("Upload your enabled cloud provider secrets to Vault?")
        )

    if token:
        configs.set("token", token)

    if download_config:
        configs.download_and_save_defaults()
        # We need to fresh the RNSClient to use the newly loaded configs
        rns_client.refresh_defaults()
    elif upload_config:
        configs.upload_defaults(defaults=configs.defaults_cache)
    else:
        # If we are not downloading or uploading config, we still want to make sure the token is valid
        try:
            configs.download_defaults()
        except:
            logger.error("Failed to validate token")
            return None

    if download_secrets:
        Secrets.download_into_env()

    if upload_secrets:
        Secrets.extract_and_upload(interactive=interactive)

    logger.info("Successfully logged into Runhouse.")
    if ret_token:
        return token


def logout(
    remove_from_rh_config: bool = True,
    delete_credentials_file: bool = False,
    delete_rh_config_file: bool = False,
    interactive: bool = False,
):
    """Logout from Runhouse. Provides option to delete credentials from the Runhouse config and the underlying
     credentials file. Token is also deleted from the config.

    Args:
        remove_from_rh_config (bool, optional): If True, removes the provider from the rh config. Defaults to True.
        delete_credentials_file (bool, optional): If True, deletes the provider credentials file. Defaults to False.
        delete_rh_config_file (bool, optional): If True, deletes the rh config file. Defaults to False.
        interactive (bool, optional): If True, runs the logout process in interactive mode. Defaults to False.

    Returns:
        None
    """
    interactive_session = is_interactive() or interactive
    configured_providers = Secrets.configured_providers()
    for provider in configured_providers:
        provider_name = provider.PROVIDER_NAME
        provider_creds_path: str = provider.credentials_path()

        if interactive_session:
            remove_from_rh_config = typer.confirm(
                f"Remove secrets for {provider_name} from Runhouse config?"
            )

            delete_credentials_file = typer.confirm(
                f"Delete credentials file for {provider_name}?"
            )

        if remove_from_rh_config and configs.get(provider_name):
            configs.delete(provider_name)
            logger.info(f"Removed {provider_name} from Runhouse config")

        if delete_credentials_file:
            provider.delete_secrets_file(provider_creds_path)
            logger.info(
                f"Deleted {provider_name} credentials file from path: {provider_creds_path}"
            )

    # Delete token from config
    # TODO [JL] invalidate the token stored in RNS?
    configs.delete(key="token")

    rh_config_path = configs.CONFIG_PATH
    if interactive_session:
        delete_rh_config_file = typer.confirm(
            f"Delete your local Runhouse config file?"
        )

    if delete_rh_config_file:
        # Delete the credentials file on the file system
        Secrets.delete_secrets_file(rh_config_path)
        logger.info(f"Deleted Runhouse config file from path: {rh_config_path}")

    logger.info("Successfully logged out of Runhouse.")
