import logging
import os

import typer

from runhouse.globals import configs, rns_client
from runhouse.resources.secrets.provider_secrets import SSHSecret
from runhouse.resources.secrets.secret import Secret
from runhouse.resources.secrets.utils import (
    _get_local_secrets_configs,
    _get_vault_secrets,
)

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
    force: bool = None,
    ret_token: bool = False,
    interactive: bool = None,
):
    """Login to Runhouse. Validates token provided, with options to upload or download stored secrets or config between
    local environment and Runhouse / Vault.

    Args:
        token (str): Runhouse token, can be found at https://www.run.house/account#token. If not provided, function will
            interactively prompt for the token to be entered manually.
        download_config (bool): Whether to download configs from your Runhouse account to local environment.
        upload_config (bool): Whether to upload local configs into your Runhouse account.
        download_secrets (bool): Whether to download secrets from your Runhouse account to local environment.
        upload_secrets (bool): Whether to upload local secrets to your Runhouse account.
        ret_token (bool): Whether to return your Runhouse token. (Default: False)
        interactive (bool): Whether to interactively go through the login flow. ``token`` must be provided to
            set this to False.

    Returns:
        Token if ``ret_token`` is set to True, otherwise nothing.
    """
    all_options_set = token and not any(
        arg is None
        for arg in (download_config, upload_config, download_secrets, upload_secrets)
    )

    if interactive is False and not token:
        raise Exception(
            "`interactive` can only be set to `False` if token is provided."
        )

    if interactive or (interactive is None and not all_options_set):
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
        account_url = f"{configs.get('dashboard_url')}/account#token"
        link = (
            f"[link={account_url}]{account_url}[/link]"
            if is_interactive()
            else account_url
        )
        if not token:
            console.print(
                f"Retrieve your token :key: here to use :person_running: :house: Runhouse for "
                f"secrets and artifact management: {link}",
                style="bold yellow",
            )
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
                "Download secrets from Vault to your local Runhouse environment?"
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
            else typer.confirm("Upload your local secrets to Vault?")
        )

    if token:
        configs.set("token", token)

    if download_config:
        configs.download_and_save_defaults()
        # We need to fresh the RNSClient to use the newly loaded configs
        rns_client.refresh_defaults()
    elif upload_config:
        configs.load_defaults_from_file()
        configs.upload_defaults(defaults=configs.defaults_cache)
    else:
        # If we are not downloading or uploading config, we still want to make sure the token is valid
        # and also download the username and default folder
        try:
            defaults = configs.download_defaults()
        except:
            logger.error("Failed to validate token")
            return None
        configs.set("username", defaults["username"])
        configs.set("default_folder", defaults["default_folder"])

    force = force if force is not None else False
    if download_secrets:
        _login_download_secrets(force=force)

    if upload_secrets:
        _login_upload_secrets(force=force)

    logger.info("Successfully logged into Runhouse.")
    if ret_token:
        return token


def logout(
    delete_loaded_secrets: bool = None,
    delete_rh_config_file: bool = None,
    interactive: bool = None,
    force: bool = None,
):
    """
    Logout from Runhouse. Provides option to delete credentials from the Runhouse config and the underlying
    credentials file. Token is also deleted from the config.

    Args:
        delete_loaded_secrets (bool, optional): If True, deletes the provider credentials file. Defaults to None.
        delete_rh_config_file (bool, optional): If True, deletes the rh config file. Defaults to None.
        interactive (bool, optional): If True, runs the logout process in interactive mode. Defaults to None.

    Returns:
        None
    """
    interactive_session: bool = (
        interactive if interactive is not None else is_interactive()
    )

    if interactive_session:
        _logout_secrets()
    else:
        force = force if force is not None else False
        delete_loaded_secrets = (
            delete_loaded_secrets if delete_loaded_secrets is not None else True
        )
        _logout_secrets(force=force, file=delete_loaded_secrets)

    # Delete token and username/default folder from rh config file
    configs.delete(key="token")
    configs.delete(key="username")
    configs.delete(key="default_folder")

    rh_config_path = configs.CONFIG_PATH
    if not delete_rh_config_file and interactive_session:
        delete_rh_config_file = typer.confirm("Delete your local Runhouse config file?")

    if delete_rh_config_file:
        # Delete the credentials file on the file system
        configs.delete_defaults(rh_config_path)
        logger.info(f"Deleted Runhouse config file from path: {rh_config_path}")

    logger.info("Successfully logged out of Runhouse.")


def _login_download_secrets(force: bool = False):
    secrets = _get_vault_secrets()
    for name, config in secrets.items():
        secret = Secret.from_config(config)
        if secret.path:
            logger.info(f"Loading down secrets for {name} into {secret.path}")
            secret.write()


def _login_upload_secrets(force: bool = False):
    # upload locally saved secrets
    local_secrets = _get_local_secrets_configs()
    for name, config in local_secrets:
        upload_secret = force or typer.confirm(f"Upload credentials values for {name}?")
        if upload_secret:
            if config["name"].startswith("~") or config["name"].startswith("^"):
                config["name"] = config["name"][2:]
            secret = Secret.from_config(config)
            secret.save()

    from runhouse.resources.secrets.provider_secrets.providers import (
        _str_to_provider_class,
    )
    from runhouse.resources.secrets.secret_factory import provider_secret

    # upload locally configured provider secrets
    for provider in _str_to_provider_class.keys():
        if provider == "ssh" or provider in local_secrets.keys():
            continue
        secret = provider_secret(provider=provider)
        if secret.values:
            upload_secret = force or typer.confirm(
                f"Upload credentials values for {provider}?"
            )
            if upload_secret:
                secret.save()

    # upload local ssh keys
    default_ssh_folder = os.path.expanduser("~/.ssh")
    ssh_files = os.listdir(default_ssh_folder)
    for file in ssh_files:
        if file != "sky-key" and f"{file}.pub" in ssh_files:
            upload_secret = force or typer.confirm(
                f"Upload credentials values for ssh key {file}?"
            )
            if upload_secret:
                secret = provider_secret(
                    provider="ssh",
                    name=file,
                    path=os.path.join(default_ssh_folder, file),
                )
                secret.save()


def _logout_secrets(file: bool = True, force: bool = False):
    rh_config_secrets = configs.get("secrets", {})
    if file:
        for name, path in rh_config_secrets.items():
            delete_secret = force or typer.confirm(
                f"Delete credentials file for {name}?"
            )
            if delete_secret:
                secret = Secret.from_name(name)
                if isinstance(secret, SSHSecret):
                    logger.info(
                        "Automatic deletion for local SSH credentials file is not supported. "
                        "Please manually delete it if you would like to remove it"
                    )
                else:
                    path = os.path.expanduser(path)
                    if os.path.exists(path):
                        logger.info(
                            f"Removing file {path} associated with Secret {name}"
                        )
                        os.remove(path)
    configs.delete("secrets")
