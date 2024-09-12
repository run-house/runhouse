import os
from typing import Dict, Optional

import requests
import typer

from runhouse.globals import configs, rns_client
from runhouse.logger import get_logger

logger = get_logger(__name__)


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
    interactive: bool = None,
    from_cli: bool = False,
    sync_secrets: bool = False,
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

        rh_config_exists = configs.CONFIG_PATH.exists()
        if not rh_config_exists:
            upload_config = False
            download_secrets = False

        # download the config automatically if no config.yaml exists
        download_config = (
            download_config
            if download_config is not None or not rh_config_exists
            else typer.confirm(
                "Download your Runhouse config to your local .rh folder?",
                default=True,
            )
        )
        upload_config = (
            upload_config
            if upload_config is not None
            else typer.confirm("Upload your local .rh config to Runhouse?")
        )

        if sync_secrets:
            from runhouse import Secret

            if Secret.vault_secrets(rns_client.request_headers()):
                download_secrets = (
                    download_secrets
                    if download_secrets is not None
                    else typer.confirm(
                        "Download secrets from Vault to your local Runhouse environment?"
                    )
                )
            upload_secrets = (
                upload_secrets
                if upload_secrets is not None
                else typer.confirm(
                    "Upload your local enabled provider secrets to Vault?"
                )
            )

    if token:
        # Note, this is to explicitly add it to the config file, as opposed to setting in python
        # via configs.token = token
        configs.set("token", token)

    if download_config:
        configs.download_and_save_defaults()
    if upload_config:
        configs.load_defaults_from_file()
        configs.upload_defaults(defaults=configs.defaults_cache)

    if not (download_config or upload_config):
        # If we are not downloading or uploading config, we still want to make sure the token is valid
        # and also download the username and default folder
        try:
            defaults = configs.download_defaults()
        except:
            logger.error("Failed to validate token")
            return None
        configs.set("username", defaults["username"])
        configs.set("default_folder", defaults["default_folder"])

    if download_secrets:
        _login_download_secrets(from_cli=from_cli)
    if upload_secrets:
        _login_upload_secrets(interactive=interactive)

    logger.info("Successfully logged into Runhouse.")
    if ret_token:
        return token


def _login_download_secrets(headers: Optional[str] = None, from_cli=False):
    from runhouse import Secret

    secrets = Secret.vault_secrets(headers=headers or rns_client.request_headers())
    env_secrets = {}
    for name in secrets:
        try:
            secret = Secret.from_name(name)
            if not (hasattr(secret, "path") or hasattr(secret, "env_vars")):
                continue

            download_path = (
                secret.path
                if secret.path
                else f"{secret._DEFAULT_CREDENTIALS_PATH}/{name}"
                if hasattr(secret, "provider") and secret.provider == "ssh"
                else secret._DEFAULT_CREDENTIALS_PATH
            )

            if download_path and not secret.env_vars:
                logger.info(f"Loading down secrets for {name} into {download_path}")
                secret.write(path=download_path)
            else:
                env_vars = secret.env_vars or secret._DEFAULT_ENV_VARS
                logger.info(
                    f"Writing down env secrets for {name} into {env_vars.values()}"
                )
                if not from_cli:
                    secret.write(env=True)
                else:
                    for key, val in secret.values.items():
                        if key in env_vars:
                            env_secrets.update({env_vars[key]: val})

        except ValueError as e:
            logger.warning(
                f"Encountered {e}. Was not able to load down secrets for {name}."
            )

    if from_cli and env_secrets:
        folder = os.path.expanduser("~/.rh/secrets")
        os.makedirs(folder, exist_ok=True)
        with open(f"{folder}/login.env", "w") as f:
            for key, val in env_secrets.items():
                f.write(f"{key}={val}\n")
        logger.info(
            "Env var secrets written down into ~/.rh/secrets/login.env. "
            "Please run `source ~/.rh/secrets/login.env` to set the environment variables."
        )


def _login_upload_secrets(interactive: bool, headers: Optional[Dict] = None):
    from runhouse import Secret

    local_secrets = Secret.local_secrets()
    provider_secrets = Secret.extract_provider_secrets()
    local_secrets.update(provider_secrets)
    names = list(local_secrets.keys())

    for name in names:
        resource_uri = rns_client.resource_uri(name)
        resp = requests.get(
            f"{rns_client.api_server_url}/resource/{resource_uri}",
            headers=headers or rns_client.request_headers(),
        )
        if resp.status_code == 200:
            local_secrets.pop(name, None)
            continue
        if interactive is not False:
            upload_secrets = typer.confirm(f"Upload secrets for {name}?")
            if not upload_secrets:
                local_secrets.pop(name, None)

    if local_secrets:
        logger.info(f"Uploading secrets for {list(local_secrets)} to Vault.")
        for _, secret in local_secrets.items():
            secret.save(save_values=True)


def logout(
    delete_loaded_secrets: bool = None,
    delete_rh_config_file: bool = None,
    interactive: bool = None,
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
    from runhouse.resources.secrets import Secret
    from runhouse.resources.secrets.provider_secrets.ssh_secret import SSHSecret

    interactive_session: bool = (
        interactive if interactive is not None else is_interactive()
    )

    config_secrets = list(configs.get("secrets", {}).items())
    for (name, value) in config_secrets:
        try:
            secret = Secret.from_name(name)

            if isinstance(secret, SSHSecret):
                logger.info(
                    "Automatic deletion for local SSH credentials file is not supported. "
                    "Please manually delete it if you would like to remove it"
                )
                configs.delete_provider(name)
                continue
        except ValueError:
            pass

        if interactive_session:
            delete_loaded_secrets = typer.confirm(
                f"Delete credentials in {value} for {name}?"
            )

        if delete_loaded_secrets:
            if isinstance(value, str):
                path = os.path.expanduser(value)
                if os.path.exists(path):
                    os.remove(path)
            else:  # list of env variables set
                for key in value:
                    del os.environ[key]

        configs.delete_provider(name)

    local_secrets = Secret.local_secrets()
    for _, secret in local_secrets.items():
        secret.delete()

    # Delete token and username/default folder from rh config file
    configs.delete(key="token")
    configs.delete(key="username")
    configs.delete(key="default_folder")

    # Delete values in configs object
    configs.token = None
    configs.username = None
    configs.default_folder = None

    # Wipe env vars
    os.environ.pop("RH_TOKEN", None)
    os.environ.pop("RH_USERNAME", None)

    rh_config_path = configs.CONFIG_PATH
    if not delete_rh_config_file and interactive_session:
        delete_rh_config_file = typer.confirm("Delete your local Runhouse config file?")

    if delete_rh_config_file:
        # Delete the credentials file on the file system
        configs.delete_defaults(rh_config_path)
        logger.info(f"Deleted Runhouse config file from path: {rh_config_path}")

    logger.info("Successfully logged out of Runhouse.")
