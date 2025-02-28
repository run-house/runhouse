import json
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
        download_secrets (bool): Whether to download secrets from your Runhouse account to local environment.
        upload_secrets (bool): Whether to upload local secrets to your Runhouse account.
        ret_token (bool): Whether to return your Runhouse token. (Default: False)
        interactive (bool): Whether to interactively go through the login flow. ``token`` must be provided to
            set this to False.

    Returns:
        Token if ``ret_token`` is set to True, otherwise nothing.
    """
    all_options_set = token and not any(
        arg is None for arg in (download_secrets, upload_secrets)
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
                f"Retrieve your token :key: to use :person_running: :house: Runhouse for "
                f"secrets and artifact management: {link}",
                style="bold yellow",
            )
            token = getpass("Token: ")

        rh_config_exists = configs.CONFIG_PATH.exists()
        if not rh_config_exists:
            download_secrets = False

        # Default ssh secret
        if not rns_client.default_ssh_key:
            default_ssh_path = typer.prompt(
                "Input the private key path for your default SSH key to use for launching clusters (e.g. ~/.ssh/id_rsa), or press `Enter` to skip",
                default="",
            )

            if default_ssh_path:
                from runhouse import provider_secret

                secret = provider_secret(provider="ssh", path=default_ssh_path)
                if secret.values:
                    secret.save()
                    configs.set("default_ssh_key", secret.name)
                else:
                    console.print(
                        f"Could not detect SSH key at {default_ssh_path}. Skipping"
                    )

        if sync_secrets:
            from runhouse import Secret

            if Secret.vault_secrets(rns_client.request_headers()):
                download_secrets = (
                    download_secrets
                    if download_secrets is not None
                    else typer.confirm(
                        "Download secrets from Vault to your local Runhouse environment?",
                        default=True,
                    )
                )
            upload_secrets = (
                upload_secrets
                if upload_secrets is not None
                else typer.confirm(
                    "Upload your local enabled provider secrets to Vault?",
                    default=True,
                )
            )

    if token:
        resp = requests.get(
            f"{rns_client.api_server_url}/user",
            headers={"Authorization": f"Bearer {token}"},
        )
        if resp.status_code != 200:
            error_msg = "Invalid token provided"
            if interactive:
                console.print(f"[red]{error_msg}[/red]")
                raise typer.Exit(1)
            else:
                raise ValueError(error_msg)

        # Note, this is to explicitly add it to the config file, as opposed to setting in python
        # via configs.token = token
        configs.set("token", token)

    autostop_mins = configs.defaults_cache.get("default_autostop")
    if autostop_mins is None:
        new_autostop_mins = typer.prompt(
            "Set the default number of minutes of inactivity after which to auto-terminate on-demand clusters. "
            "Press `Enter` to set to 60 minutes, or `-1` to disable autostop entirely.",
            default=60,
        )
        configs.set("default_autostop", int(new_autostop_mins))

    local_config: dict = configs.load_defaults_from_file()
    try:
        den_config: dict = configs.load_defaults_from_den()
        if not local_config and den_config:
            # if logging in to a new env, use the den config if we have it
            local_config = den_config
    except:
        # If we can't download the defaults, we'll just use the defaults
        den_config = {}

    if local_config and den_config:
        if json.dumps(local_config, sort_keys=True) != json.dumps(
            den_config, sort_keys=True
        ):
            message = (
                "The local Runhouse config differs from the config saved in Den.\n"
                "To replace your local version with the Den version, run: `runhouse config download`\n"
                "To replace the Den version with your local version, run: `runhouse config upload`"
            )
            if interactive:
                console.print(f"[yellow]{message}[/yellow]")
            else:
                logger.warning(message)

    _set_local_config_defaults(local_config, den_config)

    if not den_config:
        # save latest local copy in Den if none exists yet (ex: initial login)
        configs.upload_defaults(local_config)

    if download_secrets:
        _login_download_secrets(from_cli=from_cli)
    if upload_secrets:
        _login_upload_secrets(interactive=interactive)

    success_msg = "Successfully logged into Runhouse."
    if interactive:
        console.print(f"[green]{success_msg}[/green]")
    else:
        logger.info(success_msg)

    if ret_token:
        return token


def _set_local_config_defaults(local_config: dict, den_config: dict):
    """Set base default values in the local config to match those that will be saved in Den if not explicitly set by
    the user."""
    if not local_config.get("username") and den_config:
        configs.set("username", den_config["username"])

    for property in [
        "default_folder",
        "default_provider",
        "default_pool",
        "launcher",
        "use_spot",
        "autosave",
        "disable_observability",
        "api_server_url",
        "dashboard_url",
    ]:
        if not local_config.get(property) and den_config:
            value = den_config.get(property, configs.BASE_DEFAULTS[property])
            configs.set(property, value)


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
        rns_address = name if "/" in name else f"{rns_client.current_folder}/{name}"
        resource_uri = rns_client.resource_uri(rns_address)
        resp = requests.get(
            f"{rns_client.api_server_url}/resource/{resource_uri}",
            headers=headers or rns_client.request_headers(),
        )
        if resp.status_code == 200:
            local_secrets.pop(name, None)
            continue
        if interactive is not False:
            upload_secrets = typer.confirm(
                f"Upload secrets for {name}?",
                default=True,
            )
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
        delete_loaded_secrets (bool, optional): If ``True``, deletes the provider credentials file. (Default: ``None``)
        delete_rh_config_file (bool, optional): If ``True``, deletes the rh config file. (Default: ``None``)
        interactive (bool, optional): If ``True``, runs the logout process in interactive mode. (Default: ``None``)

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
                f"Delete credentials in {value} for {name}?",
                default=True,
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
        delete_rh_config_file = typer.confirm(
            "Delete your local Runhouse config file?",
            default=True,
        )

    if delete_rh_config_file:
        # Delete the credentials file on the file system
        configs.delete_defaults(rh_config_path)
        logger.info(f"Deleted Runhouse config file from path: {rh_config_path}")

    logger.info("Successfully logged out of Runhouse.")
