import logging

import typer

from runhouse.globals import configs, rns_client

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
            else typer.confirm("Upload your local enabled provider secrets to Vault?")
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

    if download_secrets:
        _login_download_secrets()

    if upload_secrets:
        _login_upload_secrets(interactive=interactive)

    logger.info("Successfully logged into Runhouse.")
    if ret_token:
        return token


def _login_download_secrets(headers: str = rns_client.request_headers):
    from runhouse import Secret

    secrets = Secret.vault_secrets(headers=headers)
    for name, secret in secrets.items():
        try:
            download_path = secret.path or secret._DEFAULT_CREDENTIALS_PATH
            if download_path:
                logger.info(f"Loading down secrets for {name} into {download_path}")
                secret.write()
        except AttributeError:
            logger.warning(f"Was not able to load down secrets for {name}.")
            continue


def _login_upload_secrets(interactive: bool):
    from runhouse import Secret

    local_secrets = Secret.local_secrets()
    provider_secrets = Secret.extract_provider_secrets()
    local_secrets.update(provider_secrets)
    names = list(local_secrets.keys())

    vault_secrets = Secret.vault_secrets()

    for name in names:
        if name in vault_secrets:
            local_secrets.pop(name, None)
            continue
        if interactive:
            upload_secrets = typer.confirm(f"Upload secrets for {name}?")
            if not upload_secrets:
                local_secrets.pop(name, None)

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

    local_secret_names = list(configs.get("secrets", {}).keys())
    for name in local_secret_names:
        secret = Secret.from_name(name)

        if isinstance(secret, SSHSecret):
            logger.info(
                "Automatic deletion for local SSH credentials file is not supported. "
                "Please manually delete it if you would like to remove it"
            )
            continue

        if interactive_session:
            delete_loaded_secrets = typer.confirm(
                f"Delete credentials file for {name}?"
            )

        if delete_loaded_secrets:
            secret.delete(contents=True)
        else:
            secret.delete()
            configs.delete(name)

    local_secrets = Secret.local_secrets()
    for _, secret in local_secrets.items():
        secret.delete()

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
