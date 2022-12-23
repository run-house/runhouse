import logging

from runhouse.rh_config import configs, rns_client
from .secrets import Secrets

logger = logging.getLogger(__name__)


def is_interactive():
    import __main__ as main
    return not hasattr(main, '__file__')


def login(token: str = None,
          download_config: bool = False,
          upload_config: bool = False,
          download_secrets: bool = False,
          upload_secrets: bool = False,
          ret_token: bool = False):
    """Login to Runhouse. Validates token provided, with options to upload or download stored secrets or config between
    local environment and Runhouse / Vault.
    """
    if not token and is_interactive():
        from getpass import getpass
        from rich.console import Console
        console = Console()
        console.print("""
            ____              __                             @ @ @     
           / __ \__  ______  / /_  ____  __  __________     []___      
          / /_/ / / / / __ \/ __ \/ __ \/ / / / ___/ _ \   /    /\____    @@
         / _, _/ /_/ / / / / / / / /_/ / /_/ (__  )  __/  /_/\_//____/\  @@@@
        /_/ |_|\__,_/_/ /_/_/ /_/\____/\__,_/____/\___/   | || |||__|||   ||
        """)
        console.print(f'Retrieve your token :key: here to use :person_running: :house: Runhouse for '
                      f'secrets and artifact management: '
                      f'[link={configs.get("api_server_url")}/dashboard/?option=token]'
                      f'https://api.run.house[/link]',
                      style='bold yellow')
        token = getpass("Token: ")

    if token:
        configs.set('token', token)

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
            logger.error('Failed to validate token')
            return None

    if download_secrets:
        Secrets.download_into_env()

    if upload_secrets:
        Secrets.extract_and_upload()

    logger.info('Successfully logged into Runhouse')
    if ret_token:
        return token
