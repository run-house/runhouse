import logging

from .. import rh_config
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
        import typer
        from rich.console import Console
        console = Console()
        console.print(f'Retrieve your token :key: here to use :person_running: :house: Runhouse for '
                      f'secrets and artifact management: '
                      f'[link={rh_config.configs.get("api_server_url")}/dashboard/?option=token]'
                      f'https://api.run.house[/link]',
                      style='bold yellow')
        token = typer.prompt("Token", type=str)

    request_headers = {"Authorization": f"Bearer {token}"}

    if download_config:
        rh_config.configs.download_and_save_defaults(headers=request_headers,
                                                     merge_with_existing=True,
                                                     merge_with_base_defaults=True,
                                                     upload_merged=upload_config)
        # We need to fresh the RNSClient to use the newly loaded configs
        rh_config.rns_client.refresh_defaults()
    elif upload_config:
        rh_config.configs.upload_defaults(headers=request_headers)
    else:
        # If we are not downloading or uploading config, we still want to make sure the token is valid
        try:
            rh_config.configs.download_defaults(headers=request_headers)
        except:
            logger.error('Failed to validate token')
            return None

    if download_secrets:
        Secrets.download_into_env(headers=request_headers)

    if upload_secrets:
        Secrets.extract_and_upload(headers=request_headers)

    logger.info('Successfully logged into Runhouse')
    if ret_token:
        return token
