import json
import logging
import os

from typing import Dict, List

import requests

from runhouse.globals import rns_client
from runhouse.resources.secrets.cluster_secrets.cluster_secret import ClusterSecret
from runhouse.resources.secrets.secret import Secret
from runhouse.rns.utils.api import read_resp_data


logger = logging.getLogger(__name__)


def _get_vault_secrets(names: List[str]) -> List[Secret]:
    resp = requests.get(
        f"{rns_client.api_server_url}/user/secret",
        headers=rns_client.request_headers,
    )

    if resp.status_code != 200:
        raise Exception("Failed to download secrets from Vault")

    secrets = read_resp_data(resp)
    if names is not None:
        secrets = {name: secrets[name] for name in names if name in secrets}

    return secrets


def _read_local_config(name: str) -> Dict:
    path = os.path.expanduser(f"~/.rh/secrets/{name}.json")
    with open(path, "r") as f:
        config = json.load(f)
    return config


def _get_local_secrets(names: List[str]) -> List[Secret]:
    all_names = [
        file.strip(".json")
        for file in os.listdir(os.path.expanduser("~/.rh/secrets"))
        if file.endswith("json")
    ]
    names = [name for name in names if name in all_names] if names else all_names
    secrets = {name: _read_local_config(name) for name in names}
    return secrets


def write_secrets(
    names: List[str] = None,
):
    secrets = _get_vault_secrets(names)

    for name in secrets.keys():
        secret = Secret.from_config(secrets[name])
        secret.write()


def upload_local_secrets(
    names: List[str] = None,
    extract_secrets: bool = None,
):
    """Upload locally configured secrets into Vault."""
    local_secrets = _get_local_secrets(names)

    for _, config in local_secrets.items():
        if config["name"].startswith("~") or config["name"].startswith("^"):
            config["name"] = config["name"][2:]
        secret = Secret.from_config(config)
        secret.save(secrets=extract_secrets)


def delete_secrets(
    names: List[str] = None,
    file: bool = True,
):
    local_secrets = _get_local_secrets(names)
    for _, config in local_secrets.items():
        secret = Secret.from_config(config)
        if not isinstance(secret, ClusterSecret):
            secret.delete(file=file)
        else:
            logger.info(
                "Automatic deletion for local SSH credentials file is not supported. "
                "Please manually delete it if you would like to remove it"
            )
