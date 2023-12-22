import json
import logging
import os
from pathlib import Path
from typing import Dict, Optional

import requests

from runhouse.globals import rns_client
from runhouse.rns.utils.api import load_resp_content, read_resp_data

logger = logging.getLogger(__name__)

USER_ENDPOINT = "user/secret"


def load_config(name: str, endpoint: str = USER_ENDPOINT):
    rns_address = rns_client.resolve_rns_path(name)
    if rns_address.startswith("/"):
        # Load via Resource API
        rns_config = rns_client.load_config(name=name)
        if not rns_config:
            raise ValueError(f"Secret {name} not found in Den.")

        # Load via Secrets API
        resource_uri = rns_client.resource_uri(name)
        secret_values = _load_vault_secret(resource_uri, endpoint)
        return {**rns_config, **{"values": secret_values}}

    # Load from local config
    return _load_local_config(name)


def _load_vault_secret(
    resource_uri: str,
    endpoint: str = USER_ENDPOINT,
    headers: Optional[Dict] = None,
):
    """Load secrets data from Vault for a particular resource URI. By default we allow for reloading shared secrets."""
    headers = headers or rns_client.request_headers
    resp = requests.get(
        f"{rns_client.api_server_url}/{endpoint}/{resource_uri}?shared=true",
        headers=headers,
    )
    if resp.status_code != 200:
        raise Exception(f"Failed to load secret from Vault: {load_resp_content(resp)}")

    config = read_resp_data(resp)

    if len(config.keys()) == 1:
        vault_key = list(config.keys())[0]
        config = config[vault_key]
    if config.get("data", None):
        return config["data"]["values"]
    return config


def _delete_vault_secrets(
    resource_uri: str,
    endpoint: str = USER_ENDPOINT,
    headers: Optional[Dict] = None,
):
    headers = headers or rns_client.request_headers
    resp = requests.delete(
        f"{rns_client.api_server_url}/{endpoint}/{resource_uri}",
        headers=headers,
    )
    if resp.status_code != 200:
        logger.error(f"Failed to delete secrets from Vault: {load_resp_content(resp)}")


def _load_local_config(name):
    if name.startswith("~") or name.startswith("^"):
        name = name[2:]
    config_path = os.path.expanduser(f"~/.rh/secrets/{name}.json")
    if not Path(config_path).exists():
        return None

    logger.info(f"Loading config from local file {config_path}")
    with open(config_path, "r") as f:
        try:
            config = json.load(f)
        except json.decoder.JSONDecodeError as e:
            logger.error(f"Error loading config from {config_path}: {e}")
            return None
    return config


def _is_matching_subset(existing_vals, new_vals):
    for key in new_vals:
        if key in existing_vals and existing_vals[key] != new_vals[key]:
            return False
    return True


def _check_file_for_mismatches(path, existing_vals, new_vals, overwrite):
    # Returns True if performs the check satisfactorily and not overriding the file.
    # Returns False if no existing vals exist or overwriting the file.
    if not existing_vals:
        return False
    if _is_matching_subset(existing_vals, new_vals):
        logger.info(f"Secrets already exist in {path}.")
        return True
    elif not overwrite:
        logger.warning(
            f"Path {path} already exists with a different set of values, "
            "and overwrite is set to False, so leaving the file as is. Please set overwrite "
            "to True or manually overwrite the secret file if you intend to do so."
        )
        return True
    return False
