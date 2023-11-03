import json
import logging
import os
from pathlib import Path

import requests

from runhouse.globals import rns_client
from runhouse.rns.utils.api import read_resp_data

logger = logging.getLogger(__name__)

USER_ENDPOINT = "user/secret"


def _load_from_vault(name, endpoint):
    resp = requests.get(
        f"{rns_client.api_server_url}/{endpoint}/{name}",
        headers=rns_client.request_headers,
    )
    if resp.status_code != 200:
        raise Exception(f"Secret {name} not found in Vault.")
    config = read_resp_data(resp)

    if len(config.keys()) == 1:
        vault_key = list(config.keys())[0]
        config = config[vault_key]
    if config.get("data", None):
        config.update(config["data"])
        del config["data"]
    return config


def _load_from_local(name):
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


def load_config(name, endpoint: str = USER_ENDPOINT):
    rns_address = rns_client.resolve_rns_path(name)

    if rns_address.startswith("/"):
        return _load_from_vault(name, endpoint)

    return _load_from_local(name)
