import json
import logging
import os
from pathlib import Path

from typing import Dict, List

import requests

from runhouse.globals import rns_client
from runhouse.rns.utils.api import read_resp_data

logger = logging.getLogger(__name__)

USER_ENDPOINT = "user/secret"


def _get_vault_secrets(names: List[str] = None) -> List["Secret"]:
    resp = requests.get(
        f"{rns_client.api_server_url}/user/secret",
        headers=rns_client.request_headers,
    )

    if resp.status_code != 200:
        raise Exception("Failed to download secrets from Vault")

    secrets = read_resp_data(resp)
    if names is not None:
        secrets = {name: secrets[name] for name in names if name in secrets}
    for name, config in secrets.items():
        if config.get("data", None):
            config.update(config["data"])
            del config["data"]
            secrets[name] = config

    return secrets


def _read_local_config(name: str) -> Dict:
    path = os.path.expanduser(f"~/.rh/secrets/{name}.json")
    with open(path, "r") as f:
        config = json.load(f)
    return config


def _get_local_secrets_configs(names: List[str] = None) -> List["Secret"]:
    if not os.path.exists(os.path.expanduser("~/.rh/secrets")):
        return {}
    all_names = [
        Path(file).stem
        for file in os.listdir(os.path.expanduser("~/.rh/secrets"))
        if file.endswith("json")
    ]
    names = [name for name in names if name in all_names] if names else all_names
    secrets = {name: _read_local_config(name) for name in names}
    return secrets


def _upload_local_secrets(
    names: List[str] = None,
    extract_values: bool = None,
):
    """Upload locally configured secrets into Vault."""
    local_secrets = _get_local_secrets_configs(names)
    for _, config in local_secrets.items():
        if config["name"].startswith("~") or config["name"].startswith("^"):
            config["name"] = config["name"][2:]
        secret = Secret.from_config(config)
        secret.save(values=extract_values)

    from runhouse.resources.secrets.provider_secrets.providers import (
        _str_to_provider_class,
    )
    from runhouse.resources.secrets.secret_factory import provider_secret

    for provider in _str_to_provider_class.keys():
        if provider in local_secrets.keys():
            continue
        secret = provider_secret(provider=provider)
        if secret.values:
            secret.save(values=extract_values)


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


def _load_from_vault(name, endpoint):
    # resource_uri = rns_client.resource_uri(name)
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


def _load_env_vars_from_path(path):
    from dotenv import dotenv_values

    path = os.path.expanduser(path)
    if Path(path).exists():
        return dotenv_values(path)
    return {}


def load_config(name, endpoint: str = USER_ENDPOINT):
    rns_address = rns_client.resolve_rns_path(name)

    if rns_address.startswith("/"):
        return _load_from_vault(name, endpoint)

    return _load_from_local(name)
