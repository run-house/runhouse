import json
import os
from pathlib import Path
from typing import Dict, Optional, Union

from runhouse.globals import rns_client
from runhouse.logger import get_logger
from runhouse.rns.utils.api import load_resp_content, read_resp_data


USER_ENDPOINT = "user/secret"


logger = get_logger(__name__)


def load_config(name: str, endpoint: str = USER_ENDPOINT):
    if "/" not in name:
        name = f"{rns_client.current_folder}/{name}"
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
    headers = headers or rns_client.request_headers()
    uri = f"{rns_client.api_server_url}/{endpoint}/{resource_uri}?shared=true"
    resp = rns_client.session.get(
        uri,
        headers=headers,
    )
    if resp.status_code != 200:
        raise Exception(
            f"Received [{resp.status_code}] from Den GET '{uri}': Failed to load secret from Vault: {load_resp_content(resp)}"
        )

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
    headers = headers or rns_client.request_headers()
    uri = f"{rns_client.api_server_url}/{endpoint}/{resource_uri}"
    resp = rns_client.session.delete(
        uri,
        headers=headers,
    )
    if resp.status_code != 200:
        logger.error(
            f"Received [{resp.status_code}] from Den DELETE '{uri}': Failed to delete secrets from Vault: {load_resp_content(resp)}"
        )


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


def setup_cluster_creds(ssh_creds: Union[Dict, str], resource_name: str):
    """
    This method creates an Secret corresponding to a cluster's SSH creds. If the passed values are paths to private
    and/or public keys, this method extracts the content of the files saved in those files, in order for them to
    be saved in Vault. (Currently if we just pass a path/to/ssh/key to the Secret constructor, the content of the file
    will not be saved to Vault. We need to pass the content itself.)

    Args:
       ssh_creds (Dict or str): the ssh credentials of the cluster, passed by the user, dict.
       resource_name (str): the name of the resource that the ssh secret is associated with.

    Returns:
       SSHSecret
    """
    import runhouse as rh
    from runhouse.resources.secrets import Secret
    from runhouse.resources.secrets.provider_secrets import ProviderSecret
    from runhouse.resources.secrets.provider_secrets.ssh_secret import SSHSecret

    if isinstance(ssh_creds, str):
        return Secret.from_name(name=ssh_creds)

    creds_keys = list(ssh_creds.keys())

    if len(creds_keys) == 1 and "ssh_private_key" in creds_keys:
        if Path(ssh_creds["ssh_private_key"]).expanduser().exists():
            values = SSHSecret.extract_secrets_from_path(path=ssh_creds["private_key"])
            values["ssh_private_key"] = ssh_creds["private_key"]
        else:
            # case where the user decides to pass the private key as text and not as path.
            raise ValueError(
                "SSH creds require both private and public key, but only private key was provided"
            )

    elif "ssh_private_key" in creds_keys and "ssh_public key" in creds_keys:
        private_key, public_key = (
            ssh_creds["ssh_private_key"],
            ssh_creds["ssh_public_key"],
        )
        private_key_path, public_key_path = (
            Path(private_key).expanduser(),
            Path(public_key).expanduser(),
        )
        if private_key_path.exists() and public_key_path.exists():
            values = SSHSecret.extract_secrets_from_path(path=private_key)
            values["ssh_private_key"], values["ssh_public_key"] = (
                private_key_path,
                public_key_path,
            )
        else:
            values = {"private_key": private_key, "public_key": public_key}

    elif "ssh_private_key" in creds_keys and "ssh_user" in creds_keys:
        private_key, username = ssh_creds["ssh_private_key"], ssh_creds["ssh_user"]
        if Path(private_key).expanduser().exists():
            private_key = SSHSecret.extract_secrets_from_path(path=private_key).get(
                "private_key"
            )
        if Path(username).expanduser().exists():
            username = ProviderSecret.extract_secrets_from_path(username)
        values = {
            "private_key": private_key,
            "ssh_user": username,
            "ssh_private_key": ssh_creds["ssh_private_key"],
        }

    elif "ssh_user" in creds_keys and "password" in creds_keys:
        password, username = ssh_creds["password"], ssh_creds["ssh_user"]
        if Path(password).expanduser().exists():
            password = ProviderSecret.extract_secrets_from_path(password)
        if Path(username).expanduser().exists():
            username = ProviderSecret.extract_secrets_from_path(username)
        values = {"password": password, "ssh_user": username}

    else:
        values = {}
        for k in creds_keys:
            v = ssh_creds[k]
            if Path(v).exists():
                v = ProviderSecret.extract_secrets_from_path(v)
            values.update({k: v})

    values_to_add = {k: ssh_creds[k] for k in ssh_creds if k not in values.keys()}
    values.update(values_to_add)

    new_secret = rh.secret(name=f"{resource_name}-ssh-secret", values=values)
    return new_secret
