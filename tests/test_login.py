import os

import runhouse as rh
import sky
from runhouse.rns.login import _login_download_secrets


def add_secrets_to_vault(headers):
    """Add some test secrets to Vault"""
    # Add real credentials for AWS and SKY to test sky status
    rh.provider_secret(
        name="/aws",  # add backslash / to name to force it to be vault secret
        provider="aws",
        values={
            "access_key": os.getenv("TEST_AWS_ACCESS_KEY"),
            "secret_key": os.getenv("TEST_AWS_SECRET_KEY"),
        },
    ).save(headers=headers)

    rh.provider_secret(
        name="/sky",
        provider="sky",
        values={
            "private_key": os.getenv("TEST_SKY_PRIVATE_KEY"),
            "public_key": os.getenv("TEST_SKY_PUBLIC_KEY"),
        },
    ).save(headers=headers)

    rh.provider_secret(
        name="/snowflake",
        provider="snowflake",
        values={"token": "ABCD1234"},
    ).save(headers=headers)


def test_login_flow_in_new_env():
    token = os.getenv("KITCHEN_TESTER_TOKEN")
    headers = {"Authorization": f"Bearer {token}"}

    add_secrets_to_vault(headers)

    secrets_in_vault = rh.Secret.vault_secrets(headers=headers)
    assert secrets_in_vault, "No secrets found in Vault"

    # Run login download secrets stored in Vault into the new env
    _login_download_secrets(headers=headers)

    # Once secrets are saved down to their local config, confirm we have sky enabled
    sky.check.check(quiet=True)
    clouds = sky.global_user_state.get_enabled_clouds()
    cloud_names = [str(c).lower() for c in clouds]
    assert "aws" in cloud_names

    for secret in secrets_in_vault.values():
        secret.delete(headers=headers)

    secrets_in_vault = rh.Secret.vault_secrets(headers=headers)
    assert not secrets_in_vault
