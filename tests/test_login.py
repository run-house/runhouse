import os

import runhouse as rh
import sky
from runhouse.rh_config import configs


def add_secrets_to_vault(headers):
    """Add some test secrets to Vault"""
    # Add real credentials for AWS and SKY to test sky status
    rh.Secrets.put(
        provider="aws",
        secret={
            "access_key": os.getenv("TEST_AWS_ACCESS_KEY"),
            "secret_key": os.getenv("TEST_AWS_SECRET_KEY"),
        },
        headers=headers,
    )
    rh.Secrets.put(
        provider="sky",
        secret={
            "ssh_private_key": os.getenv("TEST_SKY_PRIVATE_KEY"),
            "ssh_public_key": os.getenv("TEST_SKY_PUBLIC_KEY"),
        },
        headers=headers,
    )
    rh.Secrets.put(provider="snowflake", secret={"token": "ABCD1234"}, headers=headers)
    rh.Secrets.put(
        provider="ssh",
        secret={"key-one": "12345", "key-one.pub": "ABCDE"},
        headers=headers,
    )


def test_login_flow_in_new_env():
    token = os.getenv("TEST_TOKEN")
    headers = {"Authorization": f"Bearer {token}"}

    add_secrets_to_vault(headers)

    secrets_in_vault = rh.Secrets.download_into_env(
        headers=headers, save_locally=False, check_enabled=False
    )
    assert secrets_in_vault, "No secrets found in Vault"

    providers_in_vault = list(secrets_in_vault)

    # Login and download secrets stored in Vault into the new env
    rh.login(token=token, download_secrets=True)

    # Once secrets are saved down to their local config, confirm we have sky enabled
    sky.check.check(quiet=True)
    clouds = sky.global_user_state.get_enabled_clouds()
    cloud_names = [str(c).lower() for c in clouds]
    assert "aws" in cloud_names

    enabled_providers = rh.Secrets.enabled_providers(as_str=True)

    for provider in providers_in_vault:
        if provider in enabled_providers:
            assert configs.get("secrets", {}).get(
                provider
            ), f"No credentials path for {provider} stored in .rh config!"

    rh.Secrets.delete_from_vault(providers=["aws", "snowflake", "ssh", "sky"])
    secrets_in_vault = rh.Secrets.download_into_env(
        save_locally=False, check_enabled=False
    )
    assert not secrets_in_vault
