import os
import unittest

import runhouse as rh
import sky


def add_secrets_to_vault(headers):
    """Add some test secrets to Vault"""
    # Add real credentials for AWS and SKY to test sky status
    rh.provider_secret(
        provider="aws",
        values={
            "access_key": os.getenv("TEST_AWS_ACCESS_KEY"),
            "secret_key": os.getenv("TEST_AWS_SECRET_KEY"),
        },
    ).save(headers=headers)

    rh.provider_secret(
        provider="sky",
        values={
            "ssh_private_key": os.getenv("TEST_SKY_PRIVATE_KEY"),
            "ssh_public_key": os.getenv("TEST_SKY_PUBLIC_KEY"),
        },
    ).save(headers=headers)

    rh.provider_secret(
        provider="snowflake",
        values={"token": "ABCD1234"},
    ).save(headers=headers)


def test_login_flow_in_new_env():
    token = os.getenv("TEST_TOKEN")
    headers = {"Authorization": f"Bearer {token}"}

    add_secrets_to_vault(headers)

    secrets_in_vault = rh.Secret.vault_secrets(headers=headers)
    assert secrets_in_vault, "No secrets found in Vault"

    providers_in_vault = list(secrets_in_vault)

    # Login and download secrets stored in Vault into the new env
    rh.login(token=token, download_secrets=True, interactive=False)

    # Once secrets are saved down to their local config, confirm we have sky enabled
    sky.check.check(quiet=True)
    clouds = sky.global_user_state.get_enabled_clouds()
    cloud_names = [str(c).lower() for c in clouds]
    assert "aws" in cloud_names

    for provider in providers_in_vault:
        rh.secret(provider).delete(headers=headers)

    secrets_in_vault = rh.Secret.vault_secrets()
    assert not secrets_in_vault


if __name__ == "__main__":
    unittest.main()
