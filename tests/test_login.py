import os
import unittest

import pytest

import runhouse as rh
import sky

from runhouse.resources.secrets.utils import _get_vault_secrets


def add_secrets_to_vault(headers):
    """Add some test secrets to Vault"""
    # Add real credentials for AWS and SKY to test sky status
    aws_values = {
        "access_key": os.getenv("TEST_AWS_ACCESS_KEY"),
        "secret_key": os.getenv("TEST_AWS_SECRET_KEY"),
    }
    sky_values = {
        "ssh_private_key": os.getenv("TEST_SKY_PRIVATE_KEY"),
        "ssh_public_key": os.getenv("TEST_SKY_PUBLIC_KEY"),
    }

    rh.provider_secret("aws", values=aws_values).save(headers=headers)
    rh.provider_secret("sky", values=sky_values).save(headers=headers)
    rh.provider_secret("snowflake", values={"token": "ABCD1234"}).save(headers=headers)
    rh.provider_secret("ssh", values={"key-one": "12345", "key-one.pub": "ABCDE"}).save(
        headers=headers
    )


@pytest.mark.logintest
def test_login_flow_in_new_env():
    token = os.getenv("TEST_TOKEN")
    headers = {"Authorization": f"Bearer {token}"}

    add_secrets_to_vault(headers)

    secrets_in_vault = _get_vault_secrets(headers=headers)
    assert secrets_in_vault, "No secrets found in Vault"

    providers_in_vault = list(secrets_in_vault)
    assert set(providers_in_vault) == set(["aws", "snowflake", "ssh", "sky", "github"])

    # Login and download secrets stored in Vault into the new env
    rh.login(token=token, download_secrets=True, interactive=False)

    # Once secrets are saved down to their local config, confirm we have sky enabled
    sky.check.check(quiet=True)
    clouds = sky.global_user_state.get_enabled_clouds()
    cloud_names = [str(c).lower() for c in clouds]
    assert "aws" in cloud_names

    rh.Secret.delete_from_vault(secrets=providers_in_vault, headers=headers)
    secrets_in_vault = _get_vault_secrets(headers=headers)
    assert not secrets_in_vault


if __name__ == "__main__":
    unittest.main()
