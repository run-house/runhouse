import unittest

import runhouse as rh


def test_get_all_secrets():
    vault_secrets = rh.Secrets.download_into_env(save_locally=False)
    providers = rh.Secrets.enabled_providers(as_str=True)
    assert set(list(vault_secrets)).issubset(
        providers
    ), "Secrets saved in Vault which are not enabled locally!"


def test_upload_user_provider_secrets():
    provider = "aws"
    rh.Secrets.put(provider=provider)

    # Retrieve the secret from Vault
    provider_secrets = rh.Secrets.get(provider=provider)
    assert provider_secrets


def test_upload_custom_provider_secrets():
    provider = "snowflake"
    rh.Secrets.put(provider=provider, secret={"secret_key": "abcdefg"})

    # Retrieve the secret from Vault
    provider_secrets = rh.Secrets.get(provider=provider)
    assert provider_secrets


def test_upload_all_provider_secrets():
    rh.Secrets.extract_and_upload()
    # Download back from Vault
    secrets = rh.Secrets.download_into_env(save_locally=False)
    assert secrets


def test_delete_provider_secrets():
    provider = "huggingface"
    rh.Secrets.put(provider=provider, secret={"secret_key": "abcdefg"})

    rh.Secrets.delete_from_vault(providers=[provider])
    secrets = rh.Secrets.get(provider=provider)

    assert not secrets


def test_sending_secrets_to_cluster():
    cluster = rh.cluster(name="^rh-cpu").up_if_not()
    enabled_providers = rh.Secrets.enabled_providers()

    cluster.send_secrets(providers=enabled_providers)

    # Confirm the secrets now exist on the cluster
    for provider_cls in enabled_providers:
        provider_name = provider_cls.PROVIDER_NAME
        command = [
            f"from runhouse.rns.secrets.{provider_name}_secrets import {str(provider_cls)}",
            f"print({str(provider_cls)}.has_secrets_file())",
        ]
        status_codes: list = cluster.run_python(command)
        if "False" in status_codes[0][1]:
            assert False, f"No credentials file found on cluster for {provider_name}"

    assert True


@unittest.skip("This test overrides local rh token if done incorrectly")
# Running this unit test will override local rh config token to "..."
# Was meant to be run by manually inputting token, but need a better way to test
def test_login():
    # TODO [DG] create a mock account and test this properly in CI
    token = "..."

    rh.login(
        token=token,
        download_config=False,
        download_secrets=False,
        upload_secrets=False,
        upload_config=False,
    )

    rh.login(
        token=token,
        download_config=False,
        download_secrets=False,
        upload_secrets=True,
        upload_config=True,
    )

    rh.login(
        token=token,
        download_config=True,
        download_secrets=True,
        upload_secrets=False,
        upload_config=False,
    )

    rh.login(
        token=token,
        download_config=True,
        download_secrets=True,
        upload_secrets=True,
        upload_config=True,
    )

    assert rh.rns_client.default_folder == "/..."


def test_logout():
    enabled_providers = rh.Secrets.enabled_providers(as_str=True)
    current_config: dict = rh.configs.load_defaults_from_file()

    rh.logout(delete_loaded_secrets=True, delete_rh_config_file=True)

    for provider_name in enabled_providers:
        p = rh.Secrets.builtin_provider_class_from_name(provider_name)
        assert not p.has_secrets_file()
        assert not rh.configs.get(provider_name)

    assert not rh.configs.load_defaults_from_file()

    # Add back what we deleted as part of the logout
    rh.configs.save_defaults(defaults=current_config)
    rh.Secrets.download_into_env(providers=enabled_providers, save_locally=True)


if __name__ == "__main__":
    unittest.main()
