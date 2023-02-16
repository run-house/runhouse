import unittest

import runhouse as rh


def test_get_all_secrets():
    secrets = rh.Secrets.download_into_env(save_locally=True)
    providers = rh.Secrets.builtin_providers(as_str=True)
    assert set(providers) == {"aws", "gcp", "sky", "hf"}
    assert secrets


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

    configured_providers = rh.Secrets.configured_providers()

    cluster.send_secrets(providers=configured_providers)
    # Confirm the secrets now exist on the cluster
    for provider_cls in configured_providers:
        provider_name = provider_cls.PROVIDER_NAME
        p_str = str(provider_cls())
        command = [
            f"from runhouse.rns.secrets import {p_str}",
            f"print({p_str}.has_secrets_file())",
        ]
        status_codes = cluster.run_python(command)
        if "False" in status_codes[0][1]:
            assert False, f"No credentials file found on cluster for {provider_name}"

    assert True


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
    from runhouse import configs, Secrets

    configured_providers = Secrets.configured_providers(as_str=True)
    current_config: dict = configs.load_defaults_from_file()

    rh.logout(delete_loaded_secrets=True, delete_rh_config_file=True)

    for provider_name in configured_providers:
        p = Secrets.builtin_provider_class_from_name(provider_name)
        assert not p.has_secrets_file()
        assert not configs.get(provider_name)

    assert not configs.load_defaults_from_file()

    # Add back what we deleted as part of the logout
    configs.save_defaults(defaults=current_config)
    Secrets.download_into_env(providers=configured_providers)


# TODO [JL] test custom secret file paths


if __name__ == "__main__":
    unittest.main()
