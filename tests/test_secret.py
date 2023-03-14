import unittest

import runhouse as rh


def test_get_all_secrets_from_vault():
    vault_secrets = rh.Secrets.download_into_env(save_locally=False)
    providers = rh.Secrets.enabled_providers(as_str=True)
    assert set(list(vault_secrets)).issubset(
        providers
    ), "Secrets saved in Vault which are not enabled locally!"


def test_upload_custom_provider_to_vault():
    provider = "sample_provider"
    rh.Secrets.put(provider=provider, secret={"secret_key": "abcdefg"})

    # Retrieve the secret from Vault
    provider_secrets = rh.Secrets.get(provider=provider)
    assert provider_secrets

    rh.Secrets.delete_from_vault(providers=[provider])
    provider_secrets = rh.Secrets.get(provider=provider)
    assert not provider_secrets


def test_upload_aws_to_vault():
    provider = "aws"

    # NOTE: We don't need to provide any secrets here - they will be extracted from the local config file
    rh.Secrets.put(provider=provider)

    # Retrieve the secret from Vault
    provider_secrets = rh.Secrets.get(provider=provider)
    assert provider_secrets

    rh.Secrets.delete_from_vault(providers=[provider])
    provider_secrets = rh.Secrets.get(provider=provider)
    assert not provider_secrets


def test_add_custom_provider():
    import configparser
    import shutil
    from pathlib import Path

    provider = "new_provider"
    creds_dir = Path("~/.config/new_provider").expanduser()
    creds_dir.mkdir(parents=True, exist_ok=True)

    creds_file_path = str(creds_dir / "config")

    parser = configparser.ConfigParser()
    parser.add_section(provider)
    parser.set(section=provider, option="token", value="abcdefg")

    rh.Secrets.save_to_config_file(parser, creds_file_path)
    rh.configs.set_nested("secrets", {provider: creds_file_path})

    # Upload to vault
    rh.Secrets.put(provider, secret={"token": "abcdefg"})
    assert rh.Secrets.get(provider)

    rh.Secrets.delete_from_vault([provider])
    shutil.rmtree(str(creds_dir))
    rh.configs.delete(provider)

    assert not rh.Secrets.get(provider)


def test_upload_all_provider_secrets_to_vault():
    rh.Secrets.extract_and_upload()
    # Download back from Vault
    secrets = rh.Secrets.download_into_env(save_locally=False)
    assert secrets


def test_add_ssh_secrets():
    from runhouse.rns.secrets.ssh_secrets import SSHSecrets

    provider = "ssh"
    # Save to local .ssh directory
    sample_ssh_keys = {"key-one": "12345", "key-one.pub": "ABCDE"}
    SSHSecrets.save_secrets(secrets=sample_ssh_keys, overwrite=True)
    local_secrets: dict = rh.Secrets.load_provider_secrets(providers=[provider])
    assert local_secrets
    assert rh.configs.get("secrets", {}).get(provider)

    # Upload to Vault
    rh.Secrets.put(provider=provider, secret=sample_ssh_keys)
    vault_secrets = rh.Secrets.get(provider=provider)
    assert vault_secrets

    # Delete from Vault & local configs
    rh.configs.delete(provider)
    rh.Secrets.delete_from_vault([provider])
    for f_name, _ in sample_ssh_keys.items():
        ssh_key_path = f"{SSHSecrets.default_credentials_path()}/{f_name}"
        rh.Secrets.delete_secrets_file(file_path=ssh_key_path)

    assert not rh.Secrets.get(provider)
    assert not rh.configs.get("secrets", {}).get(provider)


def test_add_github_secrets():
    from runhouse.rns.secrets.github_secrets import GitHubSecrets

    provider = "github"

    vault_secrets = rh.Secrets.get(provider=provider)
    if not vault_secrets:
        # Create mock tokens and test uploading / downloading / deleting
        # Upload to Vault
        sample_gh_token = {"token": "12345"}
        rh.Secrets.put(provider=provider, secret=sample_gh_token)

        # save to local config file
        GitHubSecrets.save_secrets(secrets=sample_gh_token, overwrite=True)
        assert rh.configs.rh.configs.get("secrets", {}).get(provider)
        assert rh.Secrets.get(provider)

        # Delete from Vault & local config
        rh.Secrets.delete_from_vault(providers=[provider])
        rh.Secrets.delete_from_local_env(providers=[provider])

        assert not rh.Secrets.get(provider)
        assert not rh.configs.get("secrets", {}).get(provider)

    else:
        # Load from local config so as not to overwrite existing GitHub secrets
        local_secrets: dict = rh.Secrets.load_provider_secrets(providers=[provider])
        assert local_secrets
        assert rh.configs.get("secrets", {}).get(provider)


def test_sending_secrets_to_cluster():
    cluster = rh.cluster(name="^rh-cpu").up_if_not()
    enabled_providers: list = rh.Secrets.enabled_providers()

    cluster.send_secrets(providers=enabled_providers)

    # Confirm the secrets now exist on the cluster
    for provider_cls in enabled_providers:
        provider_name = provider_cls.PROVIDER_NAME
        commands = [
            f"from runhouse.rns.secrets.{provider_name}_secrets import {str(provider_cls)}",
            f"print({str(provider_cls)}.has_secrets_file())",
        ]
        status_codes: list = cluster.run_python(commands)
        if "False" in status_codes[0][1]:
            assert False, f"No credentials file found on cluster for {provider_name}"

    assert True


@unittest.skip("for manually debugging full login flow")
def test_login_manual():
    rh.login(
        token="...",
        download_config=False,
        download_secrets=True,
        upload_secrets=False,
        upload_config=False,
    )


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

    # Restore the config and secrets to its pre-logout state
    rh.configs.save_defaults(defaults=current_config)
    secrets = rh.Secrets.download_into_env(
        providers=enabled_providers, save_locally=True
    )
    assert secrets


if __name__ == "__main__":
    unittest.main()
