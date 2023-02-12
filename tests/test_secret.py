import unittest

import runhouse as rh


def test_get_all_secrets():
    secrets = rh.Secrets.download_into_env(save_locally=True)
    providers = rh.Secrets.builtin_providers(as_str=True)
    assert set(providers) == {"aws", "gcp", "sky", "hf"}
    assert secrets


def test_get_custom_provider_secrets():
    provider = "snowflake"
    provider_secrets = rh.Secrets.get(provider=provider)
    assert provider_secrets


def test_get_provider_secrets():
    provider = "aws"
    provider_secrets = rh.Secrets.get(provider=provider)
    assert provider_secrets


def test_upload_group_provider_secrets():
    provider = "aws"
    provider_secrets = rh.Secrets.put(provider=provider, group="ds_preproc2")
    assert provider_secrets


def test_upload_user_provider_custom_secrets():
    provider = "snowflake"
    rh.Secrets.put(provider=provider, secret={"secret_key": "abcdefg"})
    assert True


def test_upload_user_provider_enabled_secrets():
    provider = "aws"
    rh.Secrets.put(provider=provider)
    assert True


def test_upload_all_provider_secrets():
    rh.Secrets.extract_and_upload()
    assert True


def test_delete_provider_secrets():
    rh.Secrets.delete_from_vault(providers=["huggingface"])
    assert True


def test_sending_secrets_to_cluster():
    cluster = rh.cluster(name="^rh-cpu").up_if_not()

    configured_providers = rh.Secrets.configured_providers()

    rh.Secrets.to(cluster, providers=configured_providers)

    # Confirm the secrets now exist on the cluster
    for p in configured_providers:
        provider_name = p.PROVIDER_NAME
        p_str = str(p())
        command = [
            f"from runhouse.rns.secrets import {p_str}",
            f"print({p_str}.has_secrets_file())",
        ]
        status_codes = cluster.run_python(command)
        resp = status_codes[0][1].rstrip("\n").split("\n")[-1]
        if resp.lower() == "false":
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

    rh.logout(remove_from_rh_config=True)

    for provider in configured_providers:
        assert not configs.get(provider)

    assert not configs.get("token")


if __name__ == "__main__":
    unittest.main()
