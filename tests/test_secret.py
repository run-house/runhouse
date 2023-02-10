import unittest

import runhouse as rh


def test_get_all_secrets():
    secrets = rh.Secrets.download_into_env(save_locally=False)
    providers = rh.Secrets.enabled_providers(as_str=True)
    # TODO this check is dependent on local secrets config, not a catch all
    assert set(providers) == {"aws", "gcp", "lambda", "ssh", "huggingface"}
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
    rh.Secrets.extract_and_upload(interactive=False)
    assert True


def test_delete_provider_secrets():
    rh.Secrets.delete(providers=["huggingface"])
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


if __name__ == "__main__":
    unittest.main()
