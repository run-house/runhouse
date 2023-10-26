import json
import os

import pytest

import runhouse as rh
from runhouse.resources.blobs import file

from runhouse.resources.secrets.utils import load_config

secret_key = "secret_key"
secret_val = "secret_val"
addtl_key = "new_key"
addtl_val = "new_val"

base_values = {secret_key: secret_val}
addtl_values = {addtl_key: addtl_val}


def assert_delete_local(secret, file=False):
    secret.delete(file=True) if file else secret.delete()
    if file:
        assert not os.path.exists(os.path.expanduser(secret.path))
    with pytest.raises(Exception) as e:
        load_config(secret.name)
        assert isinstance(e, Exception)


# ------- BASE/CUSTOM SECRETS TEST ------- #


@pytest.mark.localtest
def test_create_secret_from_name_local():
    secret_name = "~/custom_secret"
    local_secret = rh.secret(name=secret_name, values=base_values).save()
    del local_secret

    reloaded_secret = rh.secret(name=secret_name)
    assert reloaded_secret.values == base_values

    reloaded_secret.delete()


@pytest.mark.rnstest
def test_create_secret_from_name_vault():
    secret_name = "vault_secret"
    vault_secret = rh.secret(name=secret_name, values=base_values).save()
    del vault_secret

    reloaded_secret = rh.secret(name=secret_name)
    assert reloaded_secret.values == base_values

    assert_delete_local(reloaded_secret)


@pytest.mark.rnstest
def test_custom_secret():
    custom_path = "~/custom.json"
    custom_name = "custom_secret"
    custom_secret = (
        rh.secret(name="custom_secret", values=base_values, path=custom_path)
        .write()
        .save()
    )

    assert custom_secret.values == base_values
    assert custom_secret._from_path(custom_path) == base_values
    assert load_config(custom_name)

    assert_delete_local(custom_secret, file=True)


@pytest.mark.clustertest
def test_custom_python_secret_to(ondemand_cpu_cluster):
    secret_name = "remote_secret"
    remote_path = "~/.rh/secrets/remote_secret.json"
    secret = rh.secret(name=secret_name, values=base_values).to(
        ondemand_cpu_cluster, path=remote_path
    )

    assert secret.values == base_values
    assert secret.path.system == ondemand_cpu_cluster
    assert secret.path.exists_in_system()

    secret.delete(file=True)
    assert not rh.file(path=remote_path).exists_in_system()


@pytest.mark.clustertest
def test_custom_file_secret_to(ondemand_cpu_cluster):
    local_path = "file_secret.json"
    secret_name = "remote_file_secret"
    remote_path = "~/.rh/secrets/remote_file_secret.json"

    with open(local_path, "w") as f:
        json.dump(base_values, f)
    secret = rh.secret(name=secret_name, path=local_path).to(
        ondemand_cpu_cluster, path=remote_path
    )

    assert secret.values == base_values
    assert secret.path.system == ondemand_cpu_cluster
    assert secret.path.exists_in_system()

    secret.delete(file=True)
    assert not rh.file(path=remote_path, system=ondemand_cpu_cluster).exists_in_system()


# ------- PROVIDER SECRETS TEST ------- #


@pytest.mark.rnstest
def test_custom_provider_secret():
    provider = "custom_provider"
    path = "~/custom_provider/config.json"
    custom_secret = rh.provider_secret(provider=provider, values=base_values, path=path)

    custom_secret.write()
    assert os.path.exists(os.path.expanduser(path))

    custom_secret.save()
    assert load_config(provider)

    del custom_secret
    reloaded_secret = rh.provider_secret(name=provider)
    assert reloaded_secret.values == base_values

    assert_delete_local(reloaded_secret, file=True)
    assert not reloaded_secret.in_vault()


# Provider Secrets
gcp_secret_values = {
    "client_id": "test_client_id",
    "client_secret": "test_client_secret",
}

aws_secret_values = {
    "access_key": "test_access_key",
    "secret_key": "test_secret_key",
}

azure_secret_values = {"subscription_id": "test_subscription_id"}
lambda_secret_values = {"api_key": "test_api_key"}
github_secret_values = {"oauth_token": "test_oauth_token"}
huggingface_secret_values = {"token": "test_token"}

ssh_secret_values = {
    "public_key": "test_public_key",
    "private_key": "test_private_key",
}

provider_params = [
    ("aws", "credentials", aws_secret_values),
    ("gcp", "credentials.json", gcp_secret_values),
    ("azure", "clouds.config", azure_secret_values),
    ("lambda", "lambda_key", lambda_secret_values),
    ("github", "hosts.yml", github_secret_values),
    ("huggingface", "token", huggingface_secret_values),
    ("ssh", "id_rsa", ssh_secret_values),
    ("sky", "sky-key", ssh_secret_values),
]


@pytest.mark.parametrize("provider,path,values", provider_params)
def test_local_provider_secrets(provider, path, values):
    test_path = os.path.join("~/.rh/tests", path)
    test_name = "_" + provider
    rh.provider_secret(
        name=test_name, provider=provider, path=test_path, values=values
    ).write()
    assert os.path.exists(os.path.expanduser(test_path))

    local_secret = rh.provider_secret(name=test_name, provider=provider, path=test_path)
    assert local_secret.values == values

    assert_delete_local(local_secret, file=False)

    if provider not in ["ssh", "sky"]:
        assert_delete_local(local_secret, file=True)


@pytest.mark.rnstest
@pytest.mark.parametrize("provider,path,values", provider_params)
def test_vault_provider_secrets(provider, path, values):
    test_path = os.path.join("~/.rh/tests", path)
    test_name = "_" + provider
    rh.provider_secret(
        name=test_name, provider=provider, path=test_path, values=values
    ).save()

    reloaded_secret = rh.provider_secret(name=test_name)
    assert reloaded_secret.values == values

    assert_delete_local(reloaded_secret, file=False)
    assert not reloaded_secret.in_vault()


@pytest.mark.clustertest
@pytest.mark.parametrize("provider,path,values", provider_params)
def test_provider_secret_to_cluster(provider, path, values, ondemand_cpu_cluster):
    test_path = os.path.join("~/.rh/tests", path)
    test_name = "_" + provider
    local_secret = rh.provider_secret(
        name=test_name, provider=provider, path=test_path, values=values
    ).write()
    remote_secret = local_secret.to(ondemand_cpu_cluster, path=local_secret.path)

    assert remote_secret.path.system == ondemand_cpu_cluster
    assert remote_secret.values == local_secret.values
    assert remote_secret.path.exists_in_system()

    remote_secret.delete(file=True)
    if provider not in ["ssh", "sky"]:
        assert not remote_secret.path.exists_in_system()

    if provider not in ["ssh", "sky"]:
        assert_delete_local(local_secret, file=True)


# Other Secrets functionality tests


@pytest.mark.clustertest
def test_sync_secrets(ondemand_cpu_cluster):
    aws_secret = rh.provider_secret(
        provider="aws",
        name="_aws",
        values=aws_secret_values,
        path="~/.rh/tests/aws_credentials",
    ).write()
    ondemand_cpu_cluster.sync_secrets([aws_secret])
    remote_file = file(path=aws_secret.path, system=ondemand_cpu_cluster)
    assert remote_file.exists_in_system()
    assert aws_secret._from_path(remote_file) == aws_secret_values
