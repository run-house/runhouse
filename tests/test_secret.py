import json
import os

import pytest

import runhouse as rh
from dotenv import dotenv_values

from runhouse.resources.secrets.utils import load_config

secret_key = "secret_key"
secret_val = "secret_val"
addtl_key = "new_key"
addtl_val = "new_val"

base_secrets = {secret_key: secret_val}
addtl_secrets = {addtl_key: addtl_val}


def assert_deleted(name=None, path=None):
    if path:
        assert not os.path.exists(os.path.expanduser(path))
    if name:
        with pytest.raises(Exception) as e:
            load_config(name)
            assert isinstance(e, Exception)


# ------- BASE/CUSTOM SECRETS TEST ------- #


@pytest.mark.localtest
def test_create_secret_from_name_local():
    secret_name = "~/custom_secret"
    local_secret = rh.secret(name=secret_name, secrets=base_secrets).save()
    del local_secret

    reloaded_secret = rh.secret(name=secret_name)
    assert reloaded_secret.secrets == base_secrets

    reloaded_secret.delete()


@pytest.mark.rnstest
def test_create_secret_from_name_vault():
    secret_name = "vault_secret"
    vault_secret = rh.secret(name=secret_name, secrets=base_secrets).save()
    del vault_secret

    reloaded_secret = rh.secret(name=secret_name)
    assert reloaded_secret.secrets == base_secrets

    reloaded_secret.delete()
    assert_deleted(secret_name)


@pytest.mark.rnstest
def test_custom_secret():
    custom_path = "./custom.json"
    custom_name = "custom_secret"
    custom_secret = (
        rh.secret(name="custom_secret", secrets=base_secrets, path=custom_path)
        .write()
        .save()
    )

    assert custom_secret.secrets == base_secrets
    assert custom_secret._from_path(custom_path) == base_secrets
    assert load_config(custom_name)

    custom_secret.delete(file=True)
    assert_deleted(custom_name, custom_path)


@pytest.mark.clustertest
def test_custom_python_secret_to(ondemand_cpu_cluster):
    secret_name = "remote_secret"
    remote_path = "~/.rh/secrets/remote_secret.json"
    secret = rh.secret(name=secret_name, secrets=base_secrets).to(
        ondemand_cpu_cluster, path=remote_path
    )

    assert secret.secrets == base_secrets
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
        json.dump(base_secrets, f)
    secret = rh.secret(name=secret_name, path=local_path).to(
        ondemand_cpu_cluster, path=remote_path
    )

    assert secret.secrets == base_secrets
    assert secret.path.system == ondemand_cpu_cluster
    import pdb; pdb.set_trace()
    # TODO: file is created correctly on remote but for some reason following line fails
    assert secret.path.exists_in_system()

    secret.delete(file=True)
    assert not rh.file(path=remote_path).exists_in_system()


# ------- ENV SECRETS TEST ------- #


@pytest.mark.localtest
def test_write_load_env_secret_env_file():
    path = "test.env"
    env_secret = rh.env_secret(secrets=base_secrets)
    env_secret = env_secret.write(path=path)
    del env_secret

    env_vars = dotenv_values(path)
    assert env_vars[secret_key] == secret_val

    reloaded_secret = rh.env_secret(path=path)
    assert reloaded_secret.secrets == base_secrets

    reloaded_secret.delete()
    assert_deleted(path=path)


@pytest.mark.localtest
def test_write_load_env_to_python_env():
    env_secret = rh.env_secret(secrets=base_secrets)
    env_secret.write(python_env=True)
    del env_secret
    assert os.environ[secret_key] == secret_val

    reloaded_secret = rh.env_secret(env_vars=[secret_key])
    assert reloaded_secret.secrets == base_secrets

    reloaded_secret.delete(python_env=True)
    assert secret_key not in os.environ


@pytest.mark.localtest
def test_save_env_secret_local():
    local_name = "~/env_secret"
    env_secret = rh.env_secret(name=local_name, secrets=base_secrets)
    env_secret.save()
    del env_secret

    reloaded_secret = rh.env_secret(name=local_name)
    assert reloaded_secret.secrets == base_secrets

    new_secrets = rh.env_secret(name=local_name, secrets=addtl_secrets)
    new_secrets.save()

    reloaded_secret = rh.env_secret(name=local_name)
    assert reloaded_secret.secrets == {**base_secrets, **addtl_secrets}

    reloaded_secret.delete()


@pytest.mark.rnstest
def test_save_env_secret_vault():
    name = "env_secret"
    env_secret = rh.env_secret(name=name, secrets=base_secrets)
    env_secret.save()
    del env_secret

    reloaded_secret = rh.env_secret(name=name)
    assert reloaded_secret.secrets == base_secrets

    new_secrets = rh.env_secret(name=name, secrets=addtl_secrets)
    new_secrets.save()

    new_reloaded_secret = rh.env_secret(name=name)
    assert new_reloaded_secret.secrets == {**base_secrets, **addtl_secrets}

    new_reloaded_secret.delete()
    assert_deleted(name)


def test_extract_python_env_secret():
    for key in base_secrets.keys():
        os.environ[key] = base_secrets[key]
    python_env_secret = rh.env_secret(env_vars=list(base_secrets.keys()))
    assert python_env_secret.secrets == base_secrets


# TODO: tests for python env, partial keys
@pytest.mark.clustertest
def test_env_secret_to_cluster_file(ondemand_cpu_cluster):
    # TODO: not actually properly writing down into this file..
    env_secret = rh.env_secret(name="env_secret", secrets=base_secrets).to(
        ondemand_cpu_cluster, path="~/.rh/secrets/.env"
    )
    assert env_secret.secrets == base_secrets
    assert env_secret.path.system == ondemand_cpu_cluster
    # TODO: look into this
    assert env_secret.path.exists_in_system()

    env_secret.delete(file=True)
    assert not env_secret.path.exists_in_system()


# TODO: tests for deleteing certain keys from env, from file


# ------- PROVIDER SECRETS TEST ------- #

# AWS
@pytest.mark.rnstest
def test_custom_provider_secret():
    provider = "custom_provider"
    path = "~/custom_provider/config.json"
    custom_secret = rh.provider_secret(
        provider=provider, secrets=base_secrets, path=path
    )

    custom_secret.write()
    assert os.path.exists(os.path.expanduser(path))

    custom_secret.save()
    assert load_config(provider)

    del custom_secret
    reloaded_secret = rh.provider_secret(provider)
    assert reloaded_secret.secrets == base_secrets

    reloaded_secret.delete(file=True)
    assert_deleted(provider, path)


@pytest.mark.rnstest
def test_aws_secret_vault():
    # assumes have aws secrets stored locally
    aws_secret = rh.provider_secret("aws")
    aws_secret.save()

    assert aws_secret.in_vault()

    aws_secret.delete()
    assert not aws_secret.in_vault()


@pytest.mark.clustertest
def test_aws_secret_to_cluster(ondemand_cpu_cluster):
    local_aws = rh.provider_secret("aws")
    remote_aws = local_aws.to(ondemand_cpu_cluster, path="~/.aws/credentials")
    assert remote_aws.path.system == ondemand_cpu_cluster
    assert remote_aws.secrets == local_aws.secrets  # TODO: may print secrets when we add gha testing
    assert remote_aws.path.exists_in_system()

    remote_aws.delete(file=True)
    assert not remote_aws.path.exists_in_system()

# GCP


# Lambda

