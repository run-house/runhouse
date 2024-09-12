import os

import pytest

import runhouse as rh

from runhouse.globals import rns_client

import tests.test_resources.test_resource

from tests.utils import friend_account

_provider_path_map = {
    "aws": "credentials",
    "gcp": "credentials.json",
    "azure": "clouds.config",
    "lambda": "lambda_key",
    "github": "hosts.yml",
    "huggingface": "token",
    "ssh": "id_rsa",
    "sky": "sky-key",
    "custom_provider": "~/.rh/tests/custom_provider/config.json",
}

provider_secrets = [
    "aws_secret",
    "azure_secret",
    "gcp_secret",
    "lambda_secret",
    "github_secret",
    "huggingface_secret",
    "ssh_secret",
    "sky_secret",
    "custom_provider_secret",
]

api_secrets = [
    "anthropic_secret",
    "cohere_secret",
    "langchain_secret",
    "openai_secret",
    "pinecone_secret",
    "wandb_secret",
]


def assert_delete_local(secret, contents: bool = False):
    secret.delete(contents=contents) if contents else secret.delete()
    with pytest.raises(Exception) as e:
        rh.secret(name=secret.name)
        assert isinstance(e, Exception)
    if contents and secret.path:
        assert not os.path.exists(os.path.expanduser(secret.path))


def _get_env_var_value(env_var):
    import os

    return os.environ[env_var]


@pytest.mark.secrettest
class TestSecret(tests.test_resources.test_resource.TestResource):
    MAP_FIXTURES = {"resource": "secret"}

    UNIT = {"secret": ["test_secret"] + provider_secrets + api_secrets}
    LOCAL = {
        "secret": ["test_secret"] + provider_secrets + api_secrets,
        "cluster": [
            "docker_cluster_pk_ssh_no_auth",
        ],
    }
    MINIMAL = {
        "secret": [
            "test_secret",
            "aws_secret",
            "ssh_secret",
            "openai_secret",
            "custom_provider_secret",
        ],
        "cluster": ["ondemand_aws_docker_cluster"],
    }
    RELEASE = {
        "secret": ["test_secret"] + provider_secrets,
        "cluster": [
            "ondemand_aws_docker_cluster",
            "static_cpu_pwd_cluster",
        ],
    }
    MAXIMAL = {
        "secret": ["test_secret"] + provider_secrets,
        "cluster": [
            "ondemand_aws_docker_cluster",
            "ondemand_gcp_cluster",
            "ondemand_k8s_cluster",
            "ondemand_k8s_docker_cluster",
            "ondemand_aws_https_cluster_with_auth",
            "static_cpu_pwd_cluster",
            "multinode_cpu_docker_conda_cluster",
            "docker_cluster_pk_ssh_no_auth",
            "docker_cluster_pwd_ssh_no_auth",
            "docker_cluster_pk_ssh_den_auth",
        ],
    }

    @pytest.mark.level("unit")
    def test_secret_factory_and_properties(self, secret):
        assert isinstance(secret, rh.Secret)

    @pytest.mark.level("local")
    def test_provider_secret_to_cluster_values(self, secret, cluster):
        remote_secret = secret.to(cluster)
        assert cluster.get(remote_secret.name)
        assert cluster.get(remote_secret.name).values
        cluster.delete(remote_secret.name)

    @pytest.mark.level("local")
    def test_provider_secret_to_cluster_path(self, secret, cluster):
        if not isinstance(secret, rh.ProviderSecret):
            return

        if secret.name not in _provider_path_map.keys():
            return

        test_path = os.path.join("~/.rh/tests", _provider_path_map[secret.provider])
        remote_secret = secret.to(cluster, path=test_path)

        assert remote_secret.path.system == cluster
        assert remote_secret.values == secret.values
        assert remote_secret.path.exists_in_system()

        delete_contents = secret.provider not in ["ssh", "sky"]
        remote_secret.delete(contents=delete_contents)
        assert_delete_local(secret, contents=delete_contents)

    @pytest.mark.level("local")
    def test_provider_secret_to_cluster_env(self, secret, cluster):
        if not isinstance(secret, rh.ProviderSecret):
            return

        env_vars = secret._DEFAULT_ENV_VARS
        if not env_vars:
            return

        env = rh.env()
        get_remote_val = rh.function(_get_env_var_value, name="get_env_vars").to(
            cluster, env=env
        )
        secret.to(cluster, env=env)

        for (key, val) in env_vars.items():
            assert get_remote_val(val) == secret.values[key]

    @pytest.mark.level("unit")
    def test_share_and_revoke_secret(self, test_secret):
        vault_secret = test_secret.save(f"{test_secret.name}_shared")
        vault_secret.share(
            users=["info@run.house"],
            access_level="read",
            notify_users=False,
        )

        with friend_account():
            reloaded_secret = rh.secret(name=vault_secret.rns_address)
            assert reloaded_secret.values == test_secret.values

        vault_secret.revoke(users=["info@run.house"])
        with pytest.raises(Exception):
            with friend_account():
                rh.secret(name=vault_secret.rns_address)

    @pytest.mark.level("unit")
    def test_sharing_public_secret(self, test_secret):
        # Create & share
        with friend_account():
            test_headers = rns_client.request_headers()
            vault_secret = rh.secret(name=test_secret.name, values=test_secret.values)
            vault_secret.save(headers=test_headers)

            rns_address = vault_secret.rns_address

            # Make the resource available to all users, without explicitly sharing with any users
            vault_secret.share(visibility="public", notify_users=False)

            del vault_secret

        # By default we can re-load the public resource
        reloaded_secret = rh.secret(name=rns_address)

        # NOTE: currently not loading the values for public secret resources (i.e. reloaded_secret.values will be empty)
        assert reloaded_secret

    @pytest.mark.level("local")
    def test_sync_secrets(self, secret, cluster):
        if not isinstance(secret, rh.ProviderSecret):
            return

        if secret.provider in _provider_path_map.keys():
            test_path = _provider_path_map[secret.provider]
            secret = secret.write(path=test_path)
            cluster.sync_secrets([secret])

            remote_folder = rh.folder(path=secret.path, system=cluster)
            assert remote_folder.exists_in_system()
            assert secret._from_path(remote_folder.path) == secret.values

            assert_delete_local(secret, contents=True)
            remote_folder.rm()
        else:
            cluster.sync_secrets([secret])
            assert cluster.get(secret.name)
