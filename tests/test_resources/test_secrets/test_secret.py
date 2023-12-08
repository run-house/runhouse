import json
import os

import pytest
import requests

import runhouse as rh

from runhouse.globals import rns_client

from runhouse.resources.secrets.utils import load_config

import tests.test_resources.test_resource

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


def assert_delete_local(secret, contents: bool = False):
    secret.delete(contents=contents) if contents else secret.delete()
    with pytest.raises(Exception) as e:
        rh.secret(name=secret.name)
        assert isinstance(e, Exception)
    if contents and secret.path:
        assert not os.path.exists(os.path.expanduser(secret.path))


class TestSecret(tests.test_resources.test_resource.TestResource):
    MAP_FIXTURES = {"resource": "secret"}

    UNIT = {"secret": ["custom_secret"] + provider_secrets}
    LOCAL = {
        "secret": ["custom_secret"] + provider_secrets,
        "cluster": [
            "local_docker_cluster_public_key_logged_in",
            "local_docker_cluster_public_key_logged_out",
        ],
    }
    MINIMAL = {
        "secret": [
            "custom_secret",
            "aws_secret",
            "ssh_secret",
            "custom_provider_secret",
        ],
        "cluster": ["static_cpu_cluster"],
    }
    THOROUGH = {
        "secret": ["custom_secret"] + provider_secrets,
        "cluster": ["static_cpu_cluster"],
    }
    MAXIMAL = {
        "secret": ["custom_secret"] + provider_secrets,
        "cluster": ["static_cpu_cluster", "password_cluster"],
    }

    @pytest.mark.level("unit")
    def test_secret_factory_and_properties(self, secret):
        assert isinstance(secret, rh.Secret)

    @pytest.mark.level("local")
    def test_provider_secret_to_cluster(self, secret, cluster):
        if not isinstance(secret, rh.ProviderSecret):
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
    def test_sharing(self, secret, test_account):
        username_to_share = rh.configs.defaults_cache["username"]

        # Create & share
        with test_account:
            test_headers = rns_client.request_headers
            secret.save(headers=test_headers)

            rns_address = secret.rns_address

            # Share the resource (incl. access to the secrets in Vault)
            secret.share(username_to_share, access_type="write")

        # By default we can re-load shared secrets
        reloaded_secret = rh.secret(name=rns_address)
        assert reloaded_secret.values == secret.values

    @pytest.mark.level("local")
    def test_sync_secrets(self, secret, cluster):
        if not isinstance(secret, rh.ProviderSecret):
            return
        test_path = _provider_path_map[secret.provider]
        secret = secret.write(path=test_path)
        cluster.sync_secrets([secret])

        remote_file = rh.file(path=secret.path, system=cluster)
        assert remote_file.exists_in_system()
        assert secret._from_path(remote_file) == secret.values

        assert_delete_local(secret, contents=True)
        remote_file.rm()

    @pytest.mark.level("unit")
    def test_convert_secret_resource(self, secret):
        from runhouse.rns.login import _convert_secrets_resource

        name = secret.name
        values = secret.values
        requests.put(
            f"{rns_client.api_server_url}/{rh.Secret.USER_ENDPOINT}/{name}",
            data=json.dumps(values),
            headers=rns_client.request_headers,
        )

        _convert_secrets_resource([name])
        assert load_config(name)

        rh.Secret.from_name(name).delete()
        with pytest.raises(ValueError):
            load_config(name)
