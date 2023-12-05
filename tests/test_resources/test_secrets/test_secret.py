import os

import pytest

import runhouse as rh
from pytest_lazyfixture import lazy_fixture

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


def assert_delete_local(secret, contents: bool = False):
    secret.delete(contents=contents) if contents else secret.delete()
    with pytest.raises(Exception) as e:
        rh.secret(name=secret.name)
        assert isinstance(e, Exception)
    if contents and secret.path:
        assert not os.path.exists(os.path.expanduser(secret.path))


class TestSecret(tests.test_resources.test_resource.TestResource):
    MAP_FIXTURES = {"resource": "secret"}

    UNIT = {"secret": ["custom_secret", lazy_fixture("provider_secret")]}
    # LOCAL = {
    #     "secret": ["custom_secret", "provider_secrets"],
    #     "cluster": ["local_docker_cluster_public_key_logged_in"]
    # }

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
