import pytest

import runhouse as rh

from tests.conftest import init_args

base_secret_values = {"secret_key": "secret_val"}

provider_secret_values = {
    "aws": {"access_key": "test_access_key", "secret_key": "test_secret_key"},
    "azure": {"subscription_id": "test_subscription_id"},
    "gcp": {"client_id": "test_client_id", "client_secret": "test_client_secret"},
    "lambda": {"api_key": "test_api_key"},
    "github": {"oauth_token": "test_oauth_token"},
    "huggingface": {"token": "test_token"},
    "ssh": {"public_key": "test_public_key", "private_key": "test_private_key"},
    "sky": {"public_key": "test_public_key", "private_key": "test_private_key"},
    "custom_provider": base_secret_values,
}

providers = provider_secret_values.keys()


def _provider_secret(provider):
    name = f"_test_{provider}"
    values = provider_secret_values[provider]

    args = {"name": name, "provider": provider, "values": values}
    prov_secret = rh.provider_secret(**args)
    init_args[id(prov_secret)] = args

    return prov_secret


@pytest.fixture(scope="function")
def secret(request):
    """Parametrize over multiple secrets - useful for running the same test on multiple envs."""
    return request.getfixturevalue(request.param)


@pytest.fixture(scope="function")
def test_secret():
    args = {"name": "custom_secret", "values": base_secret_values}
    custom_secret = rh.secret(**args)
    init_args[id(custom_secret)] = args
    return custom_secret


@pytest.fixture(scope="function")
def aws_secret():
    return _provider_secret("aws")


@pytest.fixture(scope="function")
def azure_secret():
    return _provider_secret("azure")


@pytest.fixture(scope="function")
def gcp_secret():
    return _provider_secret("gcp")


@pytest.fixture(scope="function")
def lambda_secret():
    return _provider_secret("lambda")


@pytest.fixture(scope="function")
def github_secret():
    return _provider_secret("github")


@pytest.fixture(scope="function")
def huggingface_secret():
    return _provider_secret("huggingface")


@pytest.fixture(scope="function")
def ssh_secret():
    return _provider_secret("ssh")


@pytest.fixture(scope="function")
def sky_secret():
    return _provider_secret("sky")


@pytest.fixture(scope="function")
def custom_provider_secret():
    return _provider_secret("custom_provider")
