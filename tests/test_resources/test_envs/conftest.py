import os

import pytest

import runhouse as rh

from tests.conftest import init_args


def _get_conda_env(name="rh-test", python_version="3.10.9"):
    conda_env = {
        "name": name,
        "channels": ["defaults"],
        "dependencies": [
            f"python={python_version}",
        ],
    }
    return conda_env


@pytest.fixture(scope="function")
def env(request):
    """Parametrize over multiple envs - useful for running the same test on multiple envs."""
    return request.getfixturevalue(request.param)


@pytest.fixture(scope="function")
def unnamed_env():
    args = {"reqs": ["npm"]}
    env = rh.env(**args)
    init_args[id(env)] = args
    return env


@pytest.fixture(scope="function")
def named_env():
    args = {"reqs": ["npm"], "name": "named_env"}
    env = rh.env(**args)
    init_args[id(env)] = args
    return env


@pytest.fixture(scope="function")
def base_conda_env():
    args = {"name": "conda_base", "reqs": ["pytest", "npm"]}
    env = rh.conda_env(**args)
    init_args[id(env)] = args
    return env


@pytest.fixture(scope="function")
def named_conda_env_from_dict():
    env_name = "conda_from_dict"
    conda_dict = _get_conda_env(name=f"{env_name}_env")

    args = {"name": env_name, "conda_env": conda_dict}
    env = rh.conda_env(**args)
    init_args[id(env)] = args
    return env


@pytest.fixture(scope="function")
def conda_env_from_path():
    env_name = "conda_from_path"
    file_path = os.path.join(os.path.dirname(__file__), "assets", "test_conda_env.yml")

    args = {"name": env_name, "conda_env": file_path}
    env = rh.conda_env(**args)
    init_args[id(env)] = args
    yield env


@pytest.fixture(scope="session")
def _local_conda_env():
    env_name = "test_conda_local_env"
    try:
        os.system(f"conda create -n {env_name} -y python==3.10.9")
        yield
    finally:
        os.system(f"conda env remove -n {env_name} -y")


@pytest.fixture(scope="function")
def conda_env_from_local(_local_conda_env):
    env_name = "test_conda_local_env"

    args = {"name": env_name, "conda_env": env_name}
    env = rh.conda_env(**args)
    init_args[id(env)] = args
    return env
