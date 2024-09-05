import pytest

import runhouse as rh

from tests.conftest import init_args


def summer(a: int, b: int):
    print("Running summer function")
    return a + b


def save_and_load_artifacts():
    cpu = rh.ondemand_cluster("^rh-cpu").save()
    loaded_cluster = rh.load(name=cpu.name)
    return loaded_cluster.name


def slow_running_func(a, b):
    import time

    time.sleep(20)
    return a + b


@pytest.fixture(scope="session")
def summer_func(ondemand_aws_docker_cluster):
    args = {"name": "summer_func", "fn": summer}
    f = rh.function(**args).to(ondemand_aws_docker_cluster, env=["pytest"])
    init_args[id(f)] = args
    return f


@pytest.fixture(scope="session")
def summer_func_with_auth(ondemand_aws_https_cluster_with_auth):
    return rh.function(summer, name="summer_func").to(
        ondemand_aws_https_cluster_with_auth, env=["pytest"]
    )


@pytest.fixture(scope="session")
def summer_func_shared(shared_cluster):
    return rh.function(summer, name="summer_func").to(shared_cluster, env=["pytest"])


@pytest.fixture(scope="session")
def func_with_artifacts(ondemand_aws_docker_cluster):
    return rh.function(save_and_load_artifacts, name="artifacts_func").to(
        ondemand_aws_docker_cluster, env=["pytest"]
    )


@pytest.fixture(scope="session")
def slow_func(ondemand_aws_docker_cluster):
    return rh.function(slow_running_func, name="slow_func").to(
        ondemand_aws_docker_cluster, env=["pytest"]
    )
