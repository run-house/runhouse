from pathlib import Path

import pytest

import runhouse as rh

from runhouse.constants import DEFAULT_HTTPS_PORT

from tests.conftest import init_args
from tests.utils import friend_account, test_env


@pytest.fixture(scope="session")
def on_demand_cluster(request):
    return request.getfixturevalue(request.param)


@pytest.fixture(
    params=[
        "ondemand_cpu_cluster",
        "v100_gpu_cluster",
        "k80_gpu_cluster",
        "a10g_gpu_cluster",
        "kubernetes_cpu_cluster",
    ],
    ids=["cpu", "v100", "k80", "a10g", "kubernetes"],
)
def ondemand_cluster(request):
    return request.getfixturevalue(request.param)


@pytest.fixture(scope="session")
def ondemand_cpu_cluster():
    args = {"name": "^rh-cpu"}
    c = rh.ondemand_cluster(**args)
    init_args[id(c)] = args

    c.up_if_not()

    # Save to RNS - to be loaded in other tests (ex: Runs)
    c.save()

    test_env().to(c)
    return c


@pytest.fixture(scope="session")
def v100_gpu_cluster():
    return rh.ondemand_cluster(
        name="rh-v100", instance_type="V100:1", provider="aws"
    ).up_if_not()


@pytest.fixture(scope="session")
def k80_gpu_cluster():
    return rh.ondemand_cluster(
        name="rh-k80", instance_type="K80:1", provider="aws"
    ).up_if_not()


@pytest.fixture(scope="session")
def a10g_gpu_cluster():
    return rh.ondemand_cluster(
        name="rh-a10x", instance_type="g5.2xlarge", provider="aws"
    ).up_if_not()


@pytest.fixture(scope="session")
def ondemand_https_cluster_with_auth():
    args = {
        "name": "rh-cpu-https",
        "instance_type": "CPU:2+",
        "den_auth": True,
        "server_connection_type": "tls",
        # Use Caddy for SSL & reverse proxying (if port not specified here will launch certs with uvicorn)
        "server_port": DEFAULT_HTTPS_PORT,
        "open_ports": [DEFAULT_HTTPS_PORT],
    }
    c = rh.ondemand_cluster(**args)
    c.up_if_not()
    init_args[id(c)] = args

    test_env().to(c)
    return c


@pytest.fixture(scope="session")
def shared_ondemand_cluster_with_auth():
    username_to_share = rh.configs.username

    with friend_account():
        args = {
            "name": "rh-cpu-shared-https",
            "instance_type": "CPU:2+",
            "den_auth": True,
            "server_connection_type": "tls",
            "server_port": DEFAULT_HTTPS_PORT,
            "open_ports": [DEFAULT_HTTPS_PORT],
        }
        c = rh.ondemand_cluster(**args)
        c.up_if_not()
        init_args[id(c)] = args

        test_env().to(c)

        # Enable den auth to properly test resource access on shared resources
        c.enable_den_auth()

        # Share the cluster with the test account
        c.share(username_to_share, notify_users=False, access_level="read")

    return c


@pytest.fixture(scope="session")
def multinode_cpu_cluster():
    args = {
        "name": "rh-cpu-multinode",
        "num_instances": 2,
        "instance_type": "CPU:2+",
    }
    c = rh.ondemand_cluster(**args)
    init_args[id(c)] = args

    c.up_if_not()

    c.save()

    test_env().to(c)
    return c


@pytest.fixture(scope="session")
def kubernetes_cpu_cluster():
    kube_config_path = Path.home() / ".kube" / "config"

    if not kube_config_path.exists():
        pytest.skip("no kubeconfig found")

    args = {
        "name": "rh-cpu-k8s",
        "provider": "kubernetes",
        "instance_type": "1CPU--1GB",
    }
    c = rh.ondemand_cluster(**args)
    init_args[id(c)] = args

    c.up_if_not()

    # Save to RNS - to be loaded in other tests (ex: Runs)
    c.save()

    # Call save before installing in the event we want to use TLS / den auth
    test_env().to(c)
    return c
