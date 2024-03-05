from pathlib import Path

import pytest

import runhouse as rh

from runhouse.constants import DEFAULT_HTTPS_PORT

from tests.conftest import init_args
from tests.utils import test_env


@pytest.fixture()
def restart_server(request):
    return request.config.getoption("--restart-server")


def setup_test_cluster(args, request):
    cluster = rh.ondemand_cluster(**args)
    init_args[id(cluster)] = args

    if not cluster.is_up():
        cluster.up()
    elif request.config.getoption("--restart-server"):
        cluster.restart_server()

    cluster.save()
    test_env().to(cluster)
    return cluster


@pytest.fixture(
    params=[
        "ondemand_aws_cluster",
        "ondemand_gcp_cluster",
        "ondemand_k8s_cluster",
        "v100_gpu_cluster",
        "k80_gpu_cluster",
        "a10g_gpu_cluster",
    ],
    ids=["aws_cpu", "gcp_cpu", "k8s_cpu", "v100", "k80", "a10g"],
)
def ondemand_cluster(request):
    return request.getfixturevalue(request.param)


@pytest.fixture(scope="session")
def ondemand_aws_cluster():
    args = {"name": "aws-cpu", "instance_type": "CPU:2+", "provider": "aws"}
    cluster = setup_test_cluster(args)
    return cluster


@pytest.fixture(scope="session")
def ondemand_aws_https_cluster_with_auth():
    args = {
        "name": "aws-cpu-https",
        "instance_type": "CPU:2+",
        "provider": "aws",
        "den_auth": True,
        "server_connection_type": "tls",
        # Use Caddy for SSL & reverse proxying (if port not specified here will launch certs with uvicorn)
        # "server_port": DEFAULT_HTTPS_PORT,
        "open_ports": [DEFAULT_HTTPS_PORT],
    }
    cluster = setup_test_cluster(args)
    return cluster


@pytest.fixture(scope="session")
def ondemand_gcp_cluster():
    args = {"name": "gcp-cpu", "instance_type": "CPU:2+", "provider": "gcp"}
    cluster = setup_test_cluster(args)
    return cluster


@pytest.fixture(scope="session")
def ondemand_k8s_cluster():
    kube_config_path = Path.home() / ".kube" / "config"

    if not kube_config_path.exists():
        pytest.skip("no kubeconfig found")

    args = {
        "name": "rh-cpu-k8s",
        "provider": "kubernetes",
        "instance_type": "1CPU--1GB",
    }
    cluster = setup_test_cluster(args)
    return cluster


@pytest.fixture(scope="session")
def v100_gpu_cluster():
    args = {"name": "rh-v100", "instance_type": "V100:1", "provider": "aws"}
    cluster = setup_test_cluster(args)
    return cluster


@pytest.fixture(scope="session")
def k80_gpu_cluster():
    args = {"name": "rh-k80", "instance_type": "K80:1", "provider": "aws"}
    cluster = setup_test_cluster(args)
    return cluster


@pytest.fixture(scope="session")
def a10g_gpu_cluster():
    args = {"name": "rh-a10x", "instance_type": "g5.2xlarge", "provider": "aws"}
    cluster = setup_test_cluster(args)
    return cluster


@pytest.fixture(scope="session")
def multinode_cpu_cluster(request):
    args = {
        "name": "rh-cpu-multinode",
        "num_instances": 2,
        "instance_type": "CPU:2+",
    }
    cluster = setup_test_cluster(args, request)
    return cluster
