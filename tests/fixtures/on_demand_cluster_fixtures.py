from pathlib import Path

import pytest

import runhouse as rh

from runhouse.constants import DEFAULT_HTTPS_PORT
from runhouse.resources.images.image import Image
from tests.conftest import init_args

from tests.constants import TEST_ENV_VARS
from tests.utils import test_env

NUM_OF_NODES = 2


@pytest.fixture()
def restart_server(request):
    return request.config.getoption("--restart-server")


def setup_test_cluster(args, request, create_env=False):
    cluster = rh.ondemand_cluster(**args)
    init_args[id(cluster)] = args
    cluster.up_if_not()
    if request.config.getoption("--restart-server"):
        cluster.restart_server()

    cluster.save()

    if create_env or not cluster.image:
        test_env().to(cluster)
    return cluster


@pytest.fixture(
    params=[
        "ondemand_aws_docker_cluster",
        "ondemand_gcp_cluster",
        "ondemand_k8s_cluster",
        "ondemand_k8s_docker_cluster",
        "v100_gpu_cluster",
        "k80_gpu_cluster",
        "a10g_gpu_cluster",
    ],
    ids=["aws_cpu", "gcp_cpu", "k8s_cpu", "k8s_docker_cpu", "v100", "k80", "a10g"],
)
def ondemand_cluster(request):
    return request.getfixturevalue(request.param)


@pytest.fixture(scope="session")
def ondemand_aws_docker_cluster(request):
    """
    Note: Also used to test docker and default env with alternate Ray version.
    """
    image = (
        Image(name="default_image")
        .from_docker("rayproject/ray:latest-py311-cpu")
        .install_packages(["ray==2.30.0"])
    )
    args = {
        "name": "aws-cpu",
        "instance_type": "CPU:2+",
        "provider": "aws",
        "region": "us-east-2",
        "image": image,
        "sky_kwargs": {"launch": {"retry_until_up": True}},
    }
    cluster = setup_test_cluster(args, request, create_env=True)
    return cluster


@pytest.fixture(scope="session")
def ondemand_aws_https_cluster_with_auth(request, test_rns_folder):
    args = {
        # creating a unique name everytime, so the certs will be freshly generated on every test run.
        "name": f"{test_rns_folder}_aws-cpu-https",
        "instance_type": "CPU:2+",
        "provider": "aws",
        "den_auth": True,
        "server_connection_type": "tls",
        # Use Caddy for SSL & reverse proxying (if port not specified here will launch certs with uvicorn)
        # "server_port": DEFAULT_HTTPS_PORT,
        "open_ports": [DEFAULT_HTTPS_PORT],
    }
    cluster = setup_test_cluster(args, request)
    return cluster


@pytest.fixture(scope="session")
def ondemand_gcp_cluster(request):
    """
    Note: Also used to test conda default env.
    """
    image = (
        Image(name="default_image")
        .setup_conda_env(
            conda_env_name="base_env",
            conda_yaml={"dependencies": ["python=3.11"], "name": "base_env"},
        )
        .install_packages(test_env().reqs + ["ray==2.30.0"], conda_env_name="base_env")
        .set_env_vars(env_vars=TEST_ENV_VARS)
    )
    args = {
        "name": "gcp-cpu",
        "instance_type": "CPU:2+",
        "provider": "gcp",
        "image": image,
    }
    cluster = setup_test_cluster(args, request)
    return cluster


@pytest.fixture(scope="session")
def ondemand_k8s_cluster(request):
    kube_config_path = Path.home() / ".kube" / "config"

    if not kube_config_path.exists():
        pytest.skip("no kubeconfig found")

    # Note: Cannot specify both `instance_type` and any of `memory`, `disk_size`, `num_cpus`, or `accelerators`
    args = {
        "name": "k8s-cpu",
        "provider": "kubernetes",
        "instance_type": "CPU:1",
    }
    cluster = setup_test_cluster(args, request)
    return cluster


@pytest.fixture(scope="session")
def ondemand_k8s_docker_cluster(request):
    kube_config_path = Path.home() / ".kube" / "config"

    if not kube_config_path.exists():
        pytest.skip("no kubeconfig found")

    args = {
        "name": "k8s-docker-cpu",
        "provider": "kubernetes",
        "instance_type": "CPU:1",
        "memory": ".2",
        "image": Image(name="default_image").from_docker(
            "rayproject/ray:latest-py311-cpu"
        ),
    }
    cluster = setup_test_cluster(args, request)
    return cluster


@pytest.fixture(scope="session")
def v100_gpu_cluster(request):
    args = {
        "name": "rh-v100",
        "instance_type": "V100:1",
        "provider": "aws",
    }
    cluster = setup_test_cluster(args, request)
    return cluster


@pytest.fixture(scope="session")
def k80_gpu_cluster(request):
    args = {
        "name": "rh-k80",
        "instance_type": "K80:1",
        "provider": "aws",
    }
    cluster = setup_test_cluster(args, request)
    return cluster


@pytest.fixture(scope="session")
def a10g_gpu_cluster(request):
    args = {
        "name": "rh-a10x",
        "instance_type": "g5.2xlarge",
        "provider": "aws",
    }
    cluster = setup_test_cluster(args, request)
    return cluster


@pytest.fixture(scope="session")
def multinode_k8s_cpu_cluster(request):
    args = {
        "name": "rh-cpu-multinode",
        "num_instances": NUM_OF_NODES,
        "provider": "kubernetes",
        "instance_type": "CPU:2+",
    }
    cluster = setup_test_cluster(args, request)
    return cluster


@pytest.fixture(scope="session")
def multinode_cpu_docker_conda_cluster(request):
    image = (
        Image(name="default_image")
        .from_docker("rayproject/ray:latest-py311-cpu")
        .setup_conda_env(
            conda_env_name="base_env",
            conda_yaml={"dependencies": ["python=3.11"], "name": "base_env"},
        )
        .install_packages(test_env().reqs + ["ray==2.30.0"], conda_env_name="base_env")
    )
    args = {
        "name": "rh-cpu-multinode",
        "num_nodes": NUM_OF_NODES,
        "image": image,
        "provider": "aws",
        "instance_type": "CPU:2+",
    }
    cluster = setup_test_cluster(args, request)
    return cluster


@pytest.fixture(scope="session")
def multinode_gpu_cluster(request):
    args = {
        "name": "rh-gpu-multinode",
        "num_nodes": NUM_OF_NODES,
        "instance_type": "g5.xlarge",
    }
    cluster = setup_test_cluster(args, request)
    return cluster
