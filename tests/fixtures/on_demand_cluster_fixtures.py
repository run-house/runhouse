import os
from pathlib import Path

import pytest

import runhouse as rh

from runhouse.constants import DEFAULT_HTTPS_PORT
from runhouse.resources.hardware.utils import LauncherType
from runhouse.resources.images.image import Image
from tests.conftest import init_args

from tests.constants import TEST_ENV_VARS, TEST_REQS
from tests.utils import setup_test_base

NUM_OF_NODES = 2


@pytest.fixture()
def restart_server(request):
    return request.config.getoption("--restart-server")


def teardown_cluster_fixture(request, cluster):
    if not request.config.getoption("--detached") and cluster.is_up():
        cluster.teardown()


def setup_test_cluster(args, request, setup_base=False):
    cluster = rh.ondemand_cluster(**args)
    init_args[id(cluster)] = args
    cluster.up_if_not()
    if request.config.getoption("--restart-server"):
        cluster.restart_server()

    cluster.save()

    if setup_base or not cluster.image:
        setup_test_base(cluster)

    return cluster


@pytest.fixture(
    params=[
        "local_launched_ondemand_aws_docker_cluster",
        "den_launched_ondemand_aws_docker_cluster",
        "ondemand_gcp_cluster",
        "ondemand_k8s_cluster",
        "ondemand_k8s_docker_cluster",
        "v100_gpu_cluster",
        "den_launcher_v100_gpu_cluster",
        "k80_gpu_cluster",
        "a10g_gpu_cluster",
        "den_launched_ondemand_aws_k8s_cluster",
        "den_launched_ondemand_gcp_k8s_cluster",
    ],
    ids=[
        "aws_cpu",
        "aws_gpu_den_launcher",
        "gcp_cpu",
        "k8s_cpu",
        "k8s_docker_cpu",
        "v100",
        "v100_den_launcher",
        "k80",
        "a10g",
        "aws_k8_den_launcher",
        "gcp_k8_den_launcher",
    ],
)
def ondemand_cluster(request):
    return request.getfixturevalue(request.param)


@pytest.fixture(scope="session")
def local_launched_ondemand_aws_docker_cluster(request, test_rns_folder):
    """
    Note: Also used to test docker and default process with alternate Ray version.
    """
    image = (
        Image(name="default_image")
        .from_docker("rayproject/ray:latest-py311-cpu")
        .install_packages(TEST_REQS + ["ray==2.30.0"])
        .set_env_vars(TEST_ENV_VARS)
    )
    cluster_name = (
        "aws-cpu"
        if not request.config.getoption("--ci")
        else f"{test_rns_folder}-aws-cpu"
    )
    args = {
        "name": cluster_name,
        "instance_type": "CPU:2+",
        "provider": "aws",
        "region": "us-east-2",
        "image": image,
        "sky_kwargs": {"launch": {"retry_until_up": True}},
    }

    cluster = setup_test_cluster(args, request, setup_base=True)
    yield cluster
    teardown_cluster_fixture(request, cluster)


@pytest.fixture(scope="session")
def den_launched_ondemand_aws_docker_cluster(request, test_rns_folder):
    """
    Note: Also used to test docker and default env with alternate Ray version.
    """
    image = (
        Image(name="default_image")
        .from_docker("rayproject/ray:latest-py311-cpu")
        .install_packages(TEST_REQS + ["ray==2.30.0"])
        .set_env_vars(TEST_ENV_VARS)
    )
    cluster_name = (
        "aws-cpu-den"
        if not request.config.getoption("--ci")
        else f"{test_rns_folder}-aws-cpu-den"
    )
    args = {
        "name": cluster_name,
        "instance_type": "CPU:2+",
        "provider": "aws",
        "region": "us-east-2",
        "image": image,
        "sky_kwargs": {"launch": {"retry_until_up": True}},
        "launcher": LauncherType.DEN,
    }

    cluster = setup_test_cluster(args, request, setup_base=True)
    yield cluster
    teardown_cluster_fixture(request, cluster)


@pytest.fixture(scope="session")
def ondemand_aws_https_cluster_with_auth(request, test_rns_folder):
    cluster_name = (
        "aws-cpu-https"
        if not request.config.getoption("--ci")
        else f"{test_rns_folder}-aws-cpu-https"
    )
    args = {
        # creating a unique name everytime, so the certs will be freshly generated on every test run.
        "name": cluster_name,
        "instance_type": "CPU:2+",
        "provider": "aws",
        "den_auth": True,
        "server_connection_type": "tls",
        # Use Caddy for SSL & reverse proxying (if port not specified here will launch certs with uvicorn)
        # "server_port": DEFAULT_HTTPS_PORT,
        "open_ports": [DEFAULT_HTTPS_PORT],
    }

    cluster = setup_test_cluster(args, request)
    yield cluster
    teardown_cluster_fixture(request, cluster)


@pytest.fixture(scope="session")
def ondemand_gcp_cluster(request, test_rns_folder):
    """
    Note: Also used to test conda default env.
    """
    image = (
        Image(name="default_image")
        .setup_conda_env(
            conda_env_name="base_env",
            conda_config={"dependencies": ["python=3.11"], "name": "base_env"},
        )
        .install_packages(TEST_REQS + ["ray==2.30.0"], conda_env_name="base_env")
        .set_env_vars(env_vars=TEST_ENV_VARS)
    )
    cluster_name = (
        "gcp-cpu"
        if not request.config.getoption("--ci")
        else f"{test_rns_folder}-gcp-cpu"
    )
    args = {
        "name": cluster_name,
        "instance_type": "CPU:2+",
        "provider": "gcp",
        "image": image,
    }

    cluster = setup_test_cluster(args, request)
    yield cluster
    teardown_cluster_fixture(request, cluster)


@pytest.fixture(scope="session")
def ondemand_k8s_cluster(request, test_rns_folder):
    kube_config_path = Path.home() / ".kube" / "config"

    if not kube_config_path.exists():
        pytest.skip("no kubeconfig found")

    cluster_name = (
        "k8s-cpu"
        if not request.config.getoption("--ci")
        else f"{test_rns_folder}-k8s-cpu"
    )
    # Note: Cannot specify both `instance_type` and any of `memory`, `disk_size`, `num_cpus`, or `gpus`
    args = {
        "name": cluster_name,
        "provider": "kubernetes",
        "instance_type": "CPU:1",
        "den_auth": True,
    }

    cluster = setup_test_cluster(args, request)
    yield cluster
    teardown_cluster_fixture(request, cluster)


@pytest.fixture(scope="session")
def den_launched_ondemand_aws_k8s_cluster(request, test_rns_folder):
    kube_config_path = Path.home() / ".kube" / "config"

    if not kube_config_path.exists():
        pytest.skip("no kubeconfig found")
    cluster_name = (
        "k8s-cpu-den"
        if not request.config.getoption("--ci")
        else f"{test_rns_folder}-k8s-cpu-den"
    )
    args = {
        "name": cluster_name,
        "provider": "kubernetes",
        "instance_type": "CPU:1",
        "launcher": LauncherType.DEN,
        "context": os.getenv("EKS_ARN"),
    }

    cluster = setup_test_cluster(args, request)
    yield cluster
    teardown_cluster_fixture(request, cluster)


@pytest.fixture(scope="session")
def den_launched_ondemand_gcp_k8s_cluster(request, test_rns_folder):
    kube_config_path = Path.home() / ".kube" / "config"

    if not kube_config_path.exists():
        pytest.skip("no kubeconfig found")
    cluster_name = (
        "k8s-cpu-gke-den"
        if not request.config.getoption("--ci")
        else f"{test_rns_folder}-k8s-cpu-gke-den"
    )
    args = {
        "name": cluster_name,
        "provider": "kubernetes",
        "instance_type": "CPU:1",
        "launcher": LauncherType.DEN,
        "context": "gke_testing",
    }

    cluster = setup_test_cluster(args, request)
    yield cluster
    teardown_cluster_fixture(request, cluster)


@pytest.fixture(scope="session")
def ondemand_k8s_docker_cluster(request, test_rns_folder):
    kube_config_path = Path.home() / ".kube" / "config"

    if not kube_config_path.exists():
        pytest.skip("no kubeconfig found")

    cluster_name = (
        "k8s-docker-cpu"
        if not request.config.getoption("--ci")
        else f"{test_rns_folder}-k8s-docker-cpu"
    )
    args = {
        "name": cluster_name,
        "provider": "kubernetes",
        "instance_type": "CPU:1",
        "image": Image(name="default_image")
        .from_docker("rayproject/ray:latest-py311-cpu")
        .install_packages(TEST_REQS),
    }
    cluster = setup_test_cluster(args, request)
    yield cluster
    teardown_cluster_fixture(request, cluster)


@pytest.fixture(scope="session")
def v100_gpu_cluster(request):
    args = {
        "name": "rh-v100",
        "instance_type": "V100:1",
        "provider": "aws",
    }
    cluster = setup_test_cluster(args, request)
    yield cluster
    teardown_cluster_fixture(request, cluster)


@pytest.fixture(scope="session")
def den_launcher_v100_gpu_cluster(request):
    args = {
        "name": "rh-v100-den",
        "instance_type": "V100:1",
        "provider": "aws",
        "launcher": LauncherType.DEN,
    }
    cluster = setup_test_cluster(args, request)
    yield cluster
    teardown_cluster_fixture(request, cluster)


@pytest.fixture(scope="session")
def k80_gpu_cluster(request):
    args = {
        "name": "rh-k80",
        "instance_type": "K80:1",
        "provider": "aws",
    }
    cluster = setup_test_cluster(args, request)
    yield cluster
    teardown_cluster_fixture(request, cluster)


@pytest.fixture(scope="session")
def a10g_gpu_cluster(request):
    args = {
        "name": "rh-a10x",
        "instance_type": "g5.2xlarge",
        "provider": "aws",
    }
    cluster = setup_test_cluster(args, request)
    yield cluster
    teardown_cluster_fixture(request, cluster)


@pytest.fixture(scope="session")
def multinode_k8s_cpu_cluster(request, test_rns_folder):
    cluster_name = (
        "rh-cpu-multinode"
        if not request.config.getoption("--ci")
        else f"{test_rns_folder}-rh-cpu-multinode"
    )
    args = {
        "name": cluster_name,
        "num_nodes": NUM_OF_NODES,
        "provider": "kubernetes",
        "instance_type": "CPU:2+",
    }
    cluster = setup_test_cluster(args, request)
    yield cluster
    teardown_cluster_fixture(request, cluster)


@pytest.fixture(scope="session")
def multinode_cpu_docker_conda_cluster(request):
    image = (
        Image(name="default_image")
        .from_docker("rayproject/ray:latest-py311-cpu")
        .setup_conda_env(
            conda_env_name="base_env",
            conda_config={"dependencies": ["python=3.11"], "name": "base_env"},
        )
        .install_packages(TEST_REQS + ["ray==2.30.0"], conda_env_name="base_env")
    )
    args = {
        "name": "rh-cpu-multinode",
        "num_nodes": NUM_OF_NODES,
        "image": image,
        "provider": "aws",
        "instance_type": "CPU:2+",
    }
    cluster = setup_test_cluster(args, request)
    yield cluster
    teardown_cluster_fixture(request, cluster)


@pytest.fixture(scope="session")
def multinode_gpu_cluster(request):
    args = {
        "name": "rh-gpu-multinode",
        "num_nodes": NUM_OF_NODES,
        "instance_type": "g5.xlarge",
    }
    cluster = setup_test_cluster(args, request)
    yield cluster
    teardown_cluster_fixture(request, cluster)
