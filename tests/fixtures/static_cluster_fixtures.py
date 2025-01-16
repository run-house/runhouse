from enum import Enum
from typing import Union

import pytest

import runhouse as rh
from runhouse.resources.hardware.utils import LauncherType

from tests.conftest import init_args
from tests.utils import setup_test_base


class computeType(str, Enum):
    cpu = "cpu"
    gpu = "gpu"


def setup_static_cluster(
    test_rns_folder: str,
    launcher: Union[LauncherType, str] = None,
    compute_type: computeType = computeType.cpu,
):
    rh.constants.SSH_SKY_SECRET_NAME = (
        f"{test_rns_folder}-{rh.constants.SSH_SKY_SECRET_NAME}"
    )
    instance_type = "CPU:4" if compute_type == computeType.cpu else "g5.xlarge"
    launcher = launcher if launcher else LauncherType.LOCAL
    cluster_name = f"{test_rns_folder}-{launcher}-aws-{compute_type}-password"
    cluster = rh.cluster(
        name=cluster_name,
        instance_type=instance_type,
        provider="aws",
        launcher=launcher,
        den_auth=True,
    ).save()
    if not cluster.is_up():
        cluster.up()

        # set up password on remote
        cluster.run_bash(
            [
                'sudo sed -i "/^[^#]*PasswordAuthentication[[:space:]]no/c\PasswordAuthentication yes" /etc/ssh/sshd_config'
            ]
        )
        cluster.run_bash(["sudo /etc/init.d/ssh force-reload"])
        cluster.run_bash(["sudo /etc/init.d/ssh restart"])
        cluster.run_bash(
            ["(echo 'cluster-pass' && echo 'cluster-pass') | sudo passwd ubuntu"]
        )
        cluster.run_bash(["pip uninstall skypilot runhouse -y", "pip install pytest"])
        cluster.run_bash(["rm -rf runhouse/"])

    # instantiate byo cluster with password
    ssh_creds = {
        "ssh_user": "ubuntu",
        "ssh_private_key": "~/.ssh/sky-key",
        "password": "cluster-pass",
    }
    # added static to the cluster name so the "under the hood" on demand cluster will not get overridden in den.
    # This way, when we'll call runhouse cluster down at the end of the nightly release test, both the on-demand cluster
    # and the static cluster will be terminated. Otherwise, both of the clusters will be terminated using the on-demand
    # cluster auto-stop.
    args = dict(
        name=f"{cluster_name}-static",
        host=[cluster.head_ip],
        ssh_creds=ssh_creds,
        den_auth=True,
    )
    c = rh.cluster(**args).save()
    c.restart_server(resync_rh=True)
    init_args[id(c)] = args

    setup_test_base(c)

    return c


@pytest.fixture(scope="session")
def static_cpu_pwd_cluster(request, test_rns_folder):
    cluster = setup_static_cluster(test_rns_folder=test_rns_folder)
    yield cluster
    if not request.config.getoption("--detached"):
        # for static cluster fixtures, we first create an ondemand cluster, to mock to user's BYO cluster, spinning it
        # up and then re-save it in den as a static cluster. We save it with the "static" postfix, so we could bring
        # down the on-demand cluster using runhouse. The static cluster's status will be updated shortly after the
        # on demand cluster teardown, by the cluster refresh cronjob, which runs every 3 minutes.
        on_demand_cluster_name = cluster.name.replace("-static", "")
        rh.cluster(name=on_demand_cluster_name).teardown()


@pytest.fixture(scope="session")
def static_cpu_pwd_cluster_den_launcher(test_rns_folder):
    cluster = setup_static_cluster(
        launcher=LauncherType.DEN, test_rns_folder=test_rns_folder
    )
    yield cluster
    # for static cluster fixtures, we first create an ondemand cluster, to mock to user's BYO cluster, spinning it
    # up and then re-save it in den as a static cluster. We save it with the "static" postfix, so we could bring
    # down the on-demand cluster using runhouse. The static cluster's status will be updated shortly after the
    # on demand cluster teardown, by the cluster refresh cronjob, which runs every 3 minutes.
    on_demand_cluster_name = cluster.name.replace("-static", "")
    rh.cluster(name=on_demand_cluster_name).teardown()


@pytest.fixture(scope="session")
def static_gpu_pwd_cluster_den_launcher(test_rns_folder):
    cluster = setup_static_cluster(
        launcher=LauncherType.DEN,
        compute_type=computeType.gpu,
        test_rns_folder=test_rns_folder,
    )
    yield cluster
    # for static cluster fixtures, we first create an ondemand cluster, to mock to user's BYO cluster, spinning it
    # up and then re-save it in den as a static cluster. We save it with the "static" postfix, so we could bring
    # down the on-demand cluster using runhouse. The static cluster's status will be updated shortly after the
    # on demand cluster teardown, by the cluster refresh cronjob, which runs every 3 minutes.
    on_demand_cluster_name = cluster.name.replace("-static", "")
    rh.cluster(name=on_demand_cluster_name).teardown()
