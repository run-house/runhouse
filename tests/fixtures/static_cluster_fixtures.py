from enum import Enum
from typing import Union

import pytest

import runhouse as rh
from runhouse.resources.hardware.utils import LauncherType

from tests.conftest import init_args
from tests.utils import test_env


class computeType(str, Enum):
    cpu = "cpu"
    gpu = "gpu"


def setup_static_cluster(
    launcher_type: Union[LauncherType, str] = None,
    compute_type: computeType = computeType.cpu,
):
    instance_type = "CPU:4" if compute_type == computeType.cpu else "g5.xlarge"
    cluster = rh.cluster(
        f"aws-{compute_type}-password",
        instance_type=instance_type,
        provider="aws",
        launcher_type=launcher_type,
    ).save()
    if not cluster.is_up():
        cluster.up()

        # set up password on remote
        cluster.run(
            [
                [
                    'sudo sed -i "/^[^#]*PasswordAuthentication[[:space:]]no/c\PasswordAuthentication yes" '
                    "/etc/ssh/sshd_config"
                ]
            ]
        )
        cluster.run(["sudo /etc/init.d/ssh force-reload"])
        cluster.run(["sudo /etc/init.d/ssh restart"])
        cluster.run(
            ["(echo 'cluster-pass' && echo 'cluster-pass') | sudo passwd ubuntu"]
        )
        cluster.run(["pip uninstall skypilot runhouse -y", "pip install pytest"])
        cluster.run(["rm -rf runhouse/"])

    # instantiate byo cluster with password
    ssh_creds = {
        "ssh_user": "ubuntu",
        "ssh_private_key": "~/.ssh/sky-key",
        "password": "cluster-pass",
    }
    args = dict(name="static-cpu-password", host=[cluster.head_ip], ssh_creds=ssh_creds)
    c = rh.cluster(**args).save()
    c.restart_server(resync_rh=True)
    init_args[id(c)] = args

    test_env().to(c)

    return c


@pytest.fixture(scope="session")
def static_cpu_pwd_cluster():
    return setup_static_cluster()


@pytest.fixture(scope="session")
def static_cpu_pwd_cluster_den_launcher():
    return setup_static_cluster(launcher_type=LauncherType.DEN)


@pytest.fixture(scope="session")
def static_gpu_pwd_cluster_den_launcher():
    return setup_static_cluster(
        launcher_type=LauncherType.DEN, compute_type=computeType.gpu
    )
