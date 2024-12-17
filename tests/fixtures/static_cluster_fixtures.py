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
    launcher: Union[LauncherType, str] = None,
    compute_type: computeType = computeType.cpu,
):
    from tests.fixtures.resource_fixtures import create_folder_path

    rh.constants.SSH_SKY_SECRET_NAME = (
        f"{create_folder_path()}-{rh.constants.SSH_SKY_SECRET_NAME}"
    )
    instance_type = "CPU:4" if compute_type == computeType.cpu else "g5.xlarge"
    cluster = rh.cluster(
        f"aws-{compute_type}-password",
        instance_type=instance_type,
        provider="aws",
        launcher=launcher,
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
    args = dict(name="static-cpu-password", host=[cluster.head_ip], ssh_creds=ssh_creds)
    c = rh.cluster(**args).save()
    c.restart_server(resync_rh=True)
    init_args[id(c)] = args

    setup_test_base(c)

    return c


@pytest.fixture(scope="session")
def static_cpu_pwd_cluster():
    return setup_static_cluster()


@pytest.fixture(scope="session")
def static_cpu_pwd_cluster_den_launcher():
    return setup_static_cluster(launcher=LauncherType.DEN)


@pytest.fixture(scope="session")
def static_gpu_pwd_cluster_den_launcher():
    return setup_static_cluster(launcher=LauncherType.DEN, compute_type=computeType.gpu)
