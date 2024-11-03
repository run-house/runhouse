import pytest

import runhouse as rh

from tests.conftest import init_args
from tests.utils import test_env


@pytest.fixture(scope="session")
def static_cpu_pwd_cluster():
    sky_cluster = rh.cluster(
        "aws-cpu-password", instance_type="CPU:4", provider="aws"
    ).save()
    if not sky_cluster.is_up():
        sky_cluster.up()

        # set up password on remote
        sky_cluster.run(
            [
                [
                    'sudo sed -i "/^[^#]*PasswordAuthentication[[:space:]]no/c\PasswordAuthentication yes" '
                    "/etc/ssh/sshd_config"
                ]
            ]
        )
        sky_cluster.run(["sudo /etc/init.d/ssh force-reload"])
        sky_cluster.run(["sudo /etc/init.d/ssh restart"])
        sky_cluster.run(
            ["(echo 'cluster-pass' && echo 'cluster-pass') | sudo passwd ubuntu"]
        )
        sky_cluster.run(["pip uninstall skypilot runhouse -y", "pip install pytest"])
        sky_cluster.run(["rm -rf runhouse/"])

    # instantiate byo cluster with password
    ssh_creds = {
        "ssh_user": "ubuntu",
        "ssh_private_key": "~/.ssh/sky-key",
        "password": "cluster-pass",
    }
    args = dict(
        name="static-cpu-password", host=[sky_cluster.head_ip], ssh_creds=ssh_creds
    )
    c = rh.cluster(**args).save()
    c.restart_server(resync_rh=True)
    init_args[id(c)] = args

    test_env().to(c)

    return c
