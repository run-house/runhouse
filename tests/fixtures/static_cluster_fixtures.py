import pytest

import runhouse as rh

from tests.conftest import init_args
from tests.utils import test_env


@pytest.fixture(scope="session")
def static_cpu_cluster():
    # TODO: Spin up a new basic m5.xlarge EC2 instance
    # import boto3

    # ec2 = boto3.resource("ec2")
    # instances = ec2.create_instances(
    #     ImageId="ami-0a313d6098716f372",
    #     InstanceType="m5.xlarge",
    #     MinCount=1,
    #     MaxCount=1,
    #     KeyName="sky-key",
    #     TagSpecifications=[
    #         {
    #             "ResourceType": "instance",
    #             "Tags": [
    #                 {"Key": "Name", "Value": "rh-cpu"},
    #             ],
    #         },
    #     ],
    # )
    # instance = instances[0]
    # instance.wait_until_running()
    # instance.load()

    # ip = instance.public_ip_address

    c = (
        rh.ondemand_cluster(
            instance_type="m5.xlarge",
            provider="aws",
            region="us-east-1",
            # image_id="ami-0a313d6098716f372",  # Upgraded to python 3.11.4 which is not compatible with ray 2.4.0
            name="test-byo-cluster",
        )
        .up_if_not()
        .save()
    )

    args = dict(
        name="different-cluster",
        host=c.address,
        ssh_creds={"ssh_user": "ubuntu", "ssh_private_key": "~/.ssh/sky-key"},
    )
    c = rh.cluster(**args).save()
    c.restart_server(resync_rh=True)  # needed to override the cluster's config file
    init_args[id(c)] = args

    test_env().to(c)
    c.sync_secrets(["ssh"])

    return c


@pytest.fixture(scope="session")
def password_cluster():
    sky_cluster = rh.cluster("temp-rh-password", instance_type="CPU:4").save()
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
    ssh_creds = {"ssh_user": "ubuntu", "password": "cluster-pass"}
    args = dict(name="rh-password", host=[sky_cluster.address], ssh_creds=ssh_creds)
    c = rh.cluster(**args).save()
    init_args[id(c)] = args

    return c
