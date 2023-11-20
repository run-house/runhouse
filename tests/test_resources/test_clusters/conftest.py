import pkgutil
import shlex
import time
from pathlib import Path

import pytest

import runhouse as rh
import yaml

from tests.conftest import init_args

SSH_USER = "rh-docker-user"
BASE_LOCAL_SSH_PORT = 32320
DEFAULT_KEYPAIR_KEYPATH = "~/.ssh/sky-key"


@pytest.fixture(scope="function")
def cluster(request):
    return request.getfixturevalue(request.param.__name__)


@pytest.fixture(scope="session")
def named_cluster():
    args = {"name": "test-simple-cluster"}
    c = rh.cluster(**args)
    init_args[id(c)] = args
    return c


@pytest.fixture(scope="session")
def static_cpu_cluster():
    # Spin up a new basic m5.xlarge EC2 instance
    import boto3

    ec2 = boto3.resource("ec2")
    instances = ec2.create_instances(
        ImageId="ami-0a313d6098716f372",
        InstanceType="m5.xlarge",
        MinCount=1,
        MaxCount=1,
        KeyName="sky-key",
        TagSpecifications=[
            {
                "ResourceType": "instance",
                "Tags": [
                    {"Key": "Name", "Value": "rh-cpu"},
                ],
            },
        ],
    )
    instance = instances[0]
    instance.wait_until_running()
    instance.load()

    ip = instance.public_ip_address

    # c = (
    #     rh.ondemand_cluster(
    #         instance_type="m5.xlarge",
    #         provider="aws",
    #         region="us-east-1",
    #         # image_id="ami-0a313d6098716f372",  # Upgraded to python 3.11.4 which is not compatible with ray 2.4.0
    #         name="test-byo-cluster",
    #     )
    #     .up_if_not()
    #     .save()
    # )

    args = dict(
        name="different-cluster",
        ips=ip,
        ssh_creds={"username": "ubuntu", "ssh_private_key": "~/.ssh/sky-key"},
    )
    c = rh.cluster(**args).save()
    init_args[id(c)] = args

    c.install_packages(["pytest"])
    c.sync_secrets(["ssh"])

    return c


# Original
@pytest.fixture(scope="session")
def byo_cpu():
    # Spin up a new basic m5.xlarge EC2 instance
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

    args = dict(name="different-cluster", ips=[c.address], ssh_creds=c.ssh_creds())
    c = rh.cluster(**args).save()
    init_args[id(c)] = args

    c.install_packages(["pytest"])
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
    args = dict(name="rh-password", ips=[sky_cluster.address], ssh_creds=ssh_creds)
    c = rh.cluster(**args).save()
    init_args[id(c)] = args

    return c


########### Docker Clusters ###########


def build_and_run_image(
    image_name: str,
    container_name: str,
    detached: bool,
    dir_name: str,
    pwd_file=None,
    keypath=None,
    force_rebuild=False,
    port_fwds=["22:22"],
):
    import subprocess

    import docker

    local_rh_package_path = Path(pkgutil.get_loader("runhouse").path).parent
    dockerfile_path = local_rh_package_path / f"docker/slim/{dir_name}/Dockerfile"
    rh_parent_path = local_rh_package_path.parent
    rh_path = "runhouse" if (rh_parent_path / "setup.py").exists() else None
    rh_version = rh.__version__ if not rh_path else None

    client = docker.from_env()
    # Check if the container is already running, and if so, skip build and run
    containers = client.containers.list(
        all=True,
        filters={
            "ancestor": f"runhouse:{image_name}",
            "status": "running",
            "name": container_name,
        },
    )
    if len(containers) > 0 and detached:
        print(f"Container {container_name} already running, skipping build and run")
    else:
        # Check if image has already been built before re-building
        images = client.images.list(filters={"reference": f"runhouse:{image_name}"})
        if not images or force_rebuild:
            # Build the SSH public key based Docker image
            if keypath:
                build_cmd = [
                    "docker",
                    "build",
                    "--pull",
                    "--rm",
                    "-f",
                    str(dockerfile_path),
                    "--build-arg",
                    f"RUNHOUSE_PATH={rh_path}"
                    if rh_path
                    else f"RUNHOUSE_VERSION={rh_version}",
                    "--secret",
                    f"id=ssh_key,src={keypath}.pub",
                    "-t",
                    f"runhouse:{image_name}",
                    ".",
                ]
            elif pwd_file:
                # Build a password file based Docker image
                build_cmd = [
                    "docker",
                    "build",
                    "--pull",
                    "--rm",
                    "-f",
                    str(dockerfile_path),
                    "--build-arg",
                    f"DOCKER_USER_PASSWORD_FILE={pwd_file}",
                    "--build-arg",
                    f"RUNHOUSE_PATH={rh_path}"
                    if rh_path
                    else f"RUNHOUSE_VERSION={rh_version}",
                    "-t",
                    f"runhouse:{image_name}",
                    ".",
                ]
            else:
                raise ValueError("No keypath or password file path provided")

            print(shlex.join(build_cmd))
            run_shell_command(subprocess, build_cmd, cwd=str(rh_parent_path.parent))

        # Run the Docker image
        port_fwds = (
            "".join([f"-p {port_fwd} " for port_fwd in port_fwds]).strip().split(" ")
        )
        run_cmd = (
            ["docker", "run", "--name", container_name, "-d", "--rm", "--shm-size=4gb"]
            + port_fwds
            + [f"runhouse:{image_name}"]
        )
        print(shlex.join(run_cmd))
        res = popen_shell_command(subprocess, run_cmd, cwd=str(rh_parent_path.parent))
        stdout, stderr = res.communicate()
        if res.returncode != 0:
            raise RuntimeError(f"Failed to run docker image {image_name}: {stderr}")

    return client, rh_parent_path


def run_shell_command_direct(subprocess, cmd: str):
    # Run the command and wait for it to complete
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    if result.returncode != 0:
        print("subprocess failed, stdout: " + result.stdout)
        print("subprocess failed, stderr: " + result.stderr)

    # Check for success
    assert result.returncode == 0


def run_shell_command(subprocess, cmd: list[str], cwd: str = None):
    # Run the command and wait for it to complete
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd or Path.cwd())

    if result.returncode != 0:
        print("subprocess failed, stdout: " + result.stdout)
        print("subprocess failed, stderr: " + result.stderr)

    # Check for success
    assert result.returncode == 0


def popen_shell_command(subprocess, command: list[str], cwd: str = None):
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=cwd or Path.cwd(),
    )
    # Wait for 10 seconds before resuming execution
    time.sleep(10)
    return process


@pytest.fixture(scope="session")
def local_docker_cluster_public_key(request):
    image_name = "keypair"
    container_name = "rh-slim-public-key"
    dir_name = "public-key-auth"
    detached = request.config.getoption("--detached")
    keypath = str(
        Path(rh.configs.get("default_keypair", DEFAULT_KEYPAIR_KEYPATH)).expanduser()
    )
    local_ssh_port = BASE_LOCAL_SSH_PORT + 1

    client, rh_parent_path = build_and_run_image(
        image_name=image_name,
        container_name=container_name,
        dir_name=dir_name,
        detached=detached,
        keypath=keypath,
        force_rebuild=request.config.getoption("--force-rebuild"),
        port_fwds=[f"{local_ssh_port}:22"],
    )

    # Runhouse commands can now be run locally
    args = dict(
        name="local_docker_cluster_public_key",
        host="localhost",
        server_host="0.0.0.0",
        ssh_port=local_ssh_port,
        server_connection_type="ssh",
        ssh_creds={
            "ssh_user": SSH_USER,
            "ssh_private_key": keypath,
        },
    )
    c = rh.cluster(**args)
    init_args[id(c)] = args

    rh.env(
        reqs=["pytest", "httpx", "pytest_asyncio"],
        working_dir=None,
        name="base_env",
    ).to(c)
    c.save()

    # Yield the cluster
    yield c

    # Stop the Docker container
    if not detached:
        client.containers.get(container_name).stop()
        client.containers.prune()
        client.images.prune()


# These two clusters cannot be used in the same test together, they are are technically
# the same image, but we switch the log in.


@pytest.fixture(scope="function")
def local_docker_cluster_public_key_logged_out(local_docker_cluster_public_key):
    local_docker_cluster_public_key.run(
        ["rm ~/.rh/config.yaml"],
    )
    local_docker_cluster_public_key.restart_server(resync_rh=False)
    return local_docker_cluster_public_key


@pytest.fixture(scope="function")
def local_docker_cluster_public_key_logged_in(local_docker_cluster_public_key):
    local_docker_cluster_public_key.run(
        [
            f"mkdir -p ~/.rh; touch ~/.rh/config.yaml; "
            f'echo "{yaml.safe_dump(rh.configs.defaults_cache)}" > ~/.rh/config.yaml'
        ],
    )
    local_docker_cluster_public_key.restart_server(resync_rh=False)
    return local_docker_cluster_public_key


@pytest.fixture(scope="session")
def local_docker_cluster_telemetry_public_key(request, detached=True):
    image_name = "keypair-telemetry"
    container_name = "rh-slim-keypair-telemetry"
    dir_name = "public-key-auth"
    detached = request.config.getoption("--detached")
    keypath = str(
        Path(rh.configs.get("default_keypair", DEFAULT_KEYPAIR_KEYPATH)).expanduser()
    )
    local_ssh_port = BASE_LOCAL_SSH_PORT + 3

    client, rh_parent_path = build_and_run_image(
        image_name=image_name,
        container_name=container_name,
        dir_name=dir_name,
        detached=detached,
        keypath=keypath,
        force_rebuild=request.config.getoption("--force-rebuild"),
        port_fwds=[f"{local_ssh_port}:22"],
    )

    # Runhouse commands can now be run locally
    args = dict(
        name="local-docker-slim-keypair-telemetry",
        host="localhost",
        server_host="0.0.0.0",
        ssh_port=local_ssh_port,
        server_connection_type="ssh",
        ssh_creds={
            "ssh_user": SSH_USER,
            "ssh_private_key": keypath,
        },
        use_local_telemetry=True,
    )
    c = rh.cluster(**args)
    init_args[id(c)] = args
    rh.env(
        reqs=["pytest"],
        working_dir=None,
        setup_cmds=[
            f'mkdir -p ~/.rh; echo "token: {rh.configs.get("token")}" > ~/.rh/config.yaml'
        ],
        name="base_env",
    ).to(c)
    c.save()

    # Yield the cluster
    yield c

    # Stop the Docker container
    if not detached:
        client.containers.get(container_name).stop()
        client.containers.prune()
        client.images.prune()


@pytest.fixture(scope="function")
def local_test_account_cluster_public_key(request, test_account, detached=True):
    """
    This fixture is not parameterized for every test; it is a separate cluster started with a test account
    (username: kitchen_tester) in order to test sharing resources with other users.
    """
    with test_account as test_account_dict:
        image_name = "keypair"
        container_name = "rh-slim-test-acct"
        dir_name = "public-key-auth"
        detached = request.config.getoption("--detached")
        den_auth = "den_auth" in request.keywords
        keypath = str(
            Path(
                rh.configs.get("default_keypair", DEFAULT_KEYPAIR_KEYPATH)
            ).expanduser()
        )
        local_ssh_port = BASE_LOCAL_SSH_PORT + 4

        client, rh_parent_path = build_and_run_image(
            image_name=image_name,
            container_name=container_name,
            detached=detached,
            dir_name=dir_name,
            keypath=keypath,
            force_rebuild=request.config.getoption("--force-rebuild"),
            port_fwds=[f"{local_ssh_port}:22"],
        )

        args = dict(
            name="local-docker-slim-test-acct",
            host="localhost",
            den_auth=den_auth,
            server_host="0.0.0.0",
            ssh_port=local_ssh_port,
            server_connection_type="ssh",
            ssh_creds={
                "ssh_user": SSH_USER,
                "ssh_private_key": keypath,
            },
        )
        c = rh.cluster(**args)
        init_args[id(c)] = args

        rh.env(
            reqs=["pytest"],
            working_dir=None,
            setup_cmds=[
                f"mkdir -p ~/.rh; touch ~/.rh/config.yaml; "
                f'echo "{yaml.safe_dump(test_account_dict)}" > ~/.rh/config.yaml'
            ],
            name="base_env",
        ).to(c)
        c.save()

    # Yield the cluster
    yield c

    # Stop the Docker container
    if not detached:
        client.containers.get(container_name).stop()
        client.containers.prune()
        client.images.prune()


@pytest.fixture(scope="session")
def shared_cluster(test_account, local_test_account_cluster_public_key):
    username_to_share = rh.configs.get("username")
    with test_account:
        # Share the cluster with the test account
        local_test_account_cluster_public_key.share(
            username_to_share, access_type="read"
        )

    return local_test_account_cluster_public_key


@pytest.fixture(scope="session")
def local_docker_cluster_passwd(request, detached=True):
    image_name = "pwd"
    container_name = "rh-slim-password"
    dir_name = "password-file-auth"
    detached = request.config.getoption("--detached")
    pwd_file = "docker_user_passwd"
    local_ssh_port = BASE_LOCAL_SSH_PORT + 5

    client, rh_parent_path = build_and_run_image(
        image_name=image_name,
        container_name=container_name,
        dir_name=dir_name,
        detached=detached,
        pwd_file=pwd_file,
        force_rebuild=request.config.getoption("--force-rebuild"),
        port_fwds=[f"{local_ssh_port}:22"],
    )

    # Runhouse commands can now be run locally
    pwd = (rh_parent_path.parent / pwd_file).read_text().strip()
    args = dict(
        name="local-docker-slim-password",
        host="localhost",
        server_host="0.0.0.0",
        ssh_port=local_ssh_port,
        server_connection_type="ssh",
        ssh_creds={"ssh_user": SSH_USER, "password": pwd},
    )
    c = rh.cluster(**args)
    init_args[id(c)] = args
    rh.env(
        reqs=["pytest", "httpx", "pytest_asyncio"],
        working_dir=None,
        setup_cmds=[
            f'mkdir -p ~/.rh; echo "token: {rh.configs.get("token")}" > ~/.rh/config.yaml'
        ],
        name="base_env",
    ).to(c)
    c.save()

    # Yield the cluster
    yield c

    # Stop the Docker container
    if not detached:
        client.containers.get(container_name).stop()
        client.containers.prune()
        client.images.prune()
