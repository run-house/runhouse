import pkgutil
import shlex
import time
from pathlib import Path

from typing import Any, Dict, List

import pytest

import runhouse as rh
import yaml

from tests.conftest import init_args

SSH_USER = "rh-docker-user"
BASE_LOCAL_SSH_PORT = 32320
LOCAL_HTTPS_SERVER_PORT = 8443
LOCAL_HTTP_SERVER_PORT = 8080
DEFAULT_KEYPAIR_KEYPATH = "~/.ssh/sky-key"


def get_rh_parent_path():
    return Path(pkgutil.get_loader("runhouse").path).parent.parent


@pytest.fixture(scope="function")
def cluster(request):
    return request.getfixturevalue(request.param)


@pytest.fixture(scope="session")
def named_cluster():
    # Name cannot be the only arg or we attempt to load from name, and throw an error when it's not found
    args = dict(name="test-simple-cluster", host="my_url.com")
    c = rh.cluster(**args)
    init_args[id(c)] = args
    return c


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
    args = dict(name="rh-password", host=[sky_cluster.address], ssh_creds=ssh_creds)
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
        print(f"Container {container_name} already running, skipping build and run.")
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

    return client


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


def set_up_local_cluster(
    image_name: str,
    container_name: str,
    dir_name: str,
    detached: bool,
    force_rebuild: bool,
    port_fwds: List[str],
    local_ssh_port: int,
    additional_cluster_init_args: Dict[str, Any],
    keypath: str = None,
    pwd_file: str = None,
):
    docker_client = build_and_run_image(
        image_name=image_name,
        container_name=container_name,
        dir_name=dir_name,
        detached=detached,
        keypath=keypath,
        pwd_file=pwd_file,
        force_rebuild=force_rebuild,
        port_fwds=port_fwds,
    )

    cluster_init_args = dict(
        host="localhost",
        server_host="0.0.0.0",
        ssh_port=local_ssh_port,
        server_connection_type="ssh",
        ssh_creds={
            "ssh_user": SSH_USER,
            "ssh_private_key": keypath,
        },
    )

    for k, v in additional_cluster_init_args.items():
        cluster_init_args[k] = v

    rh_cluster = rh.cluster(**cluster_init_args)
    init_args[id(rh_cluster)] = cluster_init_args

    rh.env(
        reqs=["pytest", "httpx", "pytest_asyncio"],
        working_dir=None,
        setup_cmds=[
            f"mkdir -p ~/.rh; touch ~/.rh/config.yaml; "
            f"echo '{yaml.safe_dump(rh.configs.defaults_cache)}' > ~/.rh/config.yaml"
        ],
        name="base_env",
    ).to(rh_cluster)

    rh_cluster.save()

    def cleanup():
        if not detached:
            docker_client.containers.get(container_name).stop()
            docker_client.containers.prune()
            docker_client.images.prune()

    return rh_cluster, cleanup


@pytest.fixture(scope="session")
def local_docker_cluster_public_key(request):
    local_ssh_port = BASE_LOCAL_SSH_PORT + 1
    local_cluster, cleanup = set_up_local_cluster(
        image_name="keypair",
        container_name="rh-slim-public-key",
        dir_name="public-key-auth",
        keypath=str(
            Path(
                rh.configs.get("default_keypair", DEFAULT_KEYPAIR_KEYPATH)
            ).expanduser()
        ),
        detached=request.config.getoption("--detached"),
        force_rebuild=request.config.getoption("--force-rebuild"),
        port_fwds=[f"{local_ssh_port}:22"],
        local_ssh_port=local_ssh_port,
        additional_cluster_init_args={
            "name": "local_docker_cluster_public_key",
            "den_auth": "den_auth" in request.keywords,
        },
    )
    # Yield the cluster
    yield local_cluster

    # Stop the Docker container
    cleanup()


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


@pytest.fixture(scope="function")
def local_docker_cluster_public_key_den_auth(local_docker_cluster_public_key):
    local_docker_cluster_public_key.run(
        [
            f"mkdir -p ~/.rh; touch ~/.rh/config.yaml; "
            f'echo "{yaml.safe_dump(rh.configs.defaults_cache)}" > ~/.rh/config.yaml'
        ],
    )
    local_docker_cluster_public_key.den_auth = True
    local_docker_cluster_public_key.restart_server(resync_rh=False)
    return local_docker_cluster_public_key


@pytest.fixture(scope="session")
def local_docker_cluster_telemetry_public_key(request, detached=True):
    local_ssh_port = BASE_LOCAL_SSH_PORT + 2
    local_cluster, cleanup = set_up_local_cluster(
        image_name="keypair-telemetry",
        container_name="rh-slim-keypair-telemetry",
        dir_name="public-key-auth",
        keypath=str(
            Path(
                rh.configs.get("default_keypair", DEFAULT_KEYPAIR_KEYPATH)
            ).expanduser()
        ),
        detached=request.config.getoption("--detached"),
        force_rebuild=request.config.getoption("--force-rebuild"),
        port_fwds=[f"{local_ssh_port}:22"],
        local_ssh_port=local_ssh_port,
        additional_cluster_init_args={
            "name": "local_docker_cluster_telemetry_public_key",
            "use_local_telemetry": True,
            "den_auth": "den_auth" in request.keywords,
        },
    )
    # Yield the cluster
    yield local_cluster

    # Stop the Docker container
    cleanup()


@pytest.fixture(scope="session")
def local_docker_cluster_with_nginx_http(request):
    local_ssh_port = BASE_LOCAL_SSH_PORT + 3
    client_port = LOCAL_HTTP_SERVER_PORT + 3
    port_fwds = [f"{local_ssh_port}:22", f"{client_port}:80"]

    local_cluster, cleanup = set_up_local_cluster(
        image_name="keypair",
        container_name="rh-slim-http-nginx",
        dir_name="public-key-auth",
        keypath=str(
            Path(
                rh.configs.get("default_keypair", DEFAULT_KEYPAIR_KEYPATH)
            ).expanduser()
        ),
        detached=request.config.getoption("--detached"),
        force_rebuild=request.config.getoption("--force-rebuild"),
        port_fwds=port_fwds,
        local_ssh_port=local_ssh_port,
        additional_cluster_init_args={
            "name": "local_docker_cluster_with_nginx",
            "server_connection_type": "none",
            "server_port": 80,
            "client_port": client_port,
            "den_auth": True,
        },
    )
    # Yield the cluster
    yield local_cluster

    # Stop the Docker container
    cleanup()


@pytest.fixture(scope="session")
def local_docker_cluster_with_nginx_https(request):
    local_ssh_port = BASE_LOCAL_SSH_PORT + 4
    client_port = LOCAL_HTTPS_SERVER_PORT + 3
    port_fwds = [f"{local_ssh_port}:22", f"{client_port}:443"]

    local_cluster, cleanup = set_up_local_cluster(
        image_name="keypair",
        container_name="rh-slim-https-nginx",
        dir_name="public-key-auth",
        keypath=str(
            Path(
                rh.configs.get("default_keypair", DEFAULT_KEYPAIR_KEYPATH)
            ).expanduser()
        ),
        detached=request.config.getoption("--detached"),
        force_rebuild=request.config.getoption("--force-rebuild"),
        port_fwds=port_fwds,
        local_ssh_port=local_ssh_port,
        additional_cluster_init_args={
            "name": "local_docker_cluster_with_nginx",
            "server_connection_type": "tls",
            "server_port": 443,
            "client_port": client_port,
            "den_auth": True,
        },
    )
    # Yield the cluster
    yield local_cluster

    # Stop the Docker container
    cleanup()


@pytest.fixture(scope="function")
def local_test_account_cluster_public_key(request, test_account):
    """
    This fixture is not parameterized for every test; it is a separate cluster started with a test account
    (username: kitchen_tester) in order to test sharing resources with other users.
    """
    with test_account:

        local_ssh_port = BASE_LOCAL_SSH_PORT + 5
        local_cluster, cleanup = set_up_local_cluster(
            image_name="keypair",
            container_name="rh-slim-test-acct",
            dir_name="public-key-auth",
            keypath=str(
                Path(
                    rh.configs.get("default_keypair", DEFAULT_KEYPAIR_KEYPATH)
                ).expanduser()
            ),
            detached=request.config.getoption("--detached"),
            force_rebuild=request.config.getoption("--force-rebuild"),
            port_fwds=[f"{local_ssh_port}:22"],
            local_ssh_port=local_ssh_port,
            additional_cluster_init_args={
                "name": "local_test_account_cluster_public_key",
                "den_auth": "den_auth" in request.keywords,
            },
        )

    yield local_cluster

    cleanup()


@pytest.fixture(scope="session")
def shared_cluster(test_account, local_test_account_cluster_public_key):
    username_to_share = rh.configs.get("username")
    with test_account:
        # Share the cluster with the test account
        local_test_account_cluster_public_key.share(
            username_to_share, access_level="read"
        )

    return local_test_account_cluster_public_key


@pytest.fixture(scope="session")
def local_docker_cluster_passwd(request):
    local_ssh_port = BASE_LOCAL_SSH_PORT + 6
    pwd_file = "docker_user_passwd"
    rh_parent_path = get_rh_parent_path()
    pwd = (rh_parent_path.parent / pwd_file).read_text().strip()

    local_cluster, cleanup = set_up_local_cluster(
        image_name="pwd",
        container_name="rh-slim-password",
        dir_name="password-file-auth",
        pwd_file="docker_user_passwd",
        detached=request.config.getoption("--detached"),
        force_rebuild=request.config.getoption("--force-rebuild"),
        port_fwds=[f"{local_ssh_port}:22"],
        local_ssh_port=local_ssh_port,
        additional_cluster_init_args={
            "name": "local_docker_cluster_passwd",
            "den_auth": "den_auth" in request.keywords,
            "ssh_creds": {"ssh_user": SSH_USER, "password": pwd},
        },
    )
    # Yield the cluster
    yield local_cluster

    # Stop the Docker container
    cleanup()
