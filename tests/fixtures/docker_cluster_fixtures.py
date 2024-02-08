import importlib
import logging
import shlex
import subprocess
import time
from pathlib import Path

from typing import Any, Dict, List

import pytest

import runhouse as rh
import yaml

from runhouse.constants import DEFAULT_HTTP_PORT, DEFAULT_HTTPS_PORT, DEFAULT_SSH_PORT

from tests.conftest import init_args
from tests.utils import friend_account, test_env

SSH_USER = "rh-docker-user"
BASE_LOCAL_SSH_PORT = 32320
LOCAL_HTTPS_SERVER_PORT = 8443
LOCAL_HTTP_SERVER_PORT = 8080
DEFAULT_KEYPAIR_KEYPATH = "~/.ssh/sky-key"


def get_rh_parent_path():
    return Path(importlib.util.find_spec("runhouse").origin).parent.parent


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
def local_daemon(request):
    if not request.config.getoption("--detached") or rh.here == "file":
        logging.info("Starting local_daemon.")
        local_rh_package_path = Path(importlib.util.find_spec("runhouse").origin).parent
        subprocess.run(
            "runhouse restart",
            shell=True,  # Needed because we need to be in the right conda env
            cwd=local_rh_package_path,
            capture_output=True,
            text=True,
            check=True,
        )

    try:
        # Make sure the object store is set up correctly
        assert rh.here.on_this_cluster()
        yield rh.here

    finally:
        if not request.config.getoption("--detached"):
            subprocess.run("runhouse stop", capture_output=True, text=True, shell=True)


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

    args = dict(name="different-cluster", ips=[c.address], ssh_creds=c.ssh_creds)
    c = rh.cluster(**args).save()
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


########### Docker Clusters ###########


def build_and_run_image(
    image_name: str,
    container_name: str,
    reuse_existing_container: bool,
    dir_name: str,
    pwd_file=None,
    keypath=None,
    force_rebuild=False,
    port_fwds=[f"{DEFAULT_SSH_PORT}:{DEFAULT_SSH_PORT}"],
):
    import subprocess

    import docker

    rh_parent_path = Path(importlib.util.find_spec("runhouse").origin).parent.parent
    dockerfile_path = rh_parent_path / f"docker/slim/{dir_name}/Dockerfile"
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
    if len(containers) > 0:
        if not reuse_existing_container:
            raise ValueError(
                f"Container {container_name} already running, but reuse_existing_container=False"
            )
        else:
            logging.info(
                f"Container {container_name} already running, skipping build and run."
            )
            return client

    # The container is not running, so we need to build and run it
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
    reuse_existing_container: bool,
    force_rebuild: bool,
    port_fwds: List[str],
    local_ssh_port: int,
    additional_cluster_init_args: Dict[str, Any],
    logged_in: bool = False,
    keypath: str = None,
    pwd_file: str = None,
):
    docker_client = build_and_run_image(
        image_name=image_name,
        container_name=container_name,
        dir_name=dir_name,
        reuse_existing_container=reuse_existing_container,
        keypath=keypath,
        pwd_file=pwd_file,
        force_rebuild=force_rebuild,
        port_fwds=port_fwds,
    )

    cluster_init_args = dict(
        host="localhost",
        server_host="0.0.0.0",
        ssh_port=local_ssh_port,
        ssh_creds={
            "ssh_user": SSH_USER,
            "ssh_private_key": keypath,
        },
    )

    for k, v in additional_cluster_init_args.items():
        cluster_init_args[k] = v

    rh_cluster = rh.cluster(**cluster_init_args)
    init_args[id(rh_cluster)] = cluster_init_args

    # Save before bringing up so the correct rns_address is written to the cluster config.
    # This is necessary because we're turning on Den auth without saving the config.yaml (containing
    # the "owner's" token) to the container in many cases, so we're relying on authenticating the caller
    # to the server through Den. If the cluster isn't saved before coming up, the config in the cluster servlet
    # doesn't have the rns address, and the auth verification to Den fails.

    # Can't use the defaults_cache alone because we may need the token or username from the env variables
    config = rh.configs.defaults_cache
    config["token"] = rh.configs.token
    config["username"] = rh.configs.username

    rh.env(
        reqs=["pytest", "httpx", "pytest_asyncio", "pandas"],
        working_dir=None,
        setup_cmds=[
            f"mkdir -p ~/.rh; touch ~/.rh/config.yaml; "
            f"echo '{yaml.safe_dump(config)}' > ~/.rh/config.yaml"
        ]
        if logged_in
        else False,
        name="base_env",
    ).to(rh_cluster)

    rh_cluster.save()

    def cleanup():
        docker_client.containers.get(container_name).stop()
        docker_client.containers.prune()
        docker_client.images.prune()

    return rh_cluster, cleanup


@pytest.fixture(scope="session")
def docker_cluster_pk_tls_exposed(request):
    """This basic cluster fixture is set up with:
    - Public key authentication
    - Nginx set up on startup to forward Runhouse HTTP server to port 443
    """

    # From pytest config
    detached = request.config.getoption("--detached")
    force_rebuild = request.config.getoption("--force-rebuild")

    # Ports to use on the Docker VM such that they don't conflict
    local_ssh_port = BASE_LOCAL_SSH_PORT + 1
    local_client_port = LOCAL_HTTPS_SERVER_PORT + 1

    local_cluster, cleanup = set_up_local_cluster(
        image_name="keypair",
        container_name="rh-pk-tls-port-fwd",
        dir_name="public-key-auth",
        keypath=str(
            Path(
                rh.configs.get("default_keypair", DEFAULT_KEYPAIR_KEYPATH)
            ).expanduser()
        ),
        reuse_existing_container=detached,
        force_rebuild=force_rebuild,
        port_fwds=[
            f"{local_ssh_port}:{DEFAULT_SSH_PORT}",
            f"{local_client_port}:{DEFAULT_HTTPS_PORT}",
        ],
        local_ssh_port=local_ssh_port,
        additional_cluster_init_args={
            "name": "docker_cluster_pk_tls_exposed",
            "server_connection_type": "tls",
            "server_port": DEFAULT_HTTPS_PORT,
            "client_port": local_client_port,
            "den_auth": False,
        },
    )

    # Yield the cluster
    yield local_cluster

    # If we are running in detached mode, leave the container running, else clean it up
    if not detached:
        cleanup()


@pytest.fixture(scope="session")
def docker_cluster_pk_ssh(request):
    """This basic cluster fixture is set up with:
    - Public key authentication
    - Nginx set up on startup to forward Runhouse HTTP server to port 443
    """

    # From pytest config
    detached = request.config.getoption("--detached")
    force_rebuild = request.config.getoption("--force-rebuild")

    # Ports to use on the Docker VM such that they don't conflict
    local_ssh_port = BASE_LOCAL_SSH_PORT + 2

    local_cluster, cleanup = set_up_local_cluster(
        image_name="keypair",
        container_name="rh-pk-ssh",
        dir_name="public-key-auth",
        keypath=str(
            Path(
                rh.configs.get("default_keypair", DEFAULT_KEYPAIR_KEYPATH)
            ).expanduser()
        ),
        reuse_existing_container=detached,
        force_rebuild=force_rebuild,
        port_fwds=[f"{local_ssh_port}:{DEFAULT_SSH_PORT}"],
        local_ssh_port=local_ssh_port,
        additional_cluster_init_args={
            "name": "docker_cluster_pk_ssh",
            "server_connection_type": "ssh",
        },
    )

    # Yield the cluster
    yield local_cluster

    # If we are running in detached mode, leave the container running, else clean it up
    if not detached:
        cleanup()


# These two clusters cannot be used in the same test together, they are are technically
# the same image, but we switch the cluster parameters.
# These depend on the base fixture above and swap out the cluster parameters as needed.
@pytest.fixture(scope="function")
def docker_cluster_pk_tls_den_auth(docker_cluster_pk_tls_exposed, logged_in_account):
    """This is one of our key use cases -- TLS + Den Auth set up.

    We use the base fixture, which already has caddy set up to port forward
    from 443 to 32300, and we set the cluster parameters to use the correct ports to communicate
    with the server, in case they had been changed.
    """
    return docker_cluster_pk_tls_exposed


@pytest.fixture(scope="function")
def docker_cluster_pk_ssh_den_auth(docker_cluster_pk_ssh, logged_in_account):
    """This is our other key use case -- SSH with any Den Auth.

    We use the base fixture, and ignore the caddy/https setup, instead just communicating with
    the cluster using the base SSH credentials already present on the machine. We send a request
    to enable den auth server side.
    """
    docker_cluster_pk_ssh.enable_den_auth(flush=False)
    return docker_cluster_pk_ssh


@pytest.fixture(scope="function")
def docker_cluster_pk_ssh_no_auth(
    docker_cluster_pk_ssh,
):
    """This is our other key use case -- SSH without any Den Auth.

    We use the base fixture, and ignore the caddy/https setup, instead just communicating with
    the cluster using the base SSH credentials already present on the machine. We send a request
    to disable den auth server side.
    """
    docker_cluster_pk_ssh.disable_den_auth()
    return docker_cluster_pk_ssh


@pytest.fixture(scope="session")
def docker_cluster_pk_http_exposed(request):
    """This basic cluster fixture is set up with:
    - Public key authentication
    - Den auth enabled
    - Caddy set up on startup to forward Runhouse HTTP Server to port 80
    """

    # From pytest config
    detached = request.config.getoption("--detached")
    force_rebuild = request.config.getoption("--force-rebuild")

    # Ports to use on the Docker VM such that they don't conflict
    local_ssh_port = BASE_LOCAL_SSH_PORT + 3
    local_client_port = LOCAL_HTTP_SERVER_PORT + 3

    local_cluster, cleanup = set_up_local_cluster(
        image_name="keypair",
        container_name="rh-pk-http-port-fwd",
        dir_name="public-key-auth",
        keypath=str(
            Path(
                rh.configs.get("default_keypair", DEFAULT_KEYPAIR_KEYPATH)
            ).expanduser()
        ),
        reuse_existing_container=detached,
        force_rebuild=force_rebuild,
        port_fwds=[
            f"{local_ssh_port}:{DEFAULT_SSH_PORT}",
            f"{local_client_port}:{DEFAULT_HTTP_PORT}",
        ],
        local_ssh_port=local_ssh_port,
        additional_cluster_init_args={
            "name": "docker_cluster_with_caddy",
            "server_connection_type": "none",
            "server_port": DEFAULT_HTTP_PORT,
            "client_port": local_client_port,
            "den_auth": True,
        },
    )
    # Yield the cluster
    yield local_cluster

    # If we are running in detached mode, leave the container running, else clean it up
    if not detached:
        cleanup()


@pytest.fixture(scope="session")
def docker_cluster_pwd_ssh_no_auth(request):
    """This basic cluster fixture is set up with:
    - Password authentication
    - No Den Auth
    - No caddy/port forwarding set up
    """

    # From pytest config
    detached = request.config.getoption("--detached")
    force_rebuild = request.config.getoption("--force-rebuild")

    # Ports to use on the Docker VM such that they don't conflict
    local_ssh_port = BASE_LOCAL_SSH_PORT + 4

    pwd_file = "docker_user_passwd"
    rh_parent_path = get_rh_parent_path()
    pwd = (rh_parent_path.parent / pwd_file).read_text().strip()

    local_cluster, cleanup = set_up_local_cluster(
        image_name="pwd",
        container_name="rh-pwd",
        dir_name="password-file-auth",
        pwd_file="docker_user_passwd",
        reuse_existing_container=detached,
        force_rebuild=force_rebuild,
        port_fwds=[f"{local_ssh_port}:{DEFAULT_SSH_PORT}"],
        local_ssh_port=local_ssh_port,
        additional_cluster_init_args={
            "name": "docker_cluster_pwd_ssh_no_auth",
            "server_connection_type": "ssh",
            "ssh_creds": {"ssh_user": SSH_USER, "password": pwd},
        },
    )
    # Yield the cluster
    yield local_cluster

    # If we are running in detached mode, leave the container running, else clean it up
    if not detached:
        cleanup()


@pytest.fixture(scope="session")
def docker_cluster_pk_ssh_telemetry(request, detached=True):
    """This basic cluster fixture is set up with:
    - Public key authentication
    - No Den Auth
    - No caddy/port forwarding set up
    - Telemetry enabled
    """

    # From pytest config
    detached = request.config.getoption("--detached")
    force_rebuild = request.config.getoption("--force-rebuild")

    # Ports to use on the Docker VM such that they don't conflict
    local_ssh_port = BASE_LOCAL_SSH_PORT + 5

    local_cluster, cleanup = set_up_local_cluster(
        image_name="keypair-telemetry",
        container_name="rh-pk-telemetry",
        dir_name="public-key-auth",
        keypath=str(
            Path(
                rh.configs.get("default_keypair", DEFAULT_KEYPAIR_KEYPATH)
            ).expanduser()
        ),
        reuse_existing_container=detached,
        force_rebuild=force_rebuild,
        port_fwds=[f"{local_ssh_port}:{DEFAULT_SSH_PORT}"],
        local_ssh_port=local_ssh_port,
        additional_cluster_init_args={
            "name": "docker_cluster_pk_ssh_telemetry",
            "server_connection_type": "ssh",
            "use_local_telemetry": True,
        },
    )
    # Yield the cluster
    yield local_cluster

    # If we are running in detached mode, leave the container running, else clean it up
    if not detached:
        cleanup()


@pytest.fixture(scope="session")
def friend_account_logged_in_docker_cluster_pk_ssh(request):
    """
    This fixture is not parameterized for every test; it is a separate cluster started with a test account
    (username: kitchen_tester) in order to test sharing resources with other users.
    """

    # From pytest config
    detached = request.config.getoption("--detached")
    force_rebuild = request.config.getoption("--force-rebuild")
    with friend_account():
        # Ports to use on the Docker VM such that they don't conflict
        local_ssh_port = BASE_LOCAL_SSH_PORT + 6
        local_cluster, cleanup = set_up_local_cluster(
            image_name="keypair",
            container_name="rh-pk-test-acct",
            dir_name="public-key-auth",
            keypath=str(
                Path(
                    rh.configs.get("default_keypair", DEFAULT_KEYPAIR_KEYPATH)
                ).expanduser()
            ),
            reuse_existing_container=detached,
            force_rebuild=force_rebuild,
            port_fwds=[f"{local_ssh_port}:{DEFAULT_SSH_PORT}"],
            local_ssh_port=local_ssh_port,
            additional_cluster_init_args={
                "name": "friend_account_logged_in_docker_cluster_pk_ssh",
                "server_connection_type": "ssh",
                "den_auth": "den_auth" in request.keywords,
            },
            logged_in=True,
        )

    yield local_cluster

    # If we are running in detached mode, leave the container running, else clean it up
    if not detached:
        cleanup()


@pytest.fixture(scope="session")
def shared_cluster(friend_account_logged_in_docker_cluster_pk_ssh, logged_in_account):
    username_to_share = rh.configs.username
    with friend_account():
        # Share the cluster with the test account
        friend_account_logged_in_docker_cluster_pk_ssh.share(
            username_to_share, notify_users=False, access_level="read"
        )

    return friend_account_logged_in_docker_cluster_pk_ssh
