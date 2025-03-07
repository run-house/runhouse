import importlib
import logging
import os
import shlex
import subprocess
import time
from pathlib import Path

from typing import Any, Dict, List

import pytest

import runhouse as rh

from runhouse.constants import DEFAULT_HTTP_PORT, DEFAULT_HTTPS_PORT, DEFAULT_SSH_PORT
from runhouse.globals import rns_client
from runhouse.resources.images import Image
from tests.conftest import init_args

from tests.constants import TEST_ENV_VARS, TEST_REQS
from tests.utils import friend_account, get_default_keypair_path, setup_test_base

SSH_USER = "rh-docker-user"
BASE_LOCAL_SSH_PORT = 32320
LOCAL_HTTPS_SERVER_PORT = 8443
LOCAL_HTTP_SERVER_PORT = 8080


def get_rh_parent_path():
    return Path(importlib.util.find_spec("runhouse").origin).parent.parent


@pytest.fixture(scope="function")
def cluster(request):
    return request.getfixturevalue(request.param)


@pytest.fixture(scope="session")
def named_cluster():
    from tests.fixtures.secret_fixtures import provider_secret_values

    # Name cannot be the only arg or we attempt to load from name, and throw an error when it's not found
    args = dict(
        name="test-simple-cluster",
        host="my_url.com",
        ssh_creds=rh.provider_secret(
            provider="ssh", values=provider_secret_values["ssh"]
        ),
    )
    c = rh.cluster(**args)
    init_args[id(c)] = args
    return c


@pytest.fixture(scope="session")
def local_daemon(request):
    if not request.config.getoption("--detached") or rh.here == "file":
        logging.info("Starting local_daemon.")
        local_rh_package_path = Path(importlib.util.find_spec("runhouse").origin).parent
        subprocess.run(
            "runhouse server restart",
            shell=True,  # Needed because we need to be in the right conda env
            cwd=local_rh_package_path,
            text=True,
            check=True,
        )

    try:
        # Make sure the object store is set up correctly
        assert rh.here.on_this_cluster()
        yield rh.here

    finally:
        if not request.config.getoption("--detached"):
            subprocess.run("runhouse server stop", text=True, shell=True)


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
    dockerfile_path = rh_parent_path / f"docker/testing/{dir_name}/Dockerfile"
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
        ["docker", "run", "--name", container_name, "-d", "--rm", "--shm-size=5.04gb"]
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


def run_shell_command(subprocess, cmd: List[str], cwd: str = None):
    # Run the command and wait for it to complete
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd or Path.cwd())

    if result.returncode != 0:
        print("subprocess failed, stdout: " + result.stdout)
        print("subprocess failed, stderr: " + result.stderr)

    # Check for success
    assert result.returncode == 0


def popen_shell_command(subprocess, command: List[str], cwd: str = None):
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

    # We need to save before bringing up so the correct rns_address is written to the cluster config.
    # This is necessary because we're turning on Den auth without saving the config.yaml (containing
    # the "owner's" token) to the container in many cases, so we're relying on authenticating the caller
    # to the server through Den. If the cluster isn't saved before coming up, the config in the cluster servlet
    # doesn't have the rns address, and the auth verification to Den fails.
    rh_cluster.save()

    # Can't use the defaults_cache alone because we may need the token or username from the env variables
    config = rh.configs.defaults_cache
    config["token"] = rh.configs.token
    config["username"] = rh.configs.username

    # Runhouse is already installed on the Docker clusters, but we need to sync our actual version
    rh_cluster.start_server(resync_rh=True)

    if not rh_cluster.image:
        setup_test_base(rh_cluster, logged_in=logged_in)

    def cleanup():
        docker_client.containers.get(container_name).stop()
        docker_client.containers.prune()
        docker_client.images.prune()
        if rh_cluster._creds and "ssh-secret" in rh_cluster._creds.name:
            # secret was generated for the test cluster
            rh_cluster._creds.delete_configs()
        rh_cluster.delete_configs()

    return rh_cluster, cleanup


@pytest.fixture(scope="session")
def docker_cluster_pk_tls_exposed(request, test_rns_folder):
    """This basic cluster fixture is set up with:
    - Public key authentication
    - Caddy set up on startup to forward Runhouse HTTP server to port 443
    """
    import os

    # From pytest config
    detached = request.config.getoption("--detached")
    force_rebuild = request.config.getoption("--force-rebuild")
    api_server_url = request.config.getoption("--api-server-url")

    if not api_server_url:
        api_server_url = rns_client.api_server_url
    os.environ["API_SERVER_URL"] = api_server_url

    # Ports to use on the Docker VM such that they don't conflict
    local_ssh_port = BASE_LOCAL_SSH_PORT + 1
    local_client_port = LOCAL_HTTPS_SERVER_PORT + 1

    local_cluster, cleanup = set_up_local_cluster(
        image_name="keypair",
        container_name="rh-pk-tls-port-fwd",
        dir_name="public-key-auth",
        keypath=get_default_keypair_path(),
        reuse_existing_container=detached,
        force_rebuild=force_rebuild,
        port_fwds=[
            f"{local_ssh_port}:{DEFAULT_SSH_PORT}",
            f"{local_client_port}:{DEFAULT_HTTPS_PORT}",
        ],
        local_ssh_port=local_ssh_port,
        additional_cluster_init_args={
            "name": f"{test_rns_folder}_docker_cluster_pk_tls_exposed",
            "server_connection_type": "tls",
            "server_port": DEFAULT_HTTPS_PORT,
            "client_port": local_client_port,
            "den_auth": True,
        },
    )

    # Yield the cluster
    yield local_cluster

    # If we are running in detached mode, leave the container running, else clean it up
    if not detached:
        cleanup()
    else:
        local_cluster.delete_configs()


@pytest.fixture(scope="session")
def docker_cluster_pk_ssh(request, test_org_rns_folder):
    """This basic cluster fixture is set up with:
    - Public key authentication
    - Caddy set up on startup to forward Runhouse HTTP server to port 443
    - Default image with Ray 2.30.0
    """
    # From pytest config
    detached = request.config.getoption("--detached")
    force_rebuild = request.config.getoption("--force-rebuild")
    api_server_url = request.config.getoption("--api-server-url")

    if not api_server_url:
        api_server_url = rns_client.api_server_url
    os.environ["API_SERVER_URL"] = api_server_url

    # Ports to use on the Docker VM such that they don't conflict
    local_ssh_port = BASE_LOCAL_SSH_PORT + 2
    default_image = (
        Image(name="default_image")
        .set_env_vars(env_vars=TEST_ENV_VARS)
        .pip_install(reqs=TEST_REQS + ["ray==2.30.0"])
    )

    local_cluster, cleanup = set_up_local_cluster(
        image_name="keypair",
        container_name="rh-pk-ssh",
        dir_name="public-key-auth",
        keypath=get_default_keypair_path(),
        reuse_existing_container=detached,
        force_rebuild=force_rebuild,
        port_fwds=[f"{local_ssh_port}:{DEFAULT_SSH_PORT}"],
        local_ssh_port=local_ssh_port,
        additional_cluster_init_args={
            "name": f"{test_org_rns_folder}_docker_cluster_pk_ssh",
            "server_connection_type": "ssh",
            "image": default_image,
        },
    )

    # Yield the cluster
    yield local_cluster

    # If we are running in detached mode, leave the container running, else clean it up
    if not detached:
        cleanup()
    else:
        local_cluster.delete_configs()


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
def docker_cluster_pk_http_exposed(request, test_rns_folder):
    """This basic cluster fixture is set up with:
    - Public key authentication
    - Den auth disabled (to mimic VPC)
    - Caddy set up on startup to forward Runhouse HTTP Server to port 80
    - Default conda image with Python 3.11 and Ray 2.30.0
    """
    # From pytest config
    detached = request.config.getoption("--detached")
    force_rebuild = request.config.getoption("--force-rebuild")
    api_server_url = request.config.getoption("--api-server-url")

    if not api_server_url:
        api_server_url = rns_client.api_server_url
    os.environ["API_SERVER_URL"] = api_server_url

    # Ports to use on the Docker VM such that they don't conflict
    local_ssh_port = BASE_LOCAL_SSH_PORT + 3
    local_client_port = LOCAL_HTTP_SERVER_PORT + 3

    default_image = (
        Image(name="default_image")
        .setup_conda_env(
            conda_env_name="base_env",
            conda_config={"dependencies": ["python=3.11"], "name": "base_env"},
        )
        .pip_install(TEST_REQS)
    )

    local_cluster, cleanup = set_up_local_cluster(
        image_name="keypair-conda",
        container_name="rh-pk-http-port-fwd",
        dir_name="public-key-auth-conda",
        keypath=get_default_keypair_path(),
        reuse_existing_container=detached,
        force_rebuild=force_rebuild,
        port_fwds=[
            f"{local_ssh_port}:{DEFAULT_SSH_PORT}",
            f"{local_client_port}:{DEFAULT_HTTP_PORT}",
        ],
        local_ssh_port=local_ssh_port,
        additional_cluster_init_args={
            "name": f"{test_rns_folder}_docker_cluster_with_caddy",
            "server_connection_type": "none",
            "server_port": DEFAULT_HTTP_PORT,
            "client_port": local_client_port,
            "den_auth": False,
            "image": default_image,
        },
    )
    # Yield the cluster
    yield local_cluster

    # If we are running in detached mode, leave the container running, else clean it up
    if not detached:
        cleanup()
    else:
        local_cluster.delete_configs()


@pytest.fixture(scope="session")
def docker_cluster_pwd_ssh_no_auth(request, test_rns_folder):
    """This basic cluster fixture is set up with:
    - Password authentication
    - No Den Auth
    - No caddy/port forwarding set up
    - Python version 3.11 specified in image and using uv
    """
    import os

    # From pytest config
    detached = request.config.getoption("--detached")
    force_rebuild = request.config.getoption("--force-rebuild")
    api_server_url = request.config.getoption("--api-server-url")

    if not api_server_url:
        api_server_url = rns_client.api_server_url
    os.environ["API_SERVER_URL"] = api_server_url

    # Ports to use on the Docker VM such that they don't conflict
    local_ssh_port = BASE_LOCAL_SSH_PORT + 4

    pwd_file = "docker_user_passwd"
    rh_parent_path = get_rh_parent_path()
    pwd = (rh_parent_path.parent / pwd_file).read_text().strip()

    default_image = Image(python_version="3.11").uv_install(TEST_REQS)

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
            "name": f"{test_rns_folder}_docker_cluster_pwd_ssh_no_auth",
            "server_connection_type": "ssh",
            "ssh_creds": {"ssh_user": SSH_USER, "password": pwd},
            "image": default_image,
        },
    )
    # Yield the cluster
    yield local_cluster

    # If we are running in detached mode, leave the container running, else clean it up
    if not detached:
        cleanup()
    else:
        local_cluster.delete_configs()


@pytest.fixture(scope="session")
def friend_account_logged_in_docker_cluster_pk_ssh(request, test_rns_folder):
    """
    This fixture is not parameterized for every test; it is a separate cluster started with a test account
    (username: kitchen_tester) in order to test sharing resources with other users.
    """
    import os

    # From pytest config
    detached = request.config.getoption("--detached")
    force_rebuild = request.config.getoption("--force-rebuild")
    api_server_url = request.config.getoption("--api-server-url")

    if not api_server_url:
        api_server_url = rns_client.api_server_url
    os.environ["API_SERVER_URL"] = api_server_url

    with friend_account():
        # Ports to use on the Docker VM such that they don't conflict
        local_ssh_port = BASE_LOCAL_SSH_PORT + 5
        local_cluster, cleanup = set_up_local_cluster(
            image_name="keypair",
            container_name="rh-pk-test-acct",
            dir_name="public-key-auth",
            keypath=get_default_keypair_path(),
            reuse_existing_container=detached,
            force_rebuild=force_rebuild,
            port_fwds=[f"{local_ssh_port}:{DEFAULT_SSH_PORT}"],
            local_ssh_port=local_ssh_port,
            additional_cluster_init_args={
                "name": f"{test_rns_folder}_friend_account_logged_in_docker_cluster_pk_ssh",
                "server_connection_type": "ssh",
                "den_auth": "den_auth" in request.keywords,
            },
            logged_in=True,
        )

    yield local_cluster

    # If we are running in detached mode, leave the container running, else clean it up
    if not detached:
        cleanup()
    else:
        local_cluster.delete_configs()


@pytest.fixture(scope="session")
def shared_cluster(friend_account_logged_in_docker_cluster_pk_ssh):
    username_to_share = rh.configs.username

    # Enable den auth to properly test resource access on shared resources
    friend_account_logged_in_docker_cluster_pk_ssh.enable_den_auth()

    with friend_account():
        # Share the cluster with the test account
        friend_account_logged_in_docker_cluster_pk_ssh.share(
            username_to_share, notify_users=False, access_level="read"
        )

    return friend_account_logged_in_docker_cluster_pk_ssh


@pytest.fixture(scope="session")
def shared_function(shared_cluster):
    from tests.test_servers.conftest import summer

    username_to_share = rh.configs.username
    with friend_account():
        # Create function on shared cluster with the same test account
        f = rh.function(summer).to(shared_cluster).save()

        # Share the cluster & function with the current account
        f.share(username_to_share, access_level="read", notify_users=False)

    return f
