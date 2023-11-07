import warnings

from pathlib import Path

import httpx
import pytest
import pytest_asyncio

import runhouse as rh

from tests.conftest import build_and_run_image

# Note: Server will run on local docker container
BASE_URL = "http://localhost:32300"
KEYPATH = str(
    Path(
        rh.configs.get("default_keypair", "~/.ssh/runhouse/docker/id_rsa")
    ).expanduser()
)


# -------- HELPERS ----------- #
def summer(a, b):
    return a + b


def server_is_up():
    try:
        resp = httpx.get(f"{BASE_URL}/check")
        resp.raise_for_status()  # Will raise an exception for any status code 400 and above
        return True
    except httpx.HTTPError:
        return False


# -------- FIXTURES ----------- #
@pytest.fixture(scope="module")
def http_client():
    with httpx.Client(base_url=BASE_URL) as client:
        yield client


@pytest_asyncio.fixture(scope="function")
async def async_http_client():
    async with httpx.AsyncClient(base_url=BASE_URL) as client:
        yield client


@pytest.fixture(scope="session")
def base_cluster():
    c = rh.cluster(
        name="local-docker-slim-public-key-auth",
        host="localhost",
        den_auth=False,
        server_host="0.0.0.0",
        ssh_creds={
            "ssh_user": "rh-docker-user",
            "ssh_private_key": KEYPATH,
        },
    ).save()
    return c


@pytest.fixture(scope="session")
def cluster_with_auth():
    c = rh.cluster(
        name="local-docker-slim-public-key-auth-with-auth",
        host="localhost",
        den_auth=True,
        server_host="0.0.0.0",
        ssh_creds={
            "ssh_user": "rh-docker-user",
            "ssh_private_key": KEYPATH,
        },
    ).save()
    return c


@pytest.fixture(scope="session")
def docker_container(pytestconfig, cluster):
    if server_is_up():
        yield
        # no need to do anything else if server is already up
        return

    warnings.warn(
        "Server is not up. Launching server on local docker container. Once finished, the server "
        f"is addressable at: {BASE_URL}"
    )

    # Start the Docker container
    build_and_run_image(
        image_name="keypair",
        container_name="rh-slim-server-public-key",
        detached=True,
        dir_name="public-key-auth",
        keypath=KEYPATH,
        force_rebuild=pytestconfig.getoption("--force-rebuild"),
    )
    rh_config = rh.configs.load_defaults_from_file()
    rh.env(
        reqs=["pytest", "httpx", "pytest_asyncio"],
        working_dir=None,
        setup_cmds=[
            f"mkdir -p ~/.rh; touch ~/.rh/config.yaml; "
            f"echo '{rh_config}' > ~/.rh/config.yaml"
        ],
        name="base_env",
    ).to(cluster)

    if not server_is_up():
        # If the server still doesn't respond with a 200, raise an error
        raise RuntimeError(
            "The server is not up or not responding correctly after build. Make sure the runhouse "
            "package was copied into the container and the server was started correctly."
        )

    yield


@pytest.fixture(scope="session")
def docker_container_with_auth(pytestconfig, cluster_with_auth):
    if server_is_up():
        yield
        # no need to do anything else if server is already up
        return

    warnings.warn(
        "Server is not up. Launching server on local docker container. Once finished, the server "
        f"is addressable at: {BASE_URL}"
    )

    # Start the Docker container
    build_and_run_image(
        image_name="keypair",
        container_name="rh-slim-server-public-key-with-auth",
        detached=True,
        dir_name="public-key-auth",
        keypath=KEYPATH,
        force_rebuild=pytestconfig.getoption("--force-rebuild"),
    )
    rh_config = rh.configs.load_defaults_from_file()
    rh.env(
        reqs=["pytest", "httpx", "pytest_asyncio"],
        working_dir=None,
        setup_cmds=[
            f"mkdir -p ~/.rh; touch ~/.rh/config.yaml; "
            f"echo '{rh_config}' > ~/.rh/config.yaml"
        ],
        name="base_env",
    ).to(cluster_with_auth)

    if not server_is_up():
        # If the server still doesn't respond with a 200, raise an error
        raise RuntimeError(
            "The server is not up or not responding correctly after build. Make sure the runhouse "
            "package was copied into the container and the server was started correctly."
        )

    yield
