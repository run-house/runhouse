import subprocess
import warnings

import httpx

import pytest
import pytest_asyncio

import runhouse as rh

from runhouse.servers.http.http_server import app, HTTPServer
from runhouse.servers.obj_store import ObjStore

from tests.conftest import build_and_run_image

# Note: Server will run on local docker container
BASE_URL = "http://localhost:32300"

BASE_ENV_ACTOR_NAME = "base"
CACHE_ENV_ACTOR_NAME = "auth_cache"


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


# TODO [JL] create some sort of mock cluster that doesn't require a docker container?
@pytest.fixture(scope="session")
def local_cluster():
    c = rh.cluster(
        name="local_cluster", host="localhost", server_connection_type="none"
    )
    return c


@pytest.fixture(scope="session")
def local_client():
    from fastapi.testclient import TestClient

    HTTPServer()
    client = TestClient(app)

    yield client


@pytest.fixture(scope="function")
def local_client_with_den_auth():
    from fastapi.testclient import TestClient

    HTTPServer()
    HTTPServer.enable_den_auth()
    client = TestClient(app)

    yield client


@pytest.fixture(scope="session")
def docker_container(pytestconfig, cluster):
    """Local container which runs the HTTP server."""
    container_name = "rh-slim-server-public-key-auth"

    # Server will likely already be up bc of the fixture in the main conftest.py file (local_docker_cluster_public_key)
    server_already_up = server_is_up()
    if not server_already_up:
        warnings.warn(
            "Server is not up. Launching server on local docker container. Once finished, the server "
            f"will be addressable with URL: {BASE_URL}"
        )

        # Start the Docker container
        build_and_run_image(
            image_name="keypair",
            container_name=container_name,
            detached=True,
            dir_name="public-key-auth",
            use_keypath=True,
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

    detached = pytestconfig.getoption("--detached")
    if not detached:
        res = subprocess.run(["docker", "stop", container_name])
        if res.returncode != 0:
            raise RuntimeError(
                f"Failed to stop container {container_name}: {res.stderr}"
            )


@pytest.fixture(scope="session")
def base_servlet():
    import ray

    try:
        yield ray.get_actor(BASE_ENV_ACTOR_NAME, namespace="runhouse")
    except Exception as e:
        # Note: One easy way to ensure this base env actor is created is to run the HTTP server tests
        raise RuntimeError(e)


@pytest.fixture(scope="session")
def cache_servlet():
    import ray

    try:
        yield ray.get_actor(CACHE_ENV_ACTOR_NAME, namespace="runhouse")
    except Exception as e:
        raise RuntimeError(e)


@pytest.fixture(scope="session")
def obj_store(base_servlet):
    obj_store = ObjStore()
    obj_store.set_name(BASE_ENV_ACTOR_NAME)

    yield obj_store


@pytest.fixture(scope="session")
def obj_store_auth_cache(base_servlet):
    obj_store = ObjStore()
    obj_store.set_name(CACHE_ENV_ACTOR_NAME)

    yield obj_store
