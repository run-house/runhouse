import json
from pathlib import Path

import httpx

import pytest
import pytest_asyncio

import runhouse as rh

from runhouse.globals import rns_client
from runhouse.servers.http.http_server import app, HTTPServer

from tests.utils import friend_account, get_ray_servlet_and_obj_store

# Note: API Server will run on local docker container
BASE_URL = "http://localhost:32300"

BASE_ENV_ACTOR_NAME = "base"


# -------- HELPERS ----------- #
def summer(a, b):
    return a + b


# -------- FIXTURES ----------- #
@pytest.fixture(scope="module")
def http_client():
    with httpx.Client(base_url=BASE_URL, timeout=None) as client:
        yield client


@pytest_asyncio.fixture(scope="function")
async def async_http_client():
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=None) as client:
        yield client


@pytest.fixture(scope="session")
def local_cluster():
    return rh.cluster(
        name="faux_local_cluster",
        server_connection_type="none",
        host="localhost",
    )


@pytest.fixture(scope="session")
def local_client():
    from fastapi.testclient import TestClient

    HTTPServer(from_test=True)
    client = TestClient(app)

    yield client


@pytest.fixture(scope="function")
def local_client_with_den_auth(logged_in_account):
    from fastapi.testclient import TestClient

    HTTPServer(from_test=True)
    HTTPServer.enable_den_auth(flush=False)
    client = TestClient(app)
    with friend_account():
        client.headers = rns_client.request_headers()

    yield client

    HTTPServer.disable_den_auth()


@pytest.fixture(scope="session")
def test_servlet():
    servlet, _ = get_ray_servlet_and_obj_store("test_servlet")
    yield servlet


@pytest.fixture(scope="function")
def obj_store(request):

    # Use the parameter to set the name of the servlet actor to use
    env_servlet_name = request.param
    _, test_obj_store = get_ray_servlet_and_obj_store(env_servlet_name)

    # Clears everything, not just what's in this env servlet
    test_obj_store.clear()

    yield test_obj_store


@pytest.fixture(scope="class")
def setup_cluster_config(local_cluster):
    # Create a temporary directory that simulates the user's home directory
    home_dir = Path("~/.rh").expanduser()
    home_dir.mkdir(exist_ok=True)

    cluster_config_path = home_dir / "cluster_config.json"

    try:
        with open(cluster_config_path, "w") as file:
            json.dump(local_cluster.config_for_rns, file)

        yield

    finally:
        if cluster_config_path.exists():
            cluster_config_path.unlink()
