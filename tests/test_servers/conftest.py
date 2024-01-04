import json
from pathlib import Path

import httpx

import pytest
import pytest_asyncio

import runhouse as rh

from runhouse.globals import rns_client
from runhouse.servers.http.http_server import app, HTTPServer

from tests.utils import get_ray_servlet, get_test_obj_store, test_account

# Note: API Server will run on local docker container
BASE_URL = "http://localhost:32300"

BASE_ENV_ACTOR_NAME = "base"


# -------- HELPERS ----------- #
def summer(a, b):
    return a + b


# -------- FIXTURES ----------- #
@pytest.fixture(scope="module")
def http_client():
    with httpx.Client(base_url=BASE_URL, timeout=10.0) as client:
        yield client


@pytest_asyncio.fixture(scope="function")
async def async_http_client():
    async with httpx.AsyncClient(base_url=BASE_URL) as client:
        yield client


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
    with test_account():
        client.headers = rns_client.request_headers

    yield client

    HTTPServer.disable_den_auth()


@pytest.fixture(scope="session")
def base_servlet():
    yield get_ray_servlet(BASE_ENV_ACTOR_NAME)


@pytest.fixture(scope="function")
def obj_store(request):

    # Use the parameter to set the name of the servlet actor to use
    env_servlet_name = request.param
    test_obj_store = get_test_obj_store(env_servlet_name)

    # Clears everything, not just what's in this env servlet
    test_obj_store.clear()

    yield test_obj_store


@pytest.fixture(scope="class")
def setup_cluster_config():
    # Create a temporary directory that simulates the user's home directory
    home_dir = Path("~/.rh").expanduser()
    home_dir.mkdir(exist_ok=True)

    cluster_config_path = home_dir / "cluster_config.json"
    rns_address = "/kitchen_tester/local_cluster"

    cluster_config = {
        "name": rns_address,
        "resource_type": "cluster",
        "resource_subtype": "Cluster",
        "server_port": 32300,
        "den_auth": True,
        "server_connection_type": "ssh",
        "ips": ["localhost"],
    }
    try:
        c = rh.Cluster.from_name(rns_address)
    except ValueError:
        c = None

    try:
        if not c:
            current_username = rh.configs.get("username")
            if current_username:
                with test_account():
                    c = rh.cluster(name="local_cluster", den_auth=True).save()
                    c.share(current_username, access_level="write", notify_users=False)

        with open(cluster_config_path, "w") as file:
            json.dump(cluster_config, file)

        yield

    finally:
        if cluster_config_path.exists():
            cluster_config_path.unlink()
