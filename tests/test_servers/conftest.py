import json
import time
from pathlib import Path

import httpx
import pytest
import pytest_asyncio

import runhouse as rh

from runhouse.globals import rns_client
from runhouse.logger import get_logger

from runhouse.servers.http.certs import TLSCertConfig
from runhouse.servers.http.http_server import app, HTTPServer

from tests.utils import (
    friend_account,
    get_ray_cluster_servlet,
    get_ray_env_servlet_and_obj_store,
)

logger = get_logger(__name__)

# -------- HELPERS ----------- #
def summer(a, b):
    return a + b


def do_printing_and_logging(steps=3):
    for i in range(steps):
        # Wait to make sure we're actually streaming
        time.sleep(1)
        print(f"Hello from the cluster stdout! {i}")
        logger.info(f"Hello from the cluster logs! {i}")
    return steps


# -------- FIXTURES ----------- #
@pytest.fixture(scope="session")
def cert_config():
    cert_config = TLSCertConfig()
    address = "127.0.0.1"
    cert_config.generate_certs(address=address)

    yield cert_config

    # Clean up the generated files
    Path(cert_config.cert_path).unlink(missing_ok=True)
    Path(cert_config.key_path).unlink(missing_ok=True)


@pytest.fixture(scope="function")
def http_client(cluster, cert_config):
    addr = cluster.endpoint()
    with httpx.Client(base_url=addr, timeout=None, verify=False) as client:
        yield client


@pytest_asyncio.fixture(scope="function")
async def async_http_client(cluster, cert_config):
    addr = cluster.endpoint()
    async with httpx.AsyncClient(base_url=addr, timeout=None, verify=False) as client:
        yield client


@pytest_asyncio.fixture(scope="function")
def remote_func(cluster):
    return rh.function(summer).to(cluster)


@pytest_asyncio.fixture(scope="function")
def remote_log_streaming_func(cluster):
    return rh.function(do_printing_and_logging).to(cluster)


@pytest.fixture(scope="session")
def local_cluster():
    from tests.fixtures.secret_fixtures import provider_secret_values

    # Save to validate cluster access for HTTP requests
    return rh.cluster(
        name="faux_local_cluster",
        server_connection_type="none",
        host="localhost",
        ssh_creds=provider_secret_values["ssh"],
    ).save()


@pytest.fixture(scope="session")
def local_client():
    from fastapi.testclient import TestClient

    HTTPServer.initialize(from_test=True)
    client = TestClient(app)

    yield client


@pytest.fixture(scope="function")
def local_client_with_den_auth(logged_in_account):
    from fastapi.testclient import TestClient

    HTTPServer.initialize(from_test=True)
    HTTPServer.enable_den_auth(flush=False)
    client = TestClient(app)
    with friend_account():
        client.headers = rns_client.request_headers()

    yield client

    HTTPServer.disable_den_auth()


@pytest.fixture(scope="session")
def test_env_servlet():
    env_servlet, _ = get_ray_env_servlet_and_obj_store("test_env_servlet")
    yield env_servlet


@pytest.fixture(scope="session")
def test_cluster_servlet(request):
    cluster_servlet = get_ray_cluster_servlet()
    yield cluster_servlet


@pytest.fixture(scope="function")
def obj_store(request):

    # Use the parameter to set the name of the servlet actor to use
    env_servlet_name = request.param
    _, test_obj_store = get_ray_env_servlet_and_obj_store(env_servlet_name)

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
            json.dump(local_cluster.config(), file)

        yield

    finally:
        if cluster_config_path.exists():
            cluster_config_path.unlink()
