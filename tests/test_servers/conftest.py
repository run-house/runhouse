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


@pytest.fixture(scope="session")
def local_telemetry_agent_for_local_backend():
    """Local agent which exports to a locally running collector."""
<<<<<<< HEAD
    from runhouse.servers.telemetry import (
        TelemetryAgentExporter,
        TelemetryCollectorConfig,
=======
    from runhouse.servers.telemetry import TelemetryAgent, TelemetryAgentConfig

    # Note: For local testing purposes the backend collector will run on a lower port for HTTP (4319) and
    # GRPC (4316) to avoid collisions with the locally running agent
    agent_config = TelemetryAgentConfig(
        backend_collector_endpoint="localhost:4316",
        backend_collector_status_url="http://localhost:13133",
>>>>>>> ae221199 (instrument obj store method with otel agent)
    )

    # Note: For local testing purposes the backend collector will run on a different port for HTTP (4319),
    # GRPC (4316), and health check (13134), to avoid collisions with the locally running agent running on the
    # same machine on the standard ports (4317, 4318, and 13133).
    collector_config = TelemetryCollectorConfig(
        endpoint="localhost:4316", status_url="http://localhost:13134"
    )
    telemetry_agent = TelemetryAgentExporter(collector_config=collector_config)
    telemetry_agent.start(reload_config=True)

    assert telemetry_agent.is_up()

    # Confirm the backend collector is up and running before proceeding
    status_code = telemetry_agent.collector_health_check()
    if status_code != 200:
        raise ConnectionError(
<<<<<<< HEAD
            f"Failed to ping collector ({telemetry_agent.collector_config.status_url}), received status code "
            f"{status_code}. Is the collector up?")

    # Allow the agent to fully setup before collecting data
    time.sleep(0.5)
    yield telemetry_agent

    telemetry_agent.stop()
=======
            f"Failed to ping collector ({ta.config.backend_collector_status_url}), received status code "
            f"{status_code}. Is the collector up?"
        )

    # Allow the agent to fully setup before collecting data
    time.sleep(0.5)

    yield ta

    ta.stop()
>>>>>>> ae221199 (instrument obj store method with otel agent)


@pytest.fixture(scope="session")
def local_telemetry_agent_for_runhouse_backend():
    """Local agent which exports to the Runhouse collector."""
<<<<<<< HEAD
    from runhouse.servers.telemetry import TelemetryAgentExporter

    telemetry_agent = TelemetryAgentExporter()
    telemetry_agent.start(reload_config=True)

    assert telemetry_agent.is_up()

    # Confirm the backend collector is up and running before proceeding
    status_code = telemetry_agent.collector_health_check()
    if status_code != 200:
        raise ConnectionError(
            f"Failed to ping collector ({telemetry_agent.collector_config.status_url}), received status code "
=======
    from runhouse.servers.telemetry import TelemetryAgent

    ta = TelemetryAgent()
    ta.start(reload_config=True)

    assert ta.is_up()

    # Confirm the backend collector is up and running before proceeding
    status_code = ta.collector_health_check()
    if status_code != 200:
        raise ConnectionError(
            f"Failed to ping collector ({ta.config.backend_collector_status_url}), received status code "
>>>>>>> ae221199 (instrument obj store method with otel agent)
            f"{status_code}. Is the collector up?"
        )

    # Allow the agent to fully setup before collecting data
    time.sleep(0.5)

    yield telemetry_agent

    telemetry_agent.stop()


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
