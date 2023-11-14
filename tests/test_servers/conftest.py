import httpx

import pytest
import pytest_asyncio

import runhouse as rh

from runhouse.servers.http.http_server import app, HTTPServer
from runhouse.servers.obj_store import ObjStore

# Note: Server will run on local docker container
BASE_URL = "http://localhost:32300"

BASE_ENV_ACTOR_NAME = "base"
CACHE_ENV_ACTOR_NAME = "auth_cache"


# -------- HELPERS ----------- #
def summer(a, b):
    return a + b


def http_server_is_up():
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
def obj_store(request, base_servlet):
    base_obj_store = ObjStore()

    # Use the parameter to set the name of the servlet actor to use
    actor_name = request.param
    base_obj_store.set_name(actor_name)

    yield base_obj_store
