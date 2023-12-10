import httpx

import pytest
import pytest_asyncio

import runhouse as rh

from runhouse.servers.http.http_server import app, HTTPServer
from runhouse.servers.obj_store import ObjStore

# Note: API Server will run on local docker container
BASE_URL = "http://localhost:32300"

BASE_ENV_ACTOR_NAME = "base"
CACHE_ENV_ACTOR_NAME = "auth_cache"


# -------- HELPERS ----------- #
def summer(a, b):
    return a + b


def get_ray_servlet(env_name):
    """Helper method for getting auth servlet and base env servlet"""
    import ray

    ray.init(
        ignore_reinit_error=True,
        runtime_env=None,
        namespace="runhouse",
    )

    servlet = HTTPServer.get_env_servlet(
        env_name=env_name,
        create=True,
        runtime_env=None,
    )

    return servlet


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
    yield get_ray_servlet(BASE_ENV_ACTOR_NAME)


@pytest.fixture(scope="session")
def cache_servlet():
    yield get_ray_servlet(CACHE_ENV_ACTOR_NAME)


@pytest.fixture(scope="session")
def obj_store(request):
    base_obj_store = ObjStore()

    # Use the parameter to set the name of the servlet actor to use
    actor_name = request.param

    # Ensure servlet is running
    _ = get_ray_servlet(actor_name)

    base_obj_store.set_name(actor_name)

    yield base_obj_store
