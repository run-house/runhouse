import httpx
import pytest

from runhouse.resources.resource import Resource

from tests.conftest import init_args

######## Constants ########

RESOURCE_NAME = "my_resource"

# Note: API Server will run on local docker container
BASE_URL = "http://localhost:32300"

######## Fixtures ########


@pytest.fixture(scope="function")
def resource(request):
    return request.getfixturevalue(request.param.__name__)


@pytest.fixture(scope="session")
def unnamed_resource():
    args = {}
    r = Resource(**args)
    init_args[id(r)] = args
    return r


@pytest.fixture(scope="session")
def named_resource():
    args = {"name": RESOURCE_NAME}
    r = Resource(**args)
    init_args[id(r)] = args
    return r


@pytest.fixture(scope="session")
def local_named_resource():
    args = {"name": "~/" + RESOURCE_NAME}
    r = Resource(**args)
    init_args[id(r)] = args
    return r


def http_server_is_up():
    try:
        resp = httpx.get(f"{BASE_URL}/check")
        resp.raise_for_status()  # Will raise an exception for any status code 400 and above
        return True
    except httpx.HTTPError:
        return False
