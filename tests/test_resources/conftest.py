import pytest

from runhouse.resources.resource import Resource

from tests.conftest import init_args

######## Constants ########

RESOURCE_NAME = "my_resource"

######## Fixtures ########


@pytest.fixture(scope="function")
def resource(request):
    return request.getfixturevalue(request.param)


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
