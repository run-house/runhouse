from datetime import datetime

import pytest

from runhouse.globals import rns_client
from runhouse.resources.resource import Resource

from tests.conftest import init_args

######## Constants ########

RESOURCE_NAME = "my_resource"

######## Fixtures ########


@pytest.fixture(scope="function")
def resource(request):
    return request.getfixturevalue(request.param)


@pytest.fixture(scope="session")
def saved_resource_pool():
    try:
        pool = {}
        yield pool
    finally:
        for res in pool.values():
            # Wrap in another try/except block so we can clean up as much as possible
            try:
                res.delete_configs()
            except Exception:
                pass


@pytest.fixture(scope="session")
def test_rns_folder():
    return f"testing-{datetime.now().strftime('%Y%m%d-%H%M%S')}"


@pytest.fixture(scope="function")
def saved_resource(resource, saved_resource_pool, test_rns_folder):
    if not resource.name:
        pytest.skip("Resource must have a name to be saved")

    if resource.name not in saved_resource_pool:
        # Create a variant of the resource under a different name so we don't conflict with other tests or
        # or other runs of the test.
        resource_copy = resource.from_config(
            config=resource.config_for_rns, dryrun=True
        )
        if not resource.rns_address or resource.rns_address[:2] != "~/":
            # No need to vary the name for local resources
            # Put resource copies in a folder together so it's easier to clean up
            resource_copy.name = (
                f"{rns_client.current_folder}/{test_rns_folder}/{resource.name}"
            )
        saved_resource_pool[resource.name] = resource_copy.save()
    return saved_resource_pool[resource.name]


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
