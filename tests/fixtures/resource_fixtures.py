import uuid
from datetime import datetime

import pytest

from runhouse.constants import TEST_ORG
from runhouse.globals import rns_client
from runhouse.resources.resource import Resource

from tests.conftest import init_args

######## Constants ########

RESOURCE_NAME = "my_resource"
RESOURCES_SAVED_TO_ORG = ["docker_cluster_pk_ssh_den_auth"]


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
    folder_path = f"testing-{uuid.uuid4()}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    yield folder_path
    rns_client.delete_configs(folder_path)


@pytest.fixture(scope="session")
def test_org_rns_folder(test_rns_folder):
    folder_path = f"/{TEST_ORG}/{test_rns_folder}"
    yield folder_path
    rns_client.delete_configs(folder_path)


@pytest.fixture(scope="function")
def saved_resource(resource, saved_resource_pool, test_rns_folder):
    if not resource.name:
        pytest.skip("Resource must have a name to be saved")

    top_level_folder = None
    if resource.name not in saved_resource_pool:
        # Create a variant of the resource under a different name so we don't conflict with other tests or
        # or other runs of the test.
        resource_copy = resource.from_config(config=resource.config(), dryrun=True)
        if not resource.rns_address or resource.rns_address[:2] != "~/":
            # No need to vary the name for local resources
            # Put resource copies in a folder together so it's easier to clean up
            saved_to_org = resource.name in RESOURCES_SAVED_TO_ORG
            has_rns_address = resource.rns_address is not None
            test_org_resource = has_rns_address and resource.rns_address.startswith(
                f"/{TEST_ORG}"
            )

            top_level_folder = (
                f"/{TEST_ORG}"
                if saved_to_org or test_org_resource
                else rns_client.current_folder
            )

            resource_copy.name = f"{top_level_folder}/{test_rns_folder}/{resource.name}"
            if test_org_resource:
                from tests.utils import friend_account_in_org

                with friend_account_in_org():
                    # Need org access in order to save the resource to the org
                    # Remove subresources to avoid resaving shared resource which is not allowed
                    resource_copy._creds = None
                    resource_copy._default_env = None
                    saved_resource = resource_copy.save(folder=top_level_folder)
            else:
                saved_resource = resource_copy.save(folder=top_level_folder)
            saved_resource_pool[resource.name] = saved_resource
    return saved_resource_pool[resource.name]


@pytest.fixture(scope="session")
def unnamed_resource():
    args = {}
    r = Resource(**args)
    init_args[id(r)] = args
    return r


@pytest.fixture(scope="session")
def named_resource_for_org(test_org_rns_folder):
    # Resource saved for an org (as opposed to the current user based on the local rh config)
    args = {"name": f"{test_org_rns_folder}/{RESOURCE_NAME}"}
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
