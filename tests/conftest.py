import asyncio
import enum
import subprocess

import pytest

import runhouse as rh
from runhouse.constants import RAY_KILL_CMD, RAY_START_CMD

"""
HOW TO USE FIXTURES IN RUNHOUSE TESTS

You can parameterize a fixture to run on many variations for any test.

Make sure your parameterized fixture is defined in this file (can be imported from some other file)
like so:

@pytest.fixture(scope="function")
def cluster(request):
    return request.getfixturevalue(request.param)

You can define default_fixtures for any parameterized fixture below.

You can use MAP_FIXTURES to map a parameterized fixture to a different name. This
is useful if you have a subclass that you want to test the parent's parameterized
fixtures with.  Moreover, you can override the fixture parameters in your actual Test class.
See examples for both of these below:

class TestCluster(tests.test_resources.test_resource.TestResource):

    MAP_FIXTURES = {"resource": "cluster"}

    UNIT = {"cluster": ["named_cluster"]}
    LOCAL = {
        "cluster": [
            "docker_cluster_pk_ssh_no_auth",
            "docker_cluster_pk_ssh_den_auth",
            "docker_cluster_pwd_ssh_no_auth",
        ]
    }
    MINIMAL = {"cluster": ["ondemand_aws_docker_cluster"]}
    RELEASE = {
        "cluster": [
            "ondemand_aws_docker_cluster",
            "static_cpu_pwd_cluster",
        ]
    }
    MAXIMAL = {
        "cluster": [
            "docker_cluster_pk_ssh_no_auth",
            "docker_cluster_pk_ssh_den_auth",
            "docker_cluster_pwd_ssh_no_auth",
            "ondemand_aws_docker_cluster",
            "static_cpu_pwd_cluster",
            "multinode_cpu_docker_conda_cluster"
        ]
    }

Some key things to avoid:
- Avoid ever importing from any conftest.py file. This can cause erratic
behavior in fixture initialization, and should always be avoided. Put test
utility items in `tests/<some path>` and import from there instead.

- Avoid using nested conftest.py files if we can avoid it. We should be
able to put most of our info in our top level conftest.py file, with
`tests/fixtures/<some path>` as an organizational spot for more fixtures.
Imports that you see below from nested conftest.py files will slowly be eliminated.

"""


class TestLevels(str, enum.Enum):
    UNIT = "unit"
    LOCAL = "local"
    MINIMAL = "minimal"
    RELEASE = "release"
    MAXIMAL = "maximal"


DEFAULT_LEVEL = TestLevels.UNIT

TEST_LEVEL_HIERARCHY = {
    TestLevels.UNIT: 0,
    TestLevels.LOCAL: 1,
    TestLevels.MINIMAL: 2,
    TestLevels.RELEASE: 3,
    TestLevels.MAXIMAL: 4,
}


def pytest_addoption(parser):
    parser.addoption(
        "--level",
        action="store",
        default=DEFAULT_LEVEL,
        help="Fixture set to spin up: unit, local, minimal, release, or maximal",
    )
    parser.addoption(
        "--force-rebuild",
        action="store_true",
        default=False,
        help="Force rebuild of the relevant Runhouse image",
    )
    parser.addoption(
        "--detached",
        action="store_true",
        default=False,
        help="Whether to run container in detached mode",
    )
    parser.addoption(
        "--ignore-filters",
        action="store_true",
        default=False,
        help="Don't filter tests by marks.",
    )
    parser.addoption(
        "--restart-server",
        action="store_true",
        default=False,
        help="Restart the server on the cluster fixtures.",
    )

    parser.addoption(
        "--api-server-url",
        action="store",
        default="https://api.run.house",
        help="URL of Runhouse Den",
    )


def pytest_generate_tests(metafunc):
    level = metafunc.config.getoption("level")
    level_fixtures = getattr(
        metafunc.cls or metafunc.module, level.upper(), default_fixtures[level]
    )

    # If a child test suite wants to override the fixture name, it can do so by setting it in the mapping
    # e.g. in the TestCluster, when the parent tests in TestResource run with the "resource" fixture, we want
    # to swap in the "cluster" fixtures instead so those resource tests run on the clusters. The resulting
    # level_fixtures dict will include the cluster fixtures for both the "cluster" and "resource" keys.
    mapping = getattr(metafunc.cls or metafunc.module, "MAP_FIXTURES", {})
    for k in mapping.keys():
        level_fixtures[k] = level_fixtures[mapping[k]]

    for fixture_name, fixture_list in level_fixtures.items():
        if fixture_name in metafunc.fixturenames:
            metafunc.parametrize(fixture_name, fixture_list, indirect=True)


def pytest_collection_modifyitems(config, items):
    ignore_filters = config.getoption("ignore_filters")
    request_level = config.getoption("level")
    if not ignore_filters:
        new_items = []

        for item in items:
            test_level = item.get_closest_marker("level")
            if (
                test_level is not None
                and TEST_LEVEL_HIERARCHY[test_level.args[0]]
                <= TEST_LEVEL_HIERARCHY[request_level]
            ):
                new_items.append(item)

        items[:] = new_items


def pytest_configure(config):
    import os

    os.environ["API_SERVER_URL"] = config.getoption("api_server_url")
    pytest.init_args = {}
    subprocess.run(RAY_START_CMD, shell=True)


def pytest_sessionfinish(session, exitstatus):
    subprocess.run(RAY_KILL_CMD, shell=True)


init_args = {}


@pytest.fixture(scope="function")
def logged_in_account():
    """Helper fixture for tests which require the logged-in test account. Throws an error if the wrong account
    is logged-in for some reason, and skips the test if the logged in state is not available."""
    token = rh.globals.configs.token
    if not token:
        pytest.skip("`RH_TOKEN` or ~/.rh/config.yaml not set, skipping test.")

    username = rh.globals.configs.username
    if username == "kitchen_tester":
        raise ValueError(
            "The friend test account should not be active while running logged-in tests."
        )


# Have to override the event_loop fixture to make it session scoped
# The cluster keeps a client, which holds an AsyncClient, which holds an event loop,
# but the event loop is closed after one test is run. This causes the next test to fail saying the
# event loop is closed. This is a workaround to keep the event loop open for the entire session.
@pytest.fixture(scope="session")
def event_loop():
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# https://docs.pytest.org/en/6.2.x/fixture.html#conftest-py-sharing-fixtures-across-multiple-files

############## HELPERS ##############


# ----------------- Clusters -----------------

from tests.fixtures.docker_cluster_fixtures import (
    build_and_run_image,  # noqa: F401
    cluster,  # noqa: F401
    docker_cluster_pk_http_exposed,  # noqa: F401
    docker_cluster_pk_ssh,  # noqa: F401
    docker_cluster_pk_ssh_den_auth,  # noqa: F401
    docker_cluster_pk_ssh_no_auth,  # noqa: F401
    docker_cluster_pk_tls_den_auth,  # noqa: F401
    docker_cluster_pk_tls_exposed,  # noqa: F401
    docker_cluster_pwd_ssh_no_auth,  # noqa: F401
    friend_account_logged_in_docker_cluster_pk_ssh,  # noqa: F401
    local_daemon,  # noqa: F401
    named_cluster,  # noqa: F401
    shared_cluster,  # noqa: F401
    shared_function,  # noqa: F401
)

from tests.fixtures.on_demand_cluster_fixtures import (
    a10g_gpu_cluster,  # noqa: F401
    k80_gpu_cluster,  # noqa: F401
    multinode_cpu_docker_conda_cluster,  # noqa: F401
    multinode_gpu_cluster,  # noqa: F401
    ondemand_aws_docker_cluster,  # noqa: F401
    ondemand_aws_https_cluster_with_auth,  # noqa: F401
    ondemand_cluster,  # noqa: F401
    ondemand_gcp_cluster,  # noqa: F401
    ondemand_k8s_cluster,  # noqa: F401
    ondemand_k8s_docker_cluster,  # noqa: F401
    v100_gpu_cluster,  # noqa: F401
)

from tests.fixtures.resource_fixtures import (
    local_named_resource,  # noqa: F401
    named_resource,  # noqa: F401
    named_resource_for_org,  # noqa: F401
    resource,  # noqa: F401
    saved_resource,  # noqa: F401
    saved_resource_pool,  # noqa: F401
    test_org_rns_folder,  # noqa: F401
    test_rns_folder,  # noqa: F401
    unnamed_resource,  # noqa: F401
)

from tests.fixtures.static_cluster_fixtures import static_cpu_pwd_cluster  # noqa: F401


# ----------------- Folders -----------------

from tests.fixtures.folder_fixtures import (  # usort: skip
    cluster_folder,  # noqa: F401
    dest,  # noqa: F401
    folder,  # noqa: F401
    gcs_folder,  # noqa: F401
    local_folder,  # noqa: F401
    docker_cluster_folder,  # noqa: F401
    s3_folder,  # noqa: F401
)

# ----------------- Packages -----------------

from tests.fixtures.package_fixtures import (
    conda_package,  # noqa: F401
    git_package,  # noqa: F401
    installed_editable_package,  # noqa: F401
    installed_editable_package_copy,  # noqa: F401
    local_package,  # noqa: F401
    package,  # noqa: F401
    pip_package,  # noqa: F401
    reqs_package,  # noqa: F401
    s3_package,  # noqa: F401
)

from tests.fixtures.secret_fixtures import (
    anthropic_secret,  # noqa: F401
    aws_secret,  # noqa: F401
    azure_secret,  # noqa: F401
    cohere_secret,  # noqa: F401
    custom_provider_secret,  # noqa: F401
    gcp_secret,  # noqa: F401
    github_secret,  # noqa: F401
    huggingface_secret,  # noqa: F401
    kubeconfig_secret,  # noqa: F401
    lambda_secret,  # noqa: F401
    langchain_secret,  # noqa: F401
    openai_secret,  # noqa: F401
    pinecone_secret,  # noqa: F401
    secret,  # noqa: F401
    sky_secret,  # noqa: F401
    ssh_secret,  # noqa: F401
    test_secret,  # noqa: F401
    wandb_secret,  # noqa: F401
)

# ----------------- Envs -----------------

from tests.test_resources.test_envs.conftest import (
    base_conda_env,  # noqa: F401
    conda_env_from_local,  # noqa: F401
    conda_env_from_path,  # noqa: F401
    env,  # noqa: F401
    named_conda_env_from_dict,  # noqa: F401
    unnamed_env,  # noqa: F401
)

# ----------------- Modules -----------------

# ----------------- Functions -----------------
from tests.test_resources.test_modules.test_functions.conftest import (
    func_with_artifacts,  # noqa: F401
    slow_func,  # noqa: F401
    slow_running_func,  # noqa: F401
    summer_func,  # noqa: F401
    summer_func_shared,  # noqa: F401
    summer_func_with_auth,  # noqa: F401
)

########## DEFAULT LEVELS ##########

default_fixtures = {}
default_fixtures[TestLevels.UNIT] = {"cluster": ["named_cluster"]}
default_fixtures[TestLevels.LOCAL] = {
    "cluster": [
        # "docker_cluster_pk_ssh_no_auth",  # Represents private dev use case
        # "docker_cluster_pk_ssh_den_auth",  # Helps isolate Auth issues
        "docker_cluster_pk_tls_den_auth",  # Represents public app use case
        # "docker_cluster_pk_http_exposed",  # Represents within VPC use case
    ]
}
default_fixtures[TestLevels.MINIMAL] = {
    "cluster": [
        "ondemand_aws_docker_cluster",
    ]
}
default_fixtures[TestLevels.RELEASE] = {
    "cluster": [
        "ondemand_aws_docker_cluster",
        "ondemand_gcp_cluster",
        "ondemand_k8s_cluster",
        "ondemand_k8s_docker_cluster",
        "ondemand_aws_https_cluster_with_auth",
        "static_cpu_pwd_cluster",
    ]
}
default_fixtures[TestLevels.MAXIMAL] = {
    "cluster": [
        "docker_cluster_pk_ssh_no_auth",
        "docker_cluster_pk_ssh_den_auth",
        "docker_cluster_pwd_ssh_no_auth",
        "ondemand_aws_docker_cluster",
        "ondemand_gcp_cluster",
        "ondemand_k8s_cluster",
        "ondemand_k8s_docker_cluster",
        "ondemand_aws_https_cluster_with_auth",
        "multinode_cpu_docker_conda_cluster",
        "static_cpu_pwd_cluster",
        "multinode_gpu_cluster",  # for testing cluster status on multinode gpu.
    ]
}
