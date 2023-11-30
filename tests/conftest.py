import contextlib
import enum

import pytest

from runhouse.globals import rns_client


class TestLevels(str, enum.Enum):
    UNIT = "unit"
    LOCAL = "local"
    MINIMAL = "minimal"
    THOROUGH = "thorough"
    MAXIMAL = "maximal"


DEFAULT_LEVEL = TestLevels.UNIT

TEST_LEVEL_HIERARCHY = {
    TestLevels.UNIT: 0,
    TestLevels.LOCAL: 1,
    TestLevels.MINIMAL: 2,
    TestLevels.THOROUGH: 3,
    TestLevels.MAXIMAL: 4,
}


def pytest_addoption(parser):
    parser.addoption(
        "--level",
        action="store",
        default=DEFAULT_LEVEL,
        help="Fixture set to spin up: unit, local, minimal, thorough, or maximal",
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


def pytest_configure():
    pytest.init_args = {}


init_args = {}


# https://docs.pytest.org/en/6.2.x/fixture.html#conftest-py-sharing-fixtures-across-multiple-files

############## HELPERS ##############


@pytest.fixture(scope="function")
@contextlib.contextmanager
def test_account():
    """Used for the purposes of testing resource sharing among different accounts.
    When inside the context manager, use the test account credentials before reverting back to the original
    account when exiting."""

    try:
        account = rns_client.load_account_from_env()
        if account is None:
            pytest.skip("`TEST_TOKEN` or `TEST_USERNAME` not set, skipping test.")
        yield account

    finally:
        rns_client.load_account_from_file()


# ----------------- Clusters -----------------

from tests.test_resources.test_clusters.conftest import (
    build_and_run_image,  # noqa: F401
    byo_cpu,  # noqa: F401
    cluster,  # noqa: F401
    local_docker_cluster_passwd,  # noqa: F401
    local_docker_cluster_public_key,  # noqa: F401
    local_docker_cluster_public_key_logged_in,  # noqa: F401
    local_docker_cluster_public_key_logged_out,  # noqa: F401
    local_docker_cluster_telemetry_public_key,  # noqa: F401
    local_docker_cluster_with_nginx,  # noqa: F401
    local_test_account_cluster_public_key,  # noqa: F401
    named_cluster,  # noqa: F401
    password_cluster,  # noqa: F401
    shared_cluster,  # noqa: F401
    static_cpu_cluster,  # noqa: F401
)
from tests.test_resources.test_clusters.test_on_demand_cluster.conftest import (
    a10g_gpu_cluster,  # noqa: F401
    k80_gpu_cluster,  # noqa: F401
    on_demand_cluster,  # noqa: F401
    ondemand_cluster,  # noqa: F401
    ondemand_cpu_cluster,  # noqa: F401
    ondemand_https_cluster_with_auth,  # noqa: F401
    v100_gpu_cluster,  # noqa: F401
)
from tests.test_resources.test_clusters.test_sagemaker_cluster.conftest import (
    other_sm_cluster,  # noqa: F401
    sm_cluster,  # noqa: F401
    sm_cluster_with_auth,  # noqa: F401
    sm_gpu_cluster,  # noqa: F401
)

# ----------------- Envs -----------------

from tests.test_resources.test_envs.conftest import (
    base_conda_env,  # noqa: F401
    base_env,  # noqa: F401
    conda_env_from_dict,  # noqa: F401
    conda_env_from_local,  # noqa: F401
    conda_env_from_path,  # noqa: F401
    env,  # noqa: F401
)

# ----------------- Blobs -----------------

from tests.test_resources.test_modules.test_blobs.conftest import (
    blob,  # noqa: F401
    blob_data,  # noqa: F401
    cluster_blob,  # noqa: F401
    cluster_file,  # noqa: F401
    file,  # noqa: F401
    gcs_blob,  # noqa: F401
    local_blob,  # noqa: F401
    local_file,  # noqa: F401
    s3_blob,  # noqa: F401
)

# ----------------- Folders -----------------

from tests.test_resources.test_modules.test_folders.conftest import (
    cluster_folder,  # noqa: F401
    folder,  # noqa: F401
    gcs_folder,  # noqa: F401
    local_folder,  # noqa: F401
    s3_folder,  # noqa: F401
)

# ----------------- Packages -----------------

from tests.test_resources.test_modules.test_folders.test_packages.conftest import (
    local_package,  # noqa: F401
    package,  # noqa: F401
    s3_package,  # noqa: F401
)

# ----------------- Modules -----------------

# ----------------- Functions -----------------
from tests.test_resources.test_modules.test_functions.conftest import (
    func_with_artifacts,  # noqa: F401
    shared_function,  # noqa: F401
    slow_func,  # noqa: F401
    slow_running_func,  # noqa: F401
    summer_func,  # noqa: F401
    summer_func_shared,  # noqa: F401
    summer_func_with_auth,  # noqa: F401
)

# ----------------- Tables -----------------

from tests.test_resources.test_modules.test_tables.conftest import (
    arrow_table,  # noqa: F401
    cudf_table,  # noqa: F401
    dask_table,  # noqa: F401
    huggingface_table,  # noqa: F401
    pandas_table,  # noqa: F401
    ray_table,  # noqa: F401
    table,  # noqa: F401
)


########## DEFAULT LEVELS ##########

default_fixtures = {}
default_fixtures[TestLevels.UNIT] = {
    "cluster": [
        local_docker_cluster_public_key_logged_in,
        local_docker_cluster_public_key_logged_out,
    ]
}
default_fixtures[TestLevels.LOCAL] = {
    "cluster": [
        local_docker_cluster_public_key_logged_in,
        local_docker_cluster_public_key_logged_out,
        local_docker_cluster_passwd,
    ]
}
default_fixtures[TestLevels.MINIMAL] = {"cluster": [ondemand_cpu_cluster]}
default_fixtures[TestLevels.THOROUGH] = {
    "cluster": [
        local_docker_cluster_passwd,
        local_docker_cluster_public_key_logged_in,
        local_docker_cluster_public_key_logged_out,
        ondemand_cpu_cluster,
        ondemand_https_cluster_with_auth,
        password_cluster,
        static_cpu_cluster,
    ]
}
default_fixtures[TestLevels.MAXIMAL] = {
    "cluster": [
        local_docker_cluster_passwd,
        local_docker_cluster_public_key_logged_in,
        local_docker_cluster_public_key_logged_out,
        local_docker_cluster_telemetry_public_key,
        ondemand_cpu_cluster,
        ondemand_https_cluster_with_auth,
        password_cluster,
        static_cpu_cluster,
    ]
}
