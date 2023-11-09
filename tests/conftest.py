import contextlib
import enum
import os

import dotenv

import pytest

import runhouse as rh


class TestLevels(str, enum.Enum):
    UNIT = "unit"
    LOCAL = "local"
    MINIMAL = "minimal"
    THOROUGH = "thorough"
    MAXIMAL = "maximal"


DEFAULT_LEVEL = TestLevels.UNIT


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


def pytest_generate_tests(metafunc):
    level = metafunc.config.getoption("level")
    level_fixtures = getattr(
        metafunc.cls or metafunc.module, level.upper(), default_fixtures[level]
    )
    for fixture_name, fixture_list in level_fixtures.items():
        if fixture_name in metafunc.fixturenames:
            metafunc.parametrize(fixture_name, fixture_list, indirect=True)


def pytest_configure():
    pytest.init_args = {}


init_args = {}


# https://docs.pytest.org/en/6.2.x/fixture.html#conftest-py-sharing-fixtures-across-multiple-files

############## HELPERS ##############


@pytest.fixture(scope="session")
@contextlib.contextmanager
def test_account():
    """Used for the purposes of testing resource sharing among different accounts.
    When inside the context manager, use the test account credentials before reverting back to the original
    account when exiting."""
    dotenv.load_dotenv()

    test_token = os.getenv("TEST_TOKEN")
    test_username = os.getenv("TEST_USERNAME")
    assert test_token and test_username

    current_token = rh.configs.get("token")
    current_username = rh.configs.get("username")

    try:
        # Assume the role of the test account when inside the context manager
        test_account_token = test_token
        test_account_username = test_username
        test_account_folder = f"/{test_account_username}"

        # Hack to avoid actually writing down these values, in case the user stops mid-test and we don't reach the
        # finally block
        rh.configs.defaults_cache["token"] = test_account_token
        rh.configs.defaults_cache["username"] = test_account_username
        rh.configs.defaults_cache["default_folder"] = test_account_folder

        yield {
            "test_token": test_account_token,
            "test_username": test_account_username,
            "test_folder": test_account_folder,
        }

    finally:
        # Reset configs back to original account
        rh.configs.defaults_cache["token"] = current_token
        rh.configs.defaults_cache["username"] = current_username
        rh.configs.defaults_cache["default_folder"] = f"/{current_username}"


# ----------------- Clusters -----------------

from tests.test_resources.test_clusters.conftest import (
    build_and_run_image,  # noqa: F401
    byo_cpu,  # noqa: F401
    cluster,  # noqa: F401
    local_docker_cluster_passwd,  # noqa: F401
    local_docker_cluster_public_key,  # noqa: F401
    local_logged_out_docker_cluster,  # noqa: F401
    local_test_account_cluster_public_key,  # noqa: F401
    named_cluster,  # noqa: F401
    password_cluster,  # noqa: F401
    shared_cluster,  # noqa: F401
    static_cpu_cluster,  # noqa: F401
    unnamed_cluster,  # noqa: F401
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
    env,  # noqa: F401
    test_env,  # noqa: F401
)

# ----------------- Blobs -----------------

from tests.test_resources.test_modules.test_blobs.conftest import (
    blob,  # noqa: F401
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
default_fixtures[TestLevels.UNIT] = {"cluster": [local_docker_cluster_public_key]}
default_fixtures[TestLevels.LOCAL] = {
    "cluster": [local_docker_cluster_passwd, local_docker_cluster_public_key]
}
default_fixtures[TestLevels.MINIMAL] = {"cluster": [ondemand_cpu_cluster]}
default_fixtures[TestLevels.THOROUGH] = {
    "cluster": [
        local_docker_cluster_passwd,
        local_docker_cluster_public_key,
        ondemand_cpu_cluster,
        ondemand_https_cluster_with_auth,
        password_cluster,
        byo_cpu,
    ]
}
default_fixtures[TestLevels.MAXIMAL] = {
    "cluster": [
        local_docker_cluster_passwd,
        local_docker_cluster_public_key,
        ondemand_cpu_cluster,
        ondemand_https_cluster_with_auth,
        password_cluster,
        byo_cpu,
    ]
}
