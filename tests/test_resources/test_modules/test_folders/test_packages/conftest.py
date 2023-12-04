import pytest

import runhouse as rh

from tests.conftest import init_args


@pytest.fixture(scope="session")
def package(request):
    """Parametrize over multiple packages - useful for running the same test on multiple storage types."""
    return request.getfixturevalue(request.param)


@pytest.fixture
def local_package(local_folder):
    args = {"path": local_folder.path, "install_method": "local"}
    p = rh.package(**args)
    init_args[id(p)] = args
    return p


@pytest.fixture
def s3_package(s3_folder):
    return rh.package(
        path=s3_folder.path, system=s3_folder.system, install_method="local"
    )
