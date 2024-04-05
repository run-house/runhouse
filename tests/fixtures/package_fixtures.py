import pytest

import runhouse as rh

from tests.conftest import init_args


@pytest.fixture
def package(request):
    return request.getfixturevalue(request.param)


@pytest.fixture(scope="session")
def pip_package():
    args = {
        "install_target": "numpy",
        "install_method": "pip",
    }
    package = rh.Package(**args)
    init_args[id(package)] = args
    return package


@pytest.fixture(scope="session")
def conda_package():
    args = {
        "install_target": "pandas",
        "install_method": "conda",
    }
    package = rh.Package(**args)
    init_args[id(package)] = args
    return package


@pytest.fixture(scope="session")
def reqs_package():
    args = {
        "install_target": rh.folder(path="."),
        "install_method": "reqs",
    }
    package = rh.Package(**args)
    init_args[id(package)] = args
    return package


@pytest.fixture
def local_package(local_folder):
    args = {
        "install_target": local_folder,
        "install_method": "local",
    }
    package = rh.Package(**args)
    init_args[id(package)] = args
    return package


@pytest.fixture
def s3_package(s3_folder):
    args = {
        "install_target": s3_folder,
        "install_method": "local",
    }
    package = rh.Package(**args)
    init_args[id(package)] = args
    return package


@pytest.fixture
def git_package():
    args = {
        "install_target": "./transformers",
        "install_method": "pip",
        "git_url": "https://github.com/huggingface/transformers.git",
        "revision": "v4.39.2",
    }
    package = rh.GitPackage(**args)
    init_args[id(package)] = args
    return package
