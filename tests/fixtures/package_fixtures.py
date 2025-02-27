import os
import shutil
import subprocess

import pytest

import runhouse as rh
from runhouse.resources.packages import InstallTarget

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


@pytest.fixture
def local_package(local_folder):
    args = {
        "install_target": InstallTarget(local_path=local_folder.path),
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


@pytest.fixture(scope="session")
def installed_editable_package(tmp_path_factory):
    tmp_package_dir = tmp_path_factory.mktemp("fake_package") / "test_fake_package"

    # Copy the test_fake_package directory that's in the same directory as this file, to the tmp_package_dir established
    # above.
    shutil.copytree(
        os.path.join(os.path.dirname(__file__), "test_fake_package"),
        tmp_package_dir,
    )

    # Run a pip install -e on the tmp_package_dir via subprocess.run, locally, not on the cluster
    subprocess.run(["pip", "install", "-e", str(tmp_package_dir)], check=True)

    yield

    # Uninstall the package after the test is done
    subprocess.run(["pip", "uninstall", "-y", "test_fake_package"], check=True)

    # Delete everything in tmp_package_dir recursively
    shutil.rmtree(tmp_package_dir)


@pytest.fixture(scope="session")
def installed_editable_package_copy(tmp_path_factory):
    tmp_package_dir = (
        tmp_path_factory.mktemp("fake_package_copy") / "test_fake_package_copy"
    )

    # Copy the test_fake_package directory that's in the same directory as this file, to the tmp_package_dir established
    # above.
    shutil.copytree(
        os.path.join(os.path.dirname(__file__), "test_fake_package_copy"),
        tmp_package_dir,
    )

    # Run a pip install -e on the tmp_package_dir via subprocess.run, locally, not on the cluster
    subprocess.run(["pip", "install", "-e", str(tmp_package_dir)], check=True)

    yield

    # Uninstall the package after the test is done
    subprocess.run(["pip", "uninstall", "-y", "test_fake_package_copy"], check=True)

    # Delete everything in tmp_package_dir recursively
    shutil.rmtree(tmp_package_dir)
