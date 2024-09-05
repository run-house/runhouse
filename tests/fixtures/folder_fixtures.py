import pytest

import runhouse as rh

from tests.conftest import init_args

from .utils import create_gcs_bucket, create_s3_bucket


@pytest.fixture
def dest(request):
    """Parametrize over multiple folders - useful for running the same test on multiple storage types."""
    return request.getfixturevalue(request.param)


@pytest.fixture
def folder(request):
    """Parametrize over multiple folders - useful for running the same test on multiple storage types."""
    return request.getfixturevalue(request.param)


@pytest.fixture
def local_folder():
    from runhouse import Folder

    args = {"path": Folder.DEFAULT_CACHE_FOLDER}
    local_folder = rh.folder(**args)
    init_args[id(local_folder)] = args
    local_folder.put(
        {f"sample_file_{i}.txt": f"file{i}".encode() for i in range(3)}, overwrite=True
    )
    return local_folder


@pytest.fixture
def docker_cluster_folder(docker_cluster_pk_ssh_no_auth):
    args = {
        "name": "test_docker_folder",
        "system": docker_cluster_pk_ssh_no_auth,
        "path": "rh-folder",
    }

    local_folder_docker = rh.folder(**args)
    init_args[id(local_folder_docker)] = args
    local_folder_docker.put(
        {f"sample_file_{i}.txt": f"file{i}" for i in range(3)}, overwrite=True
    )
    return local_folder_docker


@pytest.fixture
def cluster_folder(ondemand_aws_docker_cluster):
    args = {
        "name": "test_cluster_folder",
        "path": "rh-folder",
    }

    cluster_folder = rh.folder(**args).to(system=ondemand_aws_docker_cluster)
    init_args[id(cluster_folder)] = args
    cluster_folder.put(
        {f"sample_file_{i}.txt": f"file{i}".encode() for i in range(3)}, overwrite=True
    )
    return cluster_folder


@pytest.fixture
def s3_folder(local_folder):
    # create s3 folder and files
    tmp_folder = rh.folder(system="s3")
    tmp_folder._upload(src=local_folder.path, region="us-east-1")

    args = {
        "name": "test_s3_folder",
        "system": "s3",
        "path": tmp_folder.path,
    }

    s3_folder = rh.folder(**args)
    init_args[id(s3_folder)] = args

    yield s3_folder

    # Delete files from S3
    s3_folder.rm()


@pytest.fixture
def gcs_folder(local_folder):
    # create gcs folder and files
    gcs_path = local_folder.to(system="gs").path

    args = {
        "name": "test_gcs_folder",
        "system": "gs",
        "path": gcs_path,
    }

    gcs_folder = rh.folder(**args)
    init_args[id(gcs_folder)] = args

    yield gcs_folder

    # Delete files from GCS
    gcs_folder.rm()


# ----------------- S3 -----------------


@pytest.fixture(scope="session")
def runs_s3_bucket():
    runs_bucket = create_s3_bucket("runhouse-runs")
    return runs_bucket.name


@pytest.fixture(scope="session")
def blob_s3_bucket():
    blob_bucket = create_s3_bucket("runhouse-blob")
    return blob_bucket.name


@pytest.fixture(scope="session")
def table_s3_bucket():
    table_bucket = create_s3_bucket("runhouse-table")
    return table_bucket.name


# ----------------- GCP -----------------


@pytest.fixture(scope="session")
def blob_gcs_bucket():
    blob_bucket = create_gcs_bucket("runhouse-blob")
    return blob_bucket.name


@pytest.fixture(scope="session")
def table_gcs_bucket():
    table_bucket = create_gcs_bucket("runhouse-table")
    return table_bucket.name
