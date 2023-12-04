import pytest

import runhouse as rh

from tests.conftest import init_args

from .utils import create_gcs_bucket, create_s3_bucket


@pytest.fixture
def folder(request):
    """Parametrize over multiple folders - useful for running the same test on multiple storage types."""
    return request.getfixturevalue(request.param)


@pytest.fixture
def local_folder(tmp_path):
    args = {"path": tmp_path / "tests_tmp"}
    local_folder = rh.folder(**args)
    init_args[id(local_folder)] = args
    local_folder.put({f"sample_file_{i}.txt": f"file{i}".encode() for i in range(3)})
    return local_folder


@pytest.fixture
def cluster_folder(ondemand_cpu_cluster, local_folder):
    return local_folder.to(system=ondemand_cpu_cluster)


@pytest.fixture
def s3_folder(local_folder):
    s3_folder = local_folder.to(system="s3")
    yield s3_folder

    # Delete files from S3
    s3_folder.rm()


@pytest.fixture
def gcs_folder(local_folder):
    gcs_folder = local_folder.to(system="gs")
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
