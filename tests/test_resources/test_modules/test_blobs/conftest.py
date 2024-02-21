import numpy as np
import pytest

import runhouse as rh

from tests.conftest import init_args


@pytest.fixture
def blob(request):
    """Parametrize over multiple blobs - useful for running the same test on multiple storage types."""
    return request.getfixturevalue(request.param)


@pytest.fixture
def file(request):
    """Parametrize over multiple files - useful for running the same test on multiple storage types."""
    return request.getfixturevalue(request.param)


@pytest.fixture(scope="session")
def blob_data():
    return [np.arange(50), "test", {"a": 1, "b": 2}]


@pytest.fixture
def local_file(blob_data, tmp_path):
    args = {
        "data": blob_data,
        "system": "file",
        "path": str(tmp_path / "test_blob.pickle"),
    }
    b = rh.blob(**args)
    init_args[id(b)] = args
    return b


@pytest.fixture
def local_blob(blob_data):
    return rh.blob(
        data=blob_data,
    )


@pytest.fixture
def s3_blob(blob_data, blob_s3_bucket):
    return rh.blob(
        data=blob_data,
        system="s3",
        path=f"/{blob_s3_bucket}/test_blob.pickle",
    )


@pytest.fixture
def gcs_blob(blob_data, blob_gcs_bucket):
    return rh.blob(
        data=blob_data,
        system="gs",
        path=f"/{blob_gcs_bucket}/test_blob.pickle",
    )


@pytest.fixture
def cluster_blob(blob_data, ondemand_aws_cluster):
    return rh.blob(
        data=blob_data,
        system=ondemand_aws_cluster,
    )


@pytest.fixture
def cluster_file(blob_data, ondemand_aws_cluster):
    return rh.blob(
        data=blob_data,
        system=ondemand_aws_cluster,
        path="test_blob.pickle",
    )
