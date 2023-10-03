import unittest

import pytest

from .conftest import local_docker_http_server


@pytest.mark.httpservertest
def test_cluster_is_up():
    cluster = local_docker_http_server
    cluster.check_server()
    assert cluster.is_up()


if __name__ == "__main__":
    unittest.main()
