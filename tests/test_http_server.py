import unittest

import pytest


@pytest.mark.dockertest
@pytest.mark.httpservertest
def test_password_cluster_is_up(local_docker_cluster_passwd):
    cluster = local_docker_cluster_passwd
    cluster.check_server()
    assert cluster.is_up()  # Should be true for a Cluster object


@pytest.mark.dockertest
@pytest.mark.httpservertest
def test_public_key_cluster_is_up(local_docker_cluster_public_key):
    cluster = local_docker_cluster_public_key
    cluster.check_server()
    assert cluster.is_up()  # Should be true for a Cluster object


if __name__ == "__main__":
    unittest.main()
