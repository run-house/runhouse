import unittest

import pytest
import requests


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


@pytest.mark.dockertest
@pytest.mark.httpservertest
def test_public_key_cluster_has_telemetry(local_docker_cluster_telemetry_public_key):
    cluster = local_docker_cluster_telemetry_public_key
    cluster.check_server()
    assert cluster.is_up()  # Should be true for a Cluster object

    # Make a GET request to the /spans endpoint
    response = requests.get("http://127.0.0.1:32300/spans")

    # Check the status code
    assert response.status_code == 200

    # JSON parse the response
    parsed_response = response.json()

    # Assert that the key "spans" exists in the parsed response
    assert "spans" in parsed_response, "'spans' not in response"


@pytest.mark.dockertest
@pytest.mark.httpservertest
def test_public_key_cluster_has_telemetry_with_tls(
    local_docker_cluster_telemetry_public_key_with_tls,
):
    cluster = local_docker_cluster_telemetry_public_key_with_tls
    cluster.check_server()
    assert cluster.is_up()  # Should be true for a Cluster object

    # Make a GET request to the /spans endpoint
    # (ex: for local docker with nginx: http://localhost:443/spans)
    response = requests.get(f"http://{cluster.address}:{cluster.server_port}/spans")

    # Check the status code
    assert response.status_code == 200

    # JSON parse the response
    parsed_response = response.json()

    # Assert that the key "spans" exists in the parsed response
    assert "spans" in parsed_response, "'spans' not in response"


if __name__ == "__main__":
    unittest.main()
