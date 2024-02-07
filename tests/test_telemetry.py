import pytest
import requests


@pytest.mark.level("local")
def test_public_key_cluster_has_telemetry(docker_cluster_pk_ssh_telemetry):
    cluster = docker_cluster_pk_ssh_telemetry
    cluster.check_server()
    assert cluster.is_up()  # Should be true for a Cluster object

    # Make a GET request to the /spans endpoint
    response = requests.get(f"http://{cluster.address}:{cluster.server_port}/spans")

    # Check the status code
    assert response.status_code == 200

    # JSON parse the response
    parsed_response = response.json()

    # Assert that the key "spans" exists in the parsed response
    assert "spans" in parsed_response, "'spans' not in response"
