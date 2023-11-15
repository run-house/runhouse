import unittest

import pytest
import requests
import runhouse as rh


@pytest.mark.dockertest
@pytest.mark.telemetrytest
def test_public_key_cluster_has_telemetry(local_docker_cluster_telemetry_public_key):
    rh.globals.configs.defaults_cache["use_local_telemetry"] = True
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


if __name__ == "__main__":
    unittest.main()
