import unittest

import pytest
import requests


@pytest.mark.httpservertest
def test_cluster_is_up(local_docker_slim):
    cluster = local_docker_slim
    assert cluster.is_up()  # Should be true for a Cluster object


@pytest.mark.httpservertest
def test_spans_http_endpoint(self, local_docker_http_server_with_telemetry):
    cluster = local_docker_http_server_with_telemetry
    assert cluster.is_up()  # Should be true for a Cluster object

    # Make a GET request to the /spans endpoint
    response = requests.get("http://127.0.0.1:50052/spans")

    # Check the status code
    self.assertEqual(response.status_code, 200)

    # JSON parse the response
    parsed_response = response.json()
    print("Parsed response: ", parsed_response)

    # Assert that the key "spans" exists in the parsed response
    self.assertIn("spans", parsed_response)


if __name__ == "__main__":
    unittest.main()
