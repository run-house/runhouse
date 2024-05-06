import pytest
import requests

import runhouse as rh


@pytest.mark.usefixtures("cluster")
class TestTelemetry:
    LOCAL = {
        "cluster": [
            "docker_cluster_pk_ssh_den_auth",
            "docker_cluster_pk_ssh_no_auth",
            "docker_cluster_pk_http_exposed",
            "docker_cluster_pk_tls_den_auth",
        ]
    }

    @pytest.mark.level("local")
    def test_public_key_cluster_has_telemetry(self, cluster):
        cluster.check_server()
        assert cluster.is_up()  # Should be true for a Cluster object

        # Make a GET request to the /spans endpoint
        url = f"{cluster.endpoint()}/spans"
        response = requests.get(
            url,
            verify=False,
            headers=rh.globals.rns_client.request_headers(cluster.rns_address),
        )

        if cluster.use_local_telemetry or not rh.configs.get("disable_data_collection"):
            # Check the status code
            assert response.status_code == 200

            # JSON parse the response
            parsed_response = response.json()

            # Assert that the key "spans" exists in the parsed response
            assert "spans" in parsed_response, "'spans' not in response"
        else:
            # Check the status code
            assert response.status_code == 404
            assert response.text == '{"detail":"Not Found"}'
