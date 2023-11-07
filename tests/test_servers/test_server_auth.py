import unittest

import pytest

import runhouse as rh

from runhouse.globals import rns_client
from runhouse.servers.http.http_utils import b64_unpickle, pickle_b64

from tests.conftest import test_account

from tests.test_servers.conftest import summer


@pytest.mark.usefixtures("docker_container_with_auth")
class TestHTTPServerWithAuth:
    invalid_headers = {"Authorization": "Bearer InvalidToken"}

    def test_get_cert(self, http_client):
        response = http_client.get("/cert")
        assert response.status_code == 200

        error_b64 = response.json().get("error")
        error_message = b64_unpickle(error_b64)

        assert isinstance(error_message, FileNotFoundError)
        assert "No certificate found on cluster in path" in str(error_message)

    def test_check_server(self, http_client):
        response = http_client.get("/check")
        assert response.status_code == 200

    def test_put_resource(self, http_client, local_blob, cluster_with_auth):
        state = None
        resource = local_blob.to(cluster_with_auth)
        data = pickle_b64((resource.config_for_rns, state, resource.dryrun))
        response = http_client.post(
            "/resource", json={"data": data}, headers=rns_client.request_headers
        )
        assert response.status_code == 200

    def test_put_object(self, http_client):
        test_list = list(range(5, 50, 2)) + ["a string"]
        response = http_client.post(
            "/object",
            json={"data": pickle_b64(test_list), "key": "key1"},
            headers=rns_client.request_headers,
        )
        assert response.status_code == 200

    def test_rename_object(self, http_client):
        old_key = "key1"
        new_key = "key2"
        data = pickle_b64((old_key, new_key))
        response = http_client.put(
            "/object", json={"data": data}, headers=rns_client.request_headers
        )
        assert response.status_code == 200

    def test_get_keys(self, http_client):
        response = http_client.get("/keys", headers=rns_client.request_headers)
        assert response.status_code == 200
        assert "key2" in b64_unpickle(response.json().get("data"))

    def test_delete_obj(self, http_client):
        # https://www.python-httpx.org/compatibility/#request-body-on-http-methods
        keys = ["key2"]
        data = pickle_b64(keys)
        response = http_client.request(
            "delete",
            url="/object",
            json={"data": data},
            headers=rns_client.request_headers,
        )
        assert response.status_code == 200

    def test_add_secrets(self, http_client):
        secrets = {"aws": "abc123"}
        data = pickle_b64(secrets)
        response = http_client.post(
            "/secrets", json={"data": data}, headers=rns_client.request_headers
        )
        assert response.status_code == 200

    def test_call_module_method(self, http_client, base_cluster):
        # Create new func on the cluster, then call it
        remote_func = rh.function(summer).to(base_cluster)

        method_name = "call"
        module_name = remote_func.name
        args = (1, 2)
        kwargs = {"force": False}

        response = http_client.post(
            f"{module_name}/{method_name}",
            json={
                "data": pickle_b64([args, kwargs]),
                "env": None,
                "stream_logs": True,
                "save": False,
                "key": None,
                "remote": False,
                "run_async": False,
            },
            headers=rns_client.request_headers,
        )
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_async_call(self, async_http_client, base_cluster):
        remote_func = rh.function(summer).to(base_cluster)
        method = "call"

        response = await async_http_client.post(
            f"/call/{remote_func.name}/{method}?serialization=None",
            json={"args": [1, 2]},
            headers=rns_client.request_headers,
        )
        assert response.status_code == 200

    # -------- INVALID TOKEN / CLUSTER ACCESS TESTS ----------- #
    def test_request_with_no_cluster_config_yaml(self, http_client, cluster_with_auth):
        cluster_with_auth.run(
            ["mv ~/.rh/cluster_config.yaml ~/.rh/cluster_config_temp.yaml"]
        )
        response = http_client.get("/keys", headers=rns_client.request_headers)

        assert response.status_code == 404
        assert "Failed to load current cluster" in response.text

        cluster_with_auth.run(
            ["mv ~/.rh/cluster_config_temp.yaml ~/.rh/cluster_config.yaml"]
        )
        response = http_client.get("/keys", headers=self.invalid_headers)

        assert response.status_code == 500

    def test_no_access_to_cluster(self, http_client, cluster_with_auth):
        with test_account():
            response = http_client.get("/keys", headers=rns_client.request_headers)

            assert response.status_code == 404
            assert "Cluster access is required for API" in response.text

    def test_request_with_no_token(self, http_client):
        response = http_client.get("/keys")  # No headers are passed
        assert response.status_code == 404

        assert "No token found in request auth headers" in response.text

    def test_get_cert_with_invalid_token(self, http_client):
        response = http_client.get("/cert", headers=self.invalid_headers)
        # Should be able to download the cert even without a valid token
        assert response.status_code == 200

    def test_check_server_with_invalid_token(self, http_client):
        response = http_client.get("/check", headers=self.invalid_headers)
        # Should be able to ping the server even without a valid token
        assert response.status_code == 200

    def test_put_resource_with_invalid_token(
        self, http_client, local_blob, cluster_with_auth
    ):
        state = None
        resource = local_blob.to(cluster_with_auth)
        data = pickle_b64((resource.config_for_rns, state, resource.dryrun))
        response = http_client.post(
            "/resource", json={"data": data}, headers=self.invalid_headers
        )
        assert response.status_code == 500

    def test_call_module_method_with_invalid_token(self, http_client, base_cluster):
        # Create new func on the cluster, then call it
        remote_func = rh.function(summer).to(base_cluster)

        method_name = "call"
        module_name = remote_func.name
        args = (1, 2)
        kwargs = {"force": False}

        response = http_client.post(
            f"{module_name}/{method_name}",
            json={
                "data": pickle_b64([args, kwargs]),
                "env": None,
                "stream_logs": True,
                "save": False,
                "key": None,
                "remote": False,
                "run_async": False,
            },
            headers=self.invalid_headers,
        )
        assert response.status_code == 500

    def test_put_object_with_invalid_token(self, http_client):
        test_list = list(range(5, 50, 2)) + ["a string"]
        response = http_client.post(
            "/object",
            json={"data": pickle_b64(test_list), "key": "key1"},
            headers=self.invalid_headers,
        )
        assert response.status_code == 500

    def test_rename_object_with_invalid_token(self, http_client):
        old_key = "key1"
        new_key = "key2"
        data = pickle_b64((old_key, new_key))
        response = http_client.put(
            "/object", json={"data": data}, headers=self.invalid_headers
        )
        assert response.status_code == 500

    def test_get_keys_with_invalid_token(self, http_client):
        response = http_client.get("/keys", headers=self.invalid_headers)
        assert response.status_code == 500

    def test_add_secrets_with_invalid_token(self, http_client):
        secrets = {"aws": "abc123"}
        data = pickle_b64(secrets)
        response = http_client.post(
            "/secrets", json={"data": data}, headers=self.invalid_headers
        )
        assert response.status_code == 500


if __name__ == "__main__":
    unittest.main()
