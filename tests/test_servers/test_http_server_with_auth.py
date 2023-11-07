import json
import os
import shutil
import subprocess
import unittest
from pathlib import Path

import pytest

import runhouse as rh
from ray.exceptions import RayTaskError

from runhouse.globals import rns_client
from runhouse.servers.http.http_utils import b64_unpickle, pickle_b64

from tests.conftest import test_account

from tests.test_servers.conftest import summer


@pytest.mark.usefixtures("docker_container")
class TestHTTPServerWithAuth:
    """Test the HTTP server with authentication enabled on a local docker container"""

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

    def test_put_resource(self, http_client, local_blob, base_cluster):
        state = None
        resource = local_blob.to(base_cluster)
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
        remote_func = rh.function(summer, system=base_cluster)

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
        remote_func = rh.function(summer, system=base_cluster)
        method = "call"

        response = await async_http_client.post(
            f"/call/{remote_func.name}/{method}?serialization=None",
            json={"args": [1, 2]},
            headers=rns_client.request_headers,
        )
        assert response.status_code == 200

    # -------- INVALID TOKEN / CLUSTER ACCESS TESTS ----------- #
    def test_request_with_no_cluster_config_yaml(self, http_client, base_cluster):
        base_cluster.run(
            ["mv ~/.rh/cluster_config.yaml ~/.rh/cluster_config_temp.yaml"]
        )
        response = http_client.get("/keys", headers=rns_client.request_headers)

        assert response.status_code == 404
        assert "Failed to load current cluster" in response.text

        base_cluster.run(
            ["mv ~/.rh/cluster_config_temp.yaml ~/.rh/cluster_config.yaml"]
        )
        response = http_client.get("/keys", headers=self.invalid_headers)

        assert response.status_code == 500

    def test_no_access_to_cluster(self, http_client, base_cluster):
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
        self, http_client, local_blob, base_cluster
    ):
        state = None
        resource = local_blob.to(base_cluster)
        data = pickle_b64((resource.config_for_rns, state, resource.dryrun))
        response = http_client.post(
            "/resource", json={"data": data}, headers=self.invalid_headers
        )
        assert response.status_code == 500

    def test_call_module_method_with_invalid_token(self, http_client, base_cluster):
        # Create new func on the cluster, then call it
        remote_func = rh.function(summer, system=base_cluster)

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


@pytest.fixture(autouse=True)
def setup_cluster_config():
    # Create a temporary directory that simulates the user's home directory
    home_dir = Path("~/.rh").expanduser()
    home_dir.mkdir(exist_ok=True)
    cluster_config_path = home_dir / "cluster_config.yaml"
    rns_address = "/kitchen_tester/local_cluster"
    cluster_config = {
        "name": rns_address,
        "resource_type": "cluster",
        "resource_subtype": "OnDemandCluster",
        "provenance": None,
        "server_port": 32300,
        "server_connection_type": "ssh",
        "den_auth": False,
        "ips": ["localhost"],
        "instance_type": None,
        "num_instances": None,
        "provider": "cheapest",
        "autostop_mins": -1,
        "open_ports": [],
        "use_spot": False,
        "image_id": None,
        "region": None,
    }

    c = rh.OnDemandCluster.from_name(rns_address)
    if not c:
        current_username = rh.configs.get("username")
        with test_account():
            c = rh.ondemand_cluster(
                name="local_cluster", server_host="localhost", den_auth=True
            ).save()
            c.share(current_username, access_type="write", notify_users=False)

    with open(cluster_config_path, "w") as file:
        json.dump(cluster_config, file)

    yield

    if cluster_config_path.exists():
        cluster_config_path.unlink()


class TestHTTPServerWithAuthLocally:
    invalid_headers = {"Authorization": "Bearer InvalidToken"}

    @staticmethod
    def assert_ray_task_error(exc_info):
        error_msg = str(exc_info.value)

        msg_1 = "Failed to load resources for user"
        msg_2 = "Invalid or expired token"

        assert msg_1 in error_msg
        assert msg_2 in error_msg

    def test_get_cert(self, local_client_with_den_auth):
        # Define the path for the temporary certificate
        certs_dir = Path.home() / "ssl" / "certs"
        certs_dir.mkdir(
            parents=True, exist_ok=True
        )  # Create the directory if it doesn't exist
        cert_path = certs_dir / "rh_server.crt"

        # Create a temporary certificate file
        cert_content = "dummy cert content"
        with open(cert_path, "w") as cert_file:
            cert_file.write(cert_content)

        try:
            # Perform the test
            response = local_client_with_den_auth.get("/cert")
            assert response.status_code == 200

            # Check if the error message is as expected (if the logic expects an error)
            error_b64 = response.json().get("error")
            if error_b64:
                error_message = b64_unpickle(error_b64)
                assert isinstance(error_message, FileNotFoundError)
                assert "No certificate found on cluster in path" in str(error_message)
            else:
                # If the logic is to test successful retrieval, decode the data
                data_b64 = response.json().get("data")
                cert_data = b64_unpickle(data_b64)
                assert cert_data == cert_content.encode()

        finally:
            cert_path.unlink(missing_ok=True)

    def test_check_server(self, local_client_with_den_auth):
        response = local_client_with_den_auth.get("/check")
        assert response.status_code == 200

    def test_put_resource(self, local_client_with_den_auth, local_blob):
        resource_path = Path("~/rh/blob/local-blob").expanduser()
        resource_dir = resource_path.parent
        state = None
        try:
            resource = local_blob.to(system="file", path=resource_path)
            data = pickle_b64((resource.config_for_rns, state, resource.dryrun))
            response = local_client_with_den_auth.post(
                "/resource", json={"data": data}, headers=rns_client.request_headers
            )
            assert response.status_code == 200
        finally:
            if os.path.exists(resource_path):
                shutil.rmtree(resource_dir)

    def test_put_object(self, local_client_with_den_auth):
        test_list = list(range(5, 50, 2)) + ["a string"]
        response = local_client_with_den_auth.post(
            "/object",
            json={"data": pickle_b64(test_list), "key": "key1"},
            headers=rns_client.request_headers,
        )
        assert response.status_code == 200

    def test_rename_object(self, local_client_with_den_auth):
        old_key = "key1"
        new_key = "key2"
        data = pickle_b64((old_key, new_key))
        response = local_client_with_den_auth.put(
            "/object", json={"data": data}, headers=rns_client.request_headers
        )
        assert response.status_code == 200

    def test_get_keys(self, local_client_with_den_auth):
        response = local_client_with_den_auth.get(
            "/keys", headers=rns_client.request_headers
        )
        assert response.status_code == 200
        assert "key2" in b64_unpickle(response.json().get("data"))

    def test_delete_obj(self, local_client_with_den_auth):
        # https://www.python-httpx.org/compatibility/#request-body-on-http-methods
        keys = ["key2"]
        data = pickle_b64(keys)
        response = local_client_with_den_auth.request(
            "delete",
            url="/object",
            json={"data": data},
            headers=rns_client.request_headers,
        )
        assert response.status_code == 200

    def test_add_secrets(self, local_client_with_den_auth):
        secrets = {"aws": "abc123"}
        data = pickle_b64(secrets)
        response = local_client_with_den_auth.post(
            "/secrets", json={"data": data}, headers=rns_client.request_headers
        )
        assert response.status_code == 200

    @unittest.skip("Not implemented yet.")
    def test_call_module_method(self, local_client_with_den_auth):
        # Create new func on the cluster, then call it
        remote_func = rh.function(summer).to("here")

        method_name = "call"
        module_name = remote_func.name
        args = (1, 2)
        kwargs = {"force": False}

        response = local_client_with_den_auth.post(
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

    @unittest.skip("Not implemented yet.")
    @pytest.mark.asyncio
    async def test_async_call(self, async_http_client):
        remote_func = rh.function(summer).to("here")
        method = "call"

        response = await async_http_client.post(
            f"/call/{remote_func.name}/{method}?serialization=None",
            json={"args": [1, 2]},
            headers=rns_client.request_headers,
        )
        assert response.status_code == 200

    # -------- INVALID TOKEN / CLUSTER ACCESS TESTS ----------- #
    def test_request_with_no_cluster_config_yaml(self, local_client_with_den_auth):
        source_path = os.path.expanduser("~/.rh/cluster_config.yaml")
        destination_path = os.path.expanduser("~/.rh/cluster_config_temp.yaml")

        # Use the expanded paths in the command
        subprocess.run(["mv", source_path, destination_path])
        response = local_client_with_den_auth.get(
            "/keys", headers=rns_client.request_headers
        )

        assert response.status_code == 404
        assert "Failed to load current cluster" in response.text

        subprocess.run(["mv", destination_path, source_path])

        with pytest.raises(RayTaskError) as exc_info:
            local_client_with_den_auth.get("/keys", headers=self.invalid_headers)

        self.assert_ray_task_error(exc_info)

    def test_no_access_to_cluster(self, local_client_with_den_auth):
        with test_account():
            response = local_client_with_den_auth.get(
                "/keys", headers=rns_client.request_headers
            )

            assert response.status_code == 404
            assert "Cluster access is required for API" in response.text

    def test_request_with_no_token(self, local_client_with_den_auth):
        response = local_client_with_den_auth.get("/keys")  # No headers are passed
        assert response.status_code == 404

        assert "No token found in request auth headers" in response.text

    def test_get_cert_with_invalid_token(self, local_client_with_den_auth):
        response = local_client_with_den_auth.get("/cert", headers=self.invalid_headers)
        # Should be able to download the cert even without a valid token
        assert response.status_code == 200

    def test_check_server_with_invalid_token(self, local_client_with_den_auth):
        response = local_client_with_den_auth.get(
            "/check", headers=self.invalid_headers
        )
        # Should be able to ping the server even without a valid token
        assert response.status_code == 200

    def test_put_resource_with_invalid_token(
        self, local_client_with_den_auth, local_blob
    ):
        resource_path = Path("~/rh/blob/local-blob").expanduser()
        resource_dir = resource_path.parent
        state = None
        try:
            resource = local_blob.to(system="file", path=resource_path)
            data = pickle_b64((resource.config_for_rns, state, resource.dryrun))
            with pytest.raises(RayTaskError) as exc_info:
                local_client_with_den_auth.post(
                    "/resource", json={"data": data}, headers=self.invalid_headers
                )

            self.assert_ray_task_error(exc_info)

        finally:
            if os.path.exists(resource_path):
                shutil.rmtree(resource_dir)

    @unittest.skip("Not implemented yet.")
    def test_call_module_method_with_invalid_token(self, local_client_with_den_auth):
        # Create new func on the cluster, then call it
        remote_func = rh.function(summer).to("here")

        method_name = "call"
        module_name = remote_func.name
        args = (1, 2)
        kwargs = {"force": False}

        with pytest.raises(RayTaskError) as exc_info:
            local_client_with_den_auth.post(
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
        self.assert_ray_task_error(exc_info)

    def test_put_object_with_invalid_token(self, local_client_with_den_auth):
        test_list = list(range(5, 50, 2)) + ["a string"]
        with pytest.raises(RayTaskError) as exc_info:
            local_client_with_den_auth.post(
                "/object",
                json={"data": pickle_b64(test_list), "key": "key1"},
                headers=self.invalid_headers,
            )
        self.assert_ray_task_error(exc_info)

    def test_rename_object_with_invalid_token(self, local_client_with_den_auth):
        old_key = "key1"
        new_key = "key2"
        data = pickle_b64((old_key, new_key))
        with pytest.raises(RayTaskError) as exc_info:
            local_client_with_den_auth.put(
                "/object", json={"data": data}, headers=self.invalid_headers
            )
        self.assert_ray_task_error(exc_info)

    def test_get_keys_with_invalid_token(self, local_client_with_den_auth):
        with pytest.raises(RayTaskError) as exc_info:
            local_client_with_den_auth.get("/keys", headers=self.invalid_headers)

        self.assert_ray_task_error(exc_info)

    def test_add_secrets_with_invalid_token(self, local_client_with_den_auth):
        secrets = {"aws": "abc123"}
        data = pickle_b64(secrets)
        with pytest.raises(RayTaskError) as exc_info:
            local_client_with_den_auth.post(
                "/secrets", json={"data": data}, headers=self.invalid_headers
            )

        self.assert_ray_task_error(exc_info)


if __name__ == "__main__":
    unittest.main()
