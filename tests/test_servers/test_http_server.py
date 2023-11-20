import json
import tempfile
import unittest

from pathlib import Path

import pytest

import runhouse as rh
from runhouse.servers.http.http_utils import b64_unpickle, pickle_b64

from tests.test_resources.test_clusters.conftest import (
    local_docker_cluster_public_key_logged_in,
)

from tests.test_servers.conftest import summer


@pytest.mark.usefixtures("cluster")
class TestHTTPServer:
    """Start HTTP server in a docker container running locally"""

    UNIT = {"cluster": [local_docker_cluster_public_key_logged_in]}
    LOCAL = {"cluster": [local_docker_cluster_public_key_logged_in]}
    # TODO add local_docker_cluster_passwd into LOCAL too?

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

    def test_put_resource(self, http_client, cluster, local_blob):
        state = None
        resource = local_blob.to(cluster)
        data = pickle_b64((resource.config_for_rns, state, resource.dryrun))
        response = http_client.post("/resource", json={"data": data})
        assert response.status_code == 200

    def test_put_object_and_get_keys(self, http_client):
        key = "key1"
        test_list = list(range(5, 50, 2)) + ["a string"]
        response = http_client.post(
            "/object", json={"data": pickle_b64(test_list), "key": key}
        )
        assert response.status_code == 200

        response = http_client.get("/keys")
        assert response.status_code == 200
        assert key in b64_unpickle(response.json().get("data"))

    def test_rename_object(self, http_client):
        old_key = "key1"
        new_key = "key2"
        data = pickle_b64((old_key, new_key))
        response = http_client.put("/object", json={"data": data})
        assert response.status_code == 200

        response = http_client.get("/keys")
        assert new_key in b64_unpickle(response.json().get("data"))

    def test_delete_obj(self, http_client):
        # https://www.python-httpx.org/compatibility/#request-body-on-http-methods
        key = "key2"
        data = pickle_b64([key])
        response = http_client.request("delete", url="/object", json={"data": data})
        assert response.status_code == 200

        response = http_client.get("/keys")
        assert key not in b64_unpickle(response.json().get("data"))

    def test_add_secrets(self, http_client):
        secrets = {"aws": {"access_key": "abc123", "secret_key": "abc123"}}
        data = pickle_b64(secrets)
        response = http_client.post("/secrets", json={"data": data})

        assert response.status_code == 200
        assert not b64_unpickle(response.json().get("data"))

    def test_add_secrets_for_unsupported_provider(self, local_client):
        secrets = {"test_provider": {"access_key": "abc123"}}
        data = pickle_b64(secrets)
        response = local_client.post("/secrets", json={"data": data})
        assert response.status_code == 200

        resp_data = b64_unpickle(response.json().get("data"))
        assert isinstance(resp_data, dict)
        assert "test_provider is not a Runhouse builtin provider" in resp_data.values()

    def test_call_module_method(self, http_client, cluster):
        # Send func to the cluster, then call it
        remote_func = rh.function(summer, system=cluster)

        method_name = "call"
        module_name = remote_func.name
        args = (1, 2)
        kwargs = {}

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
        )
        assert response.status_code == 200

        resp_obj: dict = json.loads(response.text.split("\n")[0])
        if resp_obj["output_type"] == "stdout":
            assert resp_obj["data"] == [
                "base_env servlet: Calling method call on module summer\n"
            ]

        if resp_obj["output_type"] == "result":
            assert b64_unpickle(resp_obj["data"]) == 3

    @pytest.mark.asyncio
    async def test_async_call(self, async_http_client, cluster):
        remote_func = rh.function(summer, system=cluster)
        method = "call"

        response = await async_http_client.post(
            f"/call/{remote_func.name}/{method}?serialization=None",
            json={"args": [1, 2]},
        )
        assert response.status_code == 200
        assert response.text == "3"

    @pytest.mark.asyncio
    async def test_async_call_with_invalid_serialization(
        self, async_http_client, cluster
    ):
        remote_func = rh.function(summer, system=cluster)
        method = "call"

        response = await async_http_client.post(
            f"/call/{remote_func.name}/{method}?serialization=random",
            json={"args": [1, 2]},
        )
        assert response.status_code == 500

    @pytest.mark.asyncio
    async def test_async_call_with_pickle_serialization(
        self, async_http_client, cluster
    ):
        remote_func = rh.function(summer, system=cluster)
        method = "call"

        response = await async_http_client.post(
            f"/call/{remote_func.name}/{method}?serialization=pickle",
            json={"args": [1, 2]},
        )

        assert response.status_code == 200
        assert b64_unpickle(response.text) == 3

    @pytest.mark.asyncio
    async def test_async_call_with_json_serialization(self, async_http_client, cluster):
        remote_func = rh.function(summer, system=cluster)
        method = "call"

        response = await async_http_client.post(
            f"/call/{remote_func.name}/{method}?serialization=json",
            json={"args": [1, 2]},
        )
        assert response.status_code == 200
        assert json.loads(response.text) == "3"


class TestHTTPServerLocally:
    """Start HTTP server locally, without docker"""

    def test_get_cert(self, local_client):
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
            response = local_client.get("/cert")
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

    def test_check_server(self, local_client):
        response = local_client.get("/check")
        assert response.status_code == 200

    def test_put_resource(self, local_client, blob_data):
        with tempfile.TemporaryDirectory() as temp_dir:
            resource_path = Path(temp_dir, "local-blob")
            local_blob = rh.blob(blob_data, path=resource_path)
            resource = local_blob.to(system="file", path=resource_path)

            state = None
            data = pickle_b64((resource.config_for_rns, state, resource.dryrun))
            response = local_client.post("/resource", json={"data": data})
            assert response.status_code == 200

    def test_put_object(self, local_client):
        key = "key1"
        test_list = list(range(5, 50, 2)) + ["a string"]
        response = local_client.post(
            "/object", json={"data": pickle_b64(test_list), "key": key}
        )
        assert response.status_code == 200

        response = local_client.get("/keys")
        assert key in b64_unpickle(response.json().get("data"))

    def test_rename_object(self, local_client):
        old_key = "key1"
        new_key = "key2"
        data = pickle_b64((old_key, new_key))
        response = local_client.put("/object", json={"data": data})
        assert response.status_code == 200

        response = local_client.get("/keys")
        assert new_key in b64_unpickle(response.json().get("data"))

    def test_get_keys(self, local_client):
        response = local_client.get("/keys")
        assert response.status_code == 200
        keys = b64_unpickle(response.json().get("data"))
        assert isinstance(keys, list)
        assert len(keys) > 0

    def test_delete_obj(self, local_client):
        # https://www.python-httpx.org/compatibility/#request-body-on-http-methods
        key = "key"
        data = pickle_b64([key])
        response = local_client.request("delete", url="/object", json={"data": data})
        assert response.status_code == 200

        response = local_client.get("/keys")
        assert key not in b64_unpickle(response.json().get("data"))

    def test_add_secrets_for_unsupported_provider(self, local_client):
        secrets = {"test_provider": {"access_key": "abc123"}}
        data = pickle_b64(secrets)
        response = local_client.post("/secrets", json={"data": data})
        assert response.status_code == 200

        resp_data = b64_unpickle(response.json().get("data"))
        assert isinstance(resp_data, dict)
        assert "test_provider is not a Runhouse builtin provider" in resp_data.values()

    # TODO [JL] - Need a local cluster object?
    @pytest.mark.skip(reason="Not implemented yet")
    def test_call_module_method(self, local_client, cluster):
        # Create new func on the cluster, then call it
        remote_func = rh.function(summer, system=cluster)

        method_name = "call"
        module_name = remote_func.name
        args = (1, 2)
        kwargs = {}

        response = local_client.post(
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
        )
        assert response.status_code == 200

    # TODO [JL] - Need a local cluster object?
    @pytest.mark.skip(reason="Not implemented yet")
    @pytest.mark.asyncio
    async def test_async_call(self, local_client, cluster):
        remote_func = rh.function(summer, system=cluster)
        method = "call"

        response = await local_client.post(
            f"/call/{remote_func.name}/{method}?serialization=None",
            json={"args": [1, 2]},
        )
        assert response.status_code == 200


if __name__ == "__main__":
    unittest.main()
