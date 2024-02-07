import json
import tempfile
from pathlib import Path

import pytest

import runhouse as rh

from runhouse.globals import rns_client
from runhouse.servers.http.http_utils import (
    b64_unpickle,
    DeleteObjectParams,
    pickle_b64,
    PutObjectParams,
    PutResourceParams,
    RenameObjectParams,
)

from tests.utils import friend_account

INVALID_HEADERS = {"Authorization": "Bearer InvalidToken"}

# Helper used for testing rh.Function
def summer(a, b):
    return a + b


@pytest.mark.servertest
@pytest.mark.den_auth
@pytest.mark.usefixtures("cluster")
class TestHTTPServerDocker:
    """
    Test the HTTP server running on a local Docker container.

    These will be tested on a container with Den Auth enabled, and a container without.
    """

    UNIT = {
        "cluster": [
            "docker_cluster_pk_ssh_den_auth",
            "docker_cluster_pk_ssh_no_auth",
        ]
    }
    LOCAL = {
        "cluster": [
            "docker_cluster_pk_ssh_den_auth",
            "docker_cluster_pk_ssh_no_auth",
        ]
    }

    @pytest.mark.level("local")
    def test_get_cert(self, http_client):
        response = http_client.get("/cert")
        assert response.status_code == 200

    @pytest.mark.level("local")
    def test_check_server(self, http_client):
        response = http_client.get("/check")
        assert response.status_code == 200

    @pytest.mark.level("local")
    def test_put_resource(self, http_client, blob_data, cluster):
        state = None
        resource = rh.blob(data=blob_data, system=cluster)
        data = pickle_b64((resource.config_for_rns, state, resource.dryrun))
        response = http_client.post(
            "/resource",
            json=PutResourceParams(serialized_data=data, serialization="pickle").dict(),
            headers=rns_client.request_headers(),
        )
        assert response.status_code == 200

    @pytest.mark.level("local")
    def test_put_object_and_get_keys(self, http_client):
        key = "key1"
        test_list = list(range(5, 50, 2)) + ["a string"]
        response = http_client.post(
            "/object",
            json=PutObjectParams(
                key=key,
                serialized_data=pickle_b64(test_list),
                serialization="pickle",
            ).dict(),
            headers=rns_client.request_headers(),
        )
        assert response.status_code == 200

        response = http_client.get("/keys", headers=rns_client.request_headers())
        assert response.status_code == 200
        assert key in response.json().get("data")

    @pytest.mark.level("local")
    def test_rename_object(self, http_client):
        old_key = "key1"
        new_key = "key2"
        response = http_client.post(
            "/rename",
            json=RenameObjectParams(key=old_key, new_key=new_key).dict(),
            headers=rns_client.request_headers(),
        )
        assert response.status_code == 200

        response = http_client.get("/keys", headers=rns_client.request_headers())
        assert new_key in response.json().get("data")

    @pytest.mark.level("local")
    def test_delete_obj(self, http_client):
        # https://www.python-httpx.org/compatibility/#request-body-on-http-methods
        key = "key2"
        response = http_client.post(
            url="/delete_object",
            json=DeleteObjectParams(keys=[key]).dict(),
            headers=rns_client.request_headers(),
        )
        assert response.status_code == 200

        response = http_client.get("/keys", headers=rns_client.request_headers())
        assert key not in response.json().get("data")

    @pytest.mark.level("local")
    def test_call_module_method(self, http_client, cluster):
        # Create new func on the cluster, then call it
        remote_func = rh.function(summer).to(cluster)

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
            headers=rns_client.request_headers(),
        )
        assert response.status_code == 200

        resp_obj: dict = json.loads(response.text.split("\n")[0])

        if resp_obj["output_type"] == "stdout":
            assert resp_obj["data"] == [
                "base_env servlet: Calling method call on module summer\n"
            ]

        if resp_obj["output_type"] == "result":
            assert b64_unpickle(resp_obj["data"]) == 3

    @pytest.mark.level("local")
    @pytest.mark.asyncio
    async def test_async_call(self, async_http_client, cluster):
        remote_func = rh.function(summer).to(cluster)
        method = "call"

        response = await async_http_client.post(
            f"/call/{remote_func.name}/{method}?serialization=None",
            json={"args": [1, 2]},
            headers=rns_client.request_headers(),
        )
        assert response.status_code == 200
        assert response.text == "3"

    @pytest.mark.level("local")
    @pytest.mark.asyncio
    async def test_async_call_with_invalid_serialization(
        self, async_http_client, cluster
    ):
        remote_func = rh.function(summer, system=cluster)
        method = "call"

        response = await async_http_client.post(
            f"/call/{remote_func.name}/{method}?serialization=random",
            json={"args": [1, 2]},
            headers=rns_client.request_headers(),
        )
        assert response.status_code == 500

    @pytest.mark.level("local")
    @pytest.mark.asyncio
    async def test_async_call_with_pickle_serialization(
        self, async_http_client, cluster
    ):
        remote_func = rh.function(summer, system=cluster)
        method = "call"

        response = await async_http_client.post(
            f"/call/{remote_func.name}/{method}?serialization=pickle",
            json={"args": [1, 2]},
            headers=rns_client.request_headers(),
        )

        assert response.status_code == 200
        assert b64_unpickle(response.text) == 3

    @pytest.mark.level("local")
    @pytest.mark.asyncio
    async def test_async_call_with_json_serialization(self, async_http_client, cluster):
        remote_func = rh.function(summer, system=cluster)
        method = "call"

        response = await async_http_client.post(
            f"/call/{remote_func.name}/{method}?serialization=json",
            json={"args": [1, 2]},
            headers=rns_client.request_headers(),
        )
        assert response.status_code == 200
        assert json.loads(response.text) == "3"


@pytest.mark.servertest
@pytest.mark.den_auth
@pytest.mark.usefixtures("cluster")
class TestHTTPServerDockerDenAuthOnly:
    """
    Testing HTTP Server, but only against a container with Den Auth enabled.

    TODO: What is the expected behavior if there are invalid headers,
    but it is a server without Den Auth enabled at all?
    """

    UNIT = {"cluster": ["docker_cluster_pk_ssh_den_auth"]}
    LOCAL = {"cluster": ["docker_cluster_pk_ssh_den_auth"]}
    MINIMAL = {"cluster": []}
    THOROUGH = {"cluster": ["docker_cluster_pk_ssh_den_auth"]}
    MAXIMAL = {"cluster": ["docker_cluster_pk_ssh_den_auth"]}

    # -------- INVALID TOKEN / CLUSTER ACCESS TESTS ----------- #
    @pytest.mark.level("local")
    def test_no_access_to_cluster(self, http_client, cluster):
        # Make sure test account doesn't have access to the cluster (created by logged-in account, Den Tester in CI)
        cluster.revoke(["info@run.house"])
        cluster.enable_den_auth(flush=True)  # Flush auth cache

        import requests

        with friend_account():  # Test accounts with Den auth are created under test_account
            res = requests.get(
                f"{rns_client.api_server_url}/resource",
                headers=rns_client.request_headers(),
            )
            assert cluster.rns_address not in [
                config["name"] for config in res.json()["data"]
            ]
            response = http_client.get("/keys", headers=rns_client.request_headers())

        assert response.status_code == 403
        assert "Cluster access is required for API" in response.text

    @pytest.mark.level("local")
    def test_request_with_no_token(self, http_client):
        response = http_client.get("/keys")  # No headers are passed
        assert response.status_code == 404

        assert "No token found in request auth headers" in response.text

    @pytest.mark.level("local")
    def test_get_cert_with_invalid_token(self, http_client):
        response = http_client.get("/cert", headers=INVALID_HEADERS)
        # Should be able to download the cert even without a valid token
        assert response.status_code == 200

    @pytest.mark.level("local")
    def test_check_server_with_invalid_token(self, http_client):
        response = http_client.get("/check", headers=INVALID_HEADERS)
        # Should be able to ping the server even without a valid token
        assert response.status_code == 200

    @pytest.mark.level("local")
    def test_put_resource_with_invalid_token(self, http_client, blob_data, cluster):
        state = None
        resource = rh.blob(blob_data, system=cluster)
        data = pickle_b64((resource.config_for_rns, state, resource.dryrun))
        response = http_client.post(
            "/resource",
            json=PutResourceParams(serialized_data=data, serialization="pickle").dict(),
            headers=INVALID_HEADERS,
        )
        assert response.status_code == 403
        assert "Cluster access is required for API" in response.text

    @pytest.mark.level("local")
    def test_call_module_method_with_invalid_token(self, http_client, cluster):
        # Create new func on the cluster, then call it
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
            headers=INVALID_HEADERS,
        )
        assert "No read or write access to requested resource" in response.text

    @pytest.mark.level("local")
    def test_put_object_with_invalid_token(self, http_client):
        test_list = list(range(5, 50, 2)) + ["a string"]
        response = http_client.post(
            "/object",
            json=PutObjectParams(
                key="key1",
                serialized_data=pickle_b64(test_list),
                serialization="pickle",
            ).dict(),
            headers=INVALID_HEADERS,
        )
        assert response.status_code == 403
        assert "Cluster access is required for API" in response.text

    @pytest.mark.level("local")
    def test_rename_object_with_invalid_token(self, http_client):
        old_key = "key1"
        new_key = "key2"
        response = http_client.post(
            "/rename",
            json=RenameObjectParams(key=old_key, new_key=new_key).dict(),
            headers=INVALID_HEADERS,
        )
        assert response.status_code == 403
        assert "Cluster access is required for API" in response.text

    @pytest.mark.level("local")
    def test_get_keys_with_invalid_token(self, http_client):
        response = http_client.get("/keys", headers=INVALID_HEADERS)

        assert response.status_code == 403
        assert "Cluster access is required for API" in response.text


@pytest.fixture(scope="function")
def client(request):
    return request.getfixturevalue(request.param)


@pytest.mark.servertest
@pytest.mark.den_auth
@pytest.mark.usefixtures("setup_cluster_config")
class TestHTTPServerNoDocker:
    """
    Directly analogous to the Docker equivalent above, but with a fully
    local server instead of Docker.

    TODO (RB+JL): This class should really be
    combined with the Docker test class above using some pytest magic.
    """

    # There is no default for this fixture, we should specify
    # it for each testing level.
    UNIT = {"client": ["local_client", "local_client_with_den_auth"]}
    LOCAL = {"client": ["local_client", "local_client_with_den_auth"]}
    MINIMAL = {"client": ["local_client", "local_client_with_den_auth"]}
    THOROUGH = {"client": ["local_client", "local_client_with_den_auth"]}
    MAXIMAL = {"client": ["local_client", "local_client_with_den_auth"]}

    @pytest.mark.level("unit")
    def test_get_cert(self, client):
        # Define the path for the temporary certificate
        certs_dir = Path.home() / "ssl" / "certs"

        # Create the directory if it doesn't exist
        certs_dir.mkdir(parents=True, exist_ok=True)
        cert_path = certs_dir / "rh_server.crt"

        # Create a temporary certificate file
        cert_content = "dummy cert content"
        with open(cert_path, "w") as cert_file:
            cert_file.write(cert_content)

        try:
            # Perform the test
            response = client.get("/cert")
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

    @pytest.mark.level("unit")
    def test_check_server(self, client):
        response = client.get("/check")
        assert response.status_code == 200

    @pytest.mark.level("unit")
    def test_put_resource(self, client, blob_data):
        with tempfile.TemporaryDirectory() as temp_dir:
            resource_path = Path(temp_dir, "local-blob")
            local_blob = rh.blob(blob_data, path=resource_path)
            resource = local_blob.to(system="file", path=resource_path)

            state = None
            data = pickle_b64((resource.config_for_rns, state, resource.dryrun))
            response = client.post(
                "/resource",
                json=dict(
                    PutResourceParams(serialized_data=data, serialization="pickle")
                ),
                headers=rns_client.request_headers(),
            )
            assert response.status_code == 200

    @pytest.mark.level("unit")
    def test_put_object(self, client):
        test_list = list(range(5, 50, 2)) + ["a string"]
        response = client.post(
            "/object",
            json=PutObjectParams(
                key="key1",
                serialized_data=pickle_b64(test_list),
                serialization="pickle",
            ).dict(),
            headers=rns_client.request_headers(),
        )
        assert response.status_code == 200

    @pytest.mark.level("unit")
    def test_rename_object(self, client):
        old_key = "key1"
        new_key = "key2"
        response = client.post(
            "/rename",
            json=RenameObjectParams(key=old_key, new_key=new_key).dict(),
            headers=rns_client.request_headers(),
        )
        assert response.status_code == 200

        response = client.get(
            "/keys",
            headers=rns_client.request_headers(),
        )
        assert new_key in response.json().get("data")

    @pytest.mark.level("unit")
    def test_get_keys(self, client):
        response = client.get(
            "/keys",
            headers=rns_client.request_headers(),
        )
        assert response.status_code == 200
        assert "key2" in response.json().get("data")

    @pytest.mark.level("unit")
    def test_delete_obj(self, client):
        key = "key"
        response = client.post(
            url="/delete_object",
            json=DeleteObjectParams(keys=[key]).dict(),
            headers=rns_client.request_headers(),
        )
        assert response.status_code == 200

        response = client.get("/keys", headers=rns_client.request_headers())
        assert key not in response.json().get("data")

    # TODO [JL]: Test call_module_method and async_call with local and not just Docker.


@pytest.mark.servertest
@pytest.mark.den_auth
@pytest.mark.usefixtures("setup_cluster_config")
class TestHTTPServerNoDockerDenAuthOnly:
    """
    Directly analogous to the Docker equivalent above, but with a fully
    local server instead of Docker.

    TODO (RB+JL): This class should really be
    combined with the Docker test class above using some pytest magic.
    """

    # -------- INVALID TOKEN / CLUSTER ACCESS TESTS ----------- #
    @pytest.mark.level("unit")
    def test_request_with_no_token(self, local_client_with_den_auth):
        response = local_client_with_den_auth.get(
            "/keys", headers={"Authorization": ""}
        )  # No headers are passed
        assert response.status_code == 404

        assert "No token found in request auth headers" in response.text

    @pytest.mark.level("unit")
    def test_get_cert_with_invalid_token(self, local_client_with_den_auth):
        response = local_client_with_den_auth.get("/cert", headers=INVALID_HEADERS)
        # Should be able to download the cert even without a valid token
        assert response.status_code == 200

    @pytest.mark.level("unit")
    def test_check_server_with_invalid_token(self, local_client_with_den_auth):
        response = local_client_with_den_auth.get("/check", headers=INVALID_HEADERS)
        # Should be able to ping the server even without a valid token
        assert response.status_code == 200

    @pytest.mark.level("unit")
    def test_put_resource_with_invalid_token(
        self, local_client_with_den_auth, blob_data
    ):
        with tempfile.TemporaryDirectory() as temp_dir:
            resource_path = Path(temp_dir, "local-blob")
            local_blob = rh.blob(blob_data, path=resource_path)
            resource = local_blob.to(system="file", path=resource_path)
            state = None
            data = pickle_b64((resource.config_for_rns, state, resource.dryrun))
            resp = local_client_with_den_auth.post(
                "/resource",
                json=dict(
                    PutResourceParams(serialized_data=data, serialization="pickle")
                ),
                headers=INVALID_HEADERS,
            )

            assert resp.status_code == 403

    @pytest.mark.level("unit")
    def test_put_object_with_invalid_token(self, local_client_with_den_auth):
        test_list = list(range(5, 50, 2)) + ["a string"]
        resp = local_client_with_den_auth.post(
            "/object",
            json=PutObjectParams(
                key="key1",
                serialized_data=pickle_b64(test_list),
                serialization="pickle",
            ).dict(),
            headers=INVALID_HEADERS,
        )
        assert resp.status_code == 403

    @pytest.mark.level("unit")
    def test_rename_object_with_invalid_token(self, local_client_with_den_auth):
        old_key = "key1"
        new_key = "key2"
        resp = local_client_with_den_auth.post(
            "/rename",
            json=RenameObjectParams(key=old_key, new_key=new_key).dict(),
            headers=INVALID_HEADERS,
        )
        assert resp.status_code == 403

    @pytest.mark.level("unit")
    def test_get_keys_with_invalid_token(self, local_client_with_den_auth):
        resp = local_client_with_den_auth.get("/keys", headers=INVALID_HEADERS)
        assert resp.status_code == 403

    # TODO (JL): Test call_module_method.
