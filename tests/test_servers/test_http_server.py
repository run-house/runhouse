import json
import os
import uuid

import pytest

from runhouse.globals import rns_client
from runhouse.resources.resource import Resource
from runhouse.servers.http.http_utils import (
    DeleteObjectParams,
    deserialize_data,
    PutObjectParams,
    PutResourceParams,
    RenameObjectParams,
    serialize_data,
)

from tests.utils import friend_account

INVALID_HEADERS = {"Authorization": "Bearer InvalidToken"}

# Helper used for testing rh.Function
def summer(a, b):
    return a + b


@pytest.mark.servertest
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
            "docker_cluster_pk_tls_den_auth",  # Represents public app use case
            "docker_cluster_pk_http_exposed",  # Represents within VPC use case
        ]
    }

    @pytest.mark.level("local")
    def test_check_server(self, http_client):
        response = http_client.get("/check")
        assert response.status_code == 200

    @pytest.mark.level("local")
    def test_put_resource(self, http_client, cluster):
        state = None
        resource = Resource(name="test_resource3", system=cluster)
        data = serialize_data(
            (resource.config(condensed=False), state, resource.dryrun), "pickle"
        )
        response = http_client.post(
            "/resource",
            json=PutResourceParams(
                serialized_data=data, serialization="pickle"
            ).model_dump(),
            headers=rns_client.request_headers(cluster.rns_address),
        )
        assert response.status_code == 200

    @pytest.mark.level("local")
    def test_put_object_and_get_keys(self, http_client, cluster):
        key = "key1"
        test_list = list(range(5, 50, 2)) + ["a string"]
        response = http_client.post(
            "/object",
            json=PutObjectParams(
                key=key,
                serialized_data=serialize_data(test_list, "pickle"),
                serialization="pickle",
            ).model_dump(),
            headers=rns_client.request_headers(cluster.rns_address),
        )
        assert response.status_code == 200

        response = http_client.get(
            "/keys", headers=rns_client.request_headers(cluster.rns_address)
        )
        assert response.status_code == 200
        assert key in response.json().get("data")

    @pytest.mark.level("local")
    def test_rename_object(self, http_client, cluster):
        old_key = "key1"
        new_key = "key2"
        response = http_client.post(
            "/rename",
            json=RenameObjectParams(key=old_key, new_key=new_key).model_dump(),
            headers=rns_client.request_headers(cluster.rns_address),
        )
        assert response.status_code == 200

        response = http_client.get(
            "/keys", headers=rns_client.request_headers(cluster.rns_address)
        )
        assert new_key in response.json().get("data")

    @pytest.mark.level("local")
    def test_delete_obj(self, http_client, cluster):
        # https://www.python-httpx.org/compatibility/#request-body-on-http-methods
        key = "key2"
        response = http_client.post(
            url="/delete_object",
            json=DeleteObjectParams(keys=[key]).model_dump(),
            headers=rns_client.request_headers(cluster.rns_address),
        )
        assert response.status_code == 200

        response = http_client.get(
            "/keys", headers=rns_client.request_headers(cluster.rns_address)
        )
        assert key not in response.json().get("data")

    # TODO test get_call, refactor into proper fixtures
    @pytest.mark.level("local")
    def test_call_module_method(self, http_client, remote_func):
        # Create new func on the cluster, then call it
        method_name = "call"
        module_name = remote_func.name
        data = {"args": [1, 2], "kwargs": {}}

        response = http_client.post(
            f"{module_name}/{method_name}",
            json={
                "data": serialize_data(data, "pickle"),
                "stream_logs": True,
                "serialization": "pickle",
                "run_name": "test_call_module_method",
            },
            headers=rns_client.request_headers(remote_func.system.rns_address),
        )
        assert response.status_code == 200

        resp_obj: dict = json.loads(response.text.split("\n")[0])

        if resp_obj["output_type"] == "stdout":
            assert (
                "default_env env: Calling method call on module summer\n"
                in resp_obj["data"][0]
            )

        if resp_obj["output_type"] == "result_serialized":
            assert deserialize_data(resp_obj["data"], resp_obj["serialization"]) == 3

    @pytest.mark.level("local")
    def test_call_module_method_get_call(self, http_client, remote_func):
        method_name = "call"
        module_name = remote_func.name

        response = http_client.get(
            f"{module_name}/{method_name}",
            params={
                "a": 1,
                "b": 2,
                "run_name": "test_call_module_method_get_call",
            },
            headers=rns_client.request_headers(remote_func.system.rns_address),
        )
        assert response.status_code == 200

        resp_obj: dict = response.json()

        assert resp_obj["output_type"] == "result_serialized"

        # Right now everything via a GET request is serialized as strings
        assert deserialize_data(resp_obj["data"], resp_obj["serialization"]) == "12"

    @pytest.mark.level("local")
    def test_log_streaming_call(self, http_client, remote_log_streaming_func):
        # Create new func on the cluster, then call it
        method_name = "call"
        module_name = remote_log_streaming_func.name
        clus = remote_log_streaming_func.system
        data = {"args": [3], "kwargs": {}}

        url = f"{clus.endpoint()}/{module_name}/{method_name}"

        with http_client.stream(
            "POST",
            url,
            json={
                "data": serialize_data(data, "pickle"),
                "stream_logs": True,
                "serialization": "pickle",
                "run_name": "test_log_streaming_call",
            },
            headers=rns_client.request_headers(clus.rns_address),
        ) as r:
            assert r.status_code == 200
            for res in r.iter_lines():
                resp_obj: dict = json.loads(res)

                # Might be too aggressive to assert the exact print order and timing, but for now this works
                if resp_obj["output_type"] == "stdout":
                    if "env" in resp_obj["data"][0]:
                        assert (
                            "default_env env: Calling method call on module do_printing_and_logging\n"
                            in resp_obj["data"][0]
                        )
                    else:
                        assert "Hello from the cluster stdout!" in resp_obj["data"][0]
                        assert "Hello from the cluster logs!" in resp_obj["data"][1]

                if resp_obj["output_type"] == "result_serialized":
                    assert (
                        deserialize_data(resp_obj["data"], resp_obj["serialization"])
                        == 3
                    )

    @pytest.mark.level("local")
    def test_folder_put_and_get_and_mv(self, http_client, cluster):
        file_name = str(uuid.uuid4())
        response = http_client.post(
            "/folder/method/put",
            json={
                "path": "~/.rh",
                "contents": {f"{file_name}.txt": "Hello, world!"},
                "overwrite": True,
            },
            headers=rns_client.request_headers(cluster.rns_address),
        )
        assert response.status_code == 200

        response = http_client.post(
            "/folder/method/get",
            json={"path": f"~/.rh/{file_name}.txt"},
            headers=rns_client.request_headers(cluster.rns_address),
        )
        assert response.status_code == 200

        resp_json = response.json()

        assert resp_json["serialization"] is None
        assert resp_json["output_type"] == "result_serialized"
        assert resp_json["data"] == "Hello, world!"

        dest_path = f"~/{file_name}.txt"
        response = http_client.post(
            "/folder/method/mv",
            json={
                "path": f"~/.rh/{file_name}.txt",
                "dest_path": dest_path,
                "overwrite": True,
            },
            headers=rns_client.request_headers(cluster.rns_address),
        )
        assert response.status_code == 200

        response = http_client.post(
            "/folder/method/get",
            json={"path": dest_path},
            headers=rns_client.request_headers(cluster.rns_address),
        )
        assert response.status_code == 200
        assert response.json()["data"] == "Hello, world!"

    @pytest.mark.level("local")
    def test_folder_content_serialization_methods(self, http_client, cluster):
        file_name = str(uuid.uuid4())

        # Save data with "pickle" serialization
        pickle_serialization = "pickle"
        response = http_client.post(
            "/folder/method/put",
            json={
                "path": "~/.rh",
                "contents": serialize_data(
                    data={f"{file_name}.txt": "Hello, world!"},
                    serialization=pickle_serialization,
                ),
                "serialization": pickle_serialization,
                "overwrite": True,
            },
            headers=rns_client.request_headers(cluster.rns_address),
        )
        assert response.status_code == 200

        # Result should be returned as pickled data
        response = http_client.post(
            "/folder/method/get",
            json={"path": f"~/.rh/{file_name}.txt"},
            headers=rns_client.request_headers(cluster.rns_address),
        )

        resp_json = response.json()
        assert resp_json["serialization"] is None
        assert resp_json["output_type"] == "result_serialized"
        assert (
            deserialize_data(resp_json["data"], pickle_serialization) == "Hello, world!"
        )

        # Save data with "json" serialization
        json_serialization = "json"
        response = http_client.post(
            "/folder/method/put",
            json={
                "path": "~/.rh",
                "contents": serialize_data(
                    {"new_file.txt": "Hello, world!"}, json_serialization
                ),
                "serialization": json_serialization,
                "overwrite": True,
            },
            headers=rns_client.request_headers(cluster.rns_address),
        )
        assert response.status_code == 200

        response = http_client.post(
            "/folder/method/get",
            json={"path": "~/.rh/new_file.txt"},
            headers=rns_client.request_headers(cluster.rns_address),
        )

        resp_json = response.json()
        assert resp_json["serialization"] is None
        assert resp_json["output_type"] == "result_serialized"
        assert (
            deserialize_data(resp_json["data"], json_serialization) == "Hello, world!"
        )

        # Save data with no serialization
        response = http_client.post(
            "/folder/method/put",
            json={
                "path": "~/.rh",
                "contents": {"new_file.txt": "Hello, world!"},
                "overwrite": True,
                "serialization": None,
            },
            headers=rns_client.request_headers(cluster.rns_address),
        )
        assert response.status_code == 200

        response = http_client.post(
            "/folder/method/get",
            json={"path": "~/.rh/new_file.txt"},
            headers=rns_client.request_headers(cluster.rns_address),
        )

        resp_json = response.json()
        assert resp_json["serialization"] is None
        assert resp_json["output_type"] == "result_serialized"
        assert resp_json["data"] == "Hello, world!"

    @pytest.mark.level("local")
    def test_folder_put_pickle_object(self, http_client, cluster):
        file_name = str(uuid.uuid4())

        raw_data = [1, 2, 3]
        serialization = "pickle"
        serialized_contents = serialize_data(
            {f"{file_name}.pickle": raw_data}, serialization=serialization
        )

        # need to specify the serialization method here
        response = http_client.post(
            "/folder/method/put",
            json={
                "path": "~/.rh",
                "contents": serialized_contents,
                "overwrite": True,
                "serialization": serialization,
            },
            headers=rns_client.request_headers(cluster.rns_address),
        )
        assert response.status_code == 200

        response = http_client.post(
            "/folder/method/get",
            json={"path": f"~/.rh/{file_name}.pickle"},
            headers=rns_client.request_headers(cluster.rns_address),
        )
        assert response.status_code == 200

        resp_json = response.json()
        assert resp_json["output_type"] == "result_serialized"
        assert deserialize_data(resp_json["data"], serialization) == raw_data

    @pytest.mark.level("local")
    def test_folder_mkdir_rm_and_ls(self, http_client, cluster):
        response = http_client.post(
            "/folder/method/mkdir",
            json={"path": "~/.rh/new-folder"},
            headers=rns_client.request_headers(cluster.rns_address),
        )
        assert response.status_code == 200

        # delete the file
        response = http_client.post(
            "/folder/method/rm",
            json={
                "path": "~/.rh/new-folder",
                "contents": ["new_file.txt"],
            },
            headers=rns_client.request_headers(cluster.rns_address),
        )
        assert response.status_code == 200

        # empty folder should still be there since recursive not explicitly set to `True`
        response = http_client.post(
            "/folder/method/ls",
            json={"path": "~/.rh"},
            headers=rns_client.request_headers(cluster.rns_address),
        )
        assert response.status_code == 200

        file_names: list = response.json().get("data")
        assert "new-folder" in [os.path.basename(f) for f in file_names]

        # Delete the now empty folder
        response = http_client.post(
            "/folder/method/rm",
            json={"path": "~/.rh/new-folder"},
            headers=rns_client.request_headers(cluster.rns_address),
        )
        assert response.status_code == 200

        response = http_client.post(
            "/folder/method/ls",
            json={"path": "~/.rh"},
            headers=rns_client.request_headers(cluster.rns_address),
        )
        assert response.status_code == 200

        file_names: list = response.json().get("data")
        assert "new-folder" not in [os.path.basename(f) for f in file_names]

    @pytest.mark.level("local")
    @pytest.mark.asyncio
    async def test_async_call(self, async_http_client, remote_func):
        method = "call"

        response = await async_http_client.post(
            f"/{remote_func.name}/{method}",
            json={
                "data": {
                    "args": [1, 2],
                    "kwargs": {},
                },
                "serialization": None,
                "run_name": "test_async_call",
            },
            headers=rns_client.request_headers(remote_func.system.rns_address),
        )
        assert response.status_code == 200
        assert response.json() == 3

    @pytest.mark.level("local")
    @pytest.mark.asyncio
    async def test_async_call_with_invalid_serialization(
        self, async_http_client, remote_func
    ):
        method = "call"

        response = await async_http_client.post(
            f"/{remote_func.name}/{method}",
            json={
                "data": {
                    "args": [1, 2],
                    "kwargs": {},
                },
                "serialization": "random",
                "run_name": "test_async_call_with_invalid_serialization",
            },
            headers=rns_client.request_headers(remote_func.system.rns_address),
        )
        assert response.status_code == 400
        assert "Invalid serialization type" in response.text

    @pytest.mark.level("local")
    @pytest.mark.asyncio
    async def test_async_call_with_pickle_serialization(
        self, async_http_client, remote_func
    ):
        method = "call"

        response = await async_http_client.post(
            f"/{remote_func.name}/{method}",
            json={
                "data": serialize_data(
                    {
                        "args": [1, 2],
                        "kwargs": {},
                    },
                    "pickle",
                ),
                "serialization": "pickle",
                "run_name": "test_async_call_with_pickle_serialization",
            },
            headers=rns_client.request_headers(remote_func.system.rns_address),
        )
        assert response.status_code == 200
        assert (
            deserialize_data(response.json()["data"], response.json()["serialization"])
            == 3
        )

    @pytest.mark.level("local")
    @pytest.mark.asyncio
    async def test_async_call_with_json_serialization(
        self, async_http_client, remote_func
    ):
        method = "call"

        response = await async_http_client.post(
            f"/{remote_func.name}/{method}",
            json={
                "data": json.dumps(
                    {
                        "args": [1, 2],
                        "kwargs": {},
                    }
                ),
                "serialization": "json",
                "run_name": "test_async_call_with_json_serialization",
            },
            headers=rns_client.request_headers(remote_func.system.rns_address),
        )
        assert response.status_code == 200
        assert response.json()["data"] == "3"

    @pytest.mark.level("local")
    def test_valid_cluster_token(self, http_client, cluster):
        """User who created the cluster should be able to call cluster APIs with their default Den token
        or with a cluster subtoken."""
        response = http_client.get(
            "/keys",
            headers=rns_client.request_headers(cluster.rns_address),
        )
        assert response.status_code == 200

        # Use cluster subtoken with resource address as the current username (cluster owner)
        response = http_client.get(
            "/keys",
            headers=rns_client.request_headers(rns_client.username),
        )
        assert response.status_code == 200

        # Use cluster subtoken with resource address as the cluster's rns address (/cluster_owner/cluster_name)
        response = http_client.get(
            "/keys",
            headers=rns_client.request_headers(cluster.rns_address),
        )
        assert response.status_code == 200

    @pytest.mark.level("local")
    def test_invalid_cluster_token(self, http_client, cluster):
        """Invalid cluster token should not be able to access the cluster APIs if den auth is enabled."""
        subtoken = f"abcdefg123+{rns_client.username}+{rns_client.username}"
        response = http_client.get(
            "/keys",
            headers={"Authorization": f"Bearer {subtoken}"},
        )

        if cluster.den_auth:
            # Should not be able to access cluster APIs
            assert response.status_code == 403
            assert "Cluster access is required for this operation." in response.text
        else:
            # If den auth is not enabled token should be ignored
            assert response.status_code == 200

    @pytest.mark.level("local")
    def test_no_access_with_invalid_default_token(self, http_client, cluster):
        """Invalid bearer token should result in a 401 if den auth is enabled."""
        invalid_subtoken = "invalid_resource_address"
        response = http_client.get(
            "/keys",
            headers={"Authorization": f"Bearer {invalid_subtoken}"},
        )

        if cluster.den_auth:
            assert response.status_code == 403
        else:
            assert response.status_code == 200


@pytest.mark.servertest
@pytest.mark.usefixtures("cluster")
class TestHTTPServerDockerDenAuthOnly:
    """
    Testing HTTP Server, but only against a container with Den Auth enabled.

    TODO: What is the expected behavior if there are invalid headers,
    but it is a server without Den Auth enabled at all?
    """

    UNIT = {"cluster": ["docker_cluster_pk_ssh_den_auth"]}
    LOCAL = {"cluster": ["docker_cluster_pk_ssh_den_auth"]}
    MINIMAL = {"cluster": ["docker_cluster_pk_ssh_den_auth"]}
    RELEASE = {"cluster": ["docker_cluster_pk_ssh_den_auth"]}
    MAXIMAL = {"cluster": ["docker_cluster_pk_ssh_den_auth"]}

    # -------- INVALID TOKEN / CLUSTER ACCESS TESTS ----------- #
    @pytest.mark.level("local")
    def test_no_access_to_cluster(self, http_client, cluster):
        # Make sure test account doesn't have access to the cluster (created by logged-in account, Den Tester in CI)
        cluster.revoke(["info@run.house"])
        cluster.enable_den_auth(flush=True)  # Flush auth cache

        import requests

        with friend_account():
            # Test accounts with Den auth are created under test_account
            res = requests.get(
                f"{rns_client.api_server_url}/resource",
                headers=rns_client.request_headers(cluster.rns_address),
            )
            assert cluster.rns_address not in [
                config["name"] for config in res.json().get("data", [])
            ]
            response = http_client.get(
                "/keys", headers=rns_client.request_headers(cluster.rns_address)
            )

        assert response.status_code == 403
        assert "Cluster access is required for this operation." in response.text

    @pytest.mark.level("local")
    def test_request_with_no_token(self, http_client):
        response = http_client.get("/keys")  # No headers are passed
        assert response.status_code == 401

        assert "No Runhouse token provided." in response.text

    @pytest.mark.level("local")
    def test_check_server_with_invalid_token(self, http_client):
        response = http_client.get("/check", headers=INVALID_HEADERS)
        # Should be able to ping the server even without a valid token
        assert response.status_code == 200

    @pytest.mark.level("local")
    def test_put_resource_with_invalid_token(self, http_client, cluster):
        state = None
        resource = Resource(name="test_resource1", system=cluster)
        data = serialize_data(
            (resource.config(condensed=False), state, resource.dryrun), "pickle"
        )
        response = http_client.post(
            "/resource",
            json=PutResourceParams(
                serialized_data=data, serialization="pickle"
            ).model_dump(),
            headers=INVALID_HEADERS,
        )
        assert response.status_code == 403
        assert "Cluster access is required for this operation." in response.text

    @pytest.mark.level("local")
    def test_call_module_method_with_invalid_token(self, http_client, remote_func):
        method_name = "call"
        module_name = remote_func.name
        args = (1, 2)
        kwargs = {}

        response = http_client.post(
            f"{module_name}/{method_name}",
            json={
                "data": {"args": args, "kwargs": kwargs},
                "stream_logs": False,
                "serialization": None,
                "run_name": "test_call_module_method_with_invalid_token",
            },
            headers=INVALID_HEADERS,
        )
        assert response.status_code == 403
        assert "Unauthorized access to resource summer" in response.text

    @pytest.mark.level("local")
    def test_put_object_with_invalid_token(self, http_client):
        test_list = list(range(5, 50, 2)) + ["a string"]
        response = http_client.post(
            "/object",
            json=PutObjectParams(
                key="key1",
                serialized_data=serialize_data(test_list, "pickle"),
                serialization="pickle",
            ).model_dump(),
            headers=INVALID_HEADERS,
        )
        assert response.status_code == 403
        assert "Cluster access is required for this operation." in response.text

    @pytest.mark.level("local")
    def test_rename_object_with_invalid_token(self, http_client):
        old_key = "key1"
        new_key = "key2"
        response = http_client.post(
            "/rename",
            json=RenameObjectParams(key=old_key, new_key=new_key).model_dump(),
            headers=INVALID_HEADERS,
        )
        assert response.status_code == 403
        assert "Cluster access is required for this operation." in response.text

    @pytest.mark.level("local")
    def test_get_keys_with_invalid_token(self, http_client):
        response = http_client.get("/keys", headers=INVALID_HEADERS)

        assert response.status_code == 403
        assert "Cluster access is required for this operation." in response.text


@pytest.fixture(scope="function")
def client(request):
    return request.getfixturevalue(request.param)


@pytest.mark.servertest
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
    RELEASE = {"client": ["local_client", "local_client_with_den_auth"]}
    MAXIMAL = {"client": ["local_client", "local_client_with_den_auth"]}

    @pytest.mark.level("unit")
    def test_check_server(self, client):
        response = client.get("/check")
        assert response.status_code == 200

    @pytest.mark.level("unit")
    def test_put_resource(self, client, local_cluster):
        resource = Resource(name="local-resource")
        state = None
        data = serialize_data(
            (resource.config(condensed=False), state, resource.dryrun), "pickle"
        )
        response = client.post(
            "/resource",
            json=PutResourceParams(
                serialized_data=data, serialization="pickle"
            ).model_dump(),
            headers=rns_client.request_headers(local_cluster.rns_address),
        )
        assert response.status_code == 200

    @pytest.mark.level("unit")
    def test_put_object(self, client, local_cluster):
        test_list = list(range(5, 50, 2)) + ["a string"]
        response = client.post(
            "/object",
            json=PutObjectParams(
                key="key1",
                serialized_data=serialize_data(test_list, "pickle"),
                serialization="pickle",
            ).model_dump(),
            headers=rns_client.request_headers(local_cluster.rns_address),
        )
        assert response.status_code == 200

    @pytest.mark.level("unit")
    def test_rename_object(self, client, local_cluster):
        old_key = "key1"
        new_key = "key2"
        response = client.post(
            "/rename",
            json=RenameObjectParams(key=old_key, new_key=new_key).model_dump(),
            headers=rns_client.request_headers(local_cluster.rns_address),
        )
        assert response.status_code == 200

        response = client.get(
            "/keys",
            headers=rns_client.request_headers(local_cluster.rns_address),
        )
        assert new_key in response.json().get("data")

    @pytest.mark.level("unit")
    def test_get_keys(self, client, local_cluster):
        response = client.get(
            "/keys",
            headers=rns_client.request_headers(local_cluster.rns_address),
        )
        assert response.status_code == 200
        assert "key2" in response.json().get("data")

    @pytest.mark.level("unit")
    def test_delete_obj(self, client, local_cluster):
        key = "key"
        response = client.post(
            url="/delete_object",
            json=DeleteObjectParams(keys=[key]).model_dump(),
            headers=rns_client.request_headers(local_cluster.rns_address),
        )
        assert response.status_code == 200

        response = client.get(
            "/keys", headers=rns_client.request_headers(local_cluster.rns_address)
        )
        assert key not in response.json().get("data")

    # TODO [JL]: Test call_module_method and async_call with local and not just Docker.


@pytest.mark.servertest
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
        assert response.status_code == 401

        assert "No Runhouse token provided." in response.text

    @pytest.mark.level("unit")
    def test_check_server_with_invalid_token(self, local_client_with_den_auth):
        response = local_client_with_den_auth.get("/check", headers=INVALID_HEADERS)
        # Should be able to ping the server even without a valid token
        assert response.status_code == 200

    @pytest.mark.level("unit")
    def test_put_resource_with_invalid_token(self, local_client_with_den_auth):
        resource = Resource(name="test_resource2")
        state = None
        data = serialize_data(
            (resource.config(condensed=False), state, resource.dryrun), "pickle"
        )
        resp = local_client_with_den_auth.post(
            "/resource",
            json=PutResourceParams(
                serialized_data=data, serialization="pickle"
            ).model_dump(),
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
                serialized_data=serialize_data(test_list, "pickle"),
                serialization="pickle",
            ).model_dump(),
            headers=INVALID_HEADERS,
        )
        assert resp.status_code == 403

    @pytest.mark.level("unit")
    def test_rename_object_with_invalid_token(self, local_client_with_den_auth):
        old_key = "key1"
        new_key = "key2"
        resp = local_client_with_den_auth.post(
            "/rename",
            json=RenameObjectParams(key=old_key, new_key=new_key).model_dump(),
            headers=INVALID_HEADERS,
        )
        assert resp.status_code == 403

    @pytest.mark.level("unit")
    def test_get_keys_with_invalid_token(self, local_client_with_den_auth):
        resp = local_client_with_den_auth.get("/keys", headers=INVALID_HEADERS)
        assert resp.status_code == 403

    # TODO (JL): Test call_module_method.
