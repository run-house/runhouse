import json
import subprocess

import pytest
import requests

import runhouse as rh
from runhouse.globals import rns_client
from runhouse.servers.http.certs import TLSCertConfig

from tests.utils import friend_account


class TestResourceSharing:
    UNIT = {"resource": ["shared_function"]}
    LOCAL = {"resource": ["shared_function"]}
    MINIMAL = {"resource": ["shared_function"]}
    RELEASE = {"resource": ["shared_function"]}
    MAXIMAL = {"resource": ["shared_function"]}

    @staticmethod
    def call_func_with_curl(cluster, func_name, cluster_token, **kwargs):
        from urllib.parse import quote

        path_to_cert = TLSCertConfig().cert_path

        # Example for using the cert for verification:
        # curl --cacert '/Users/josh.l/.rh/certs/rh_server.crt' https://localhost:8444/check
        # >> {"rh_version":"0.0.18"}

        query_string = "&".join(
            [f"{quote(str(k))}={quote(str(v))}" for k, v in kwargs.items()]
        )

        cmd = f"""curl --cacert {path_to_cert} -X GET "{cluster.endpoint()}/{func_name}/call?{query_string}" -H "Content-Type: application/json" -H "Authorization: Bearer {cluster_token}" """  # noqa

        res = subprocess.run(
            cmd,
            shell=True,
            check=True,
            capture_output=True,
            text=True,
        )
        return res

    @staticmethod
    def call_cluster_methods(cluster, valid_token):
        cluster_methods = [
            (cluster.put, ("test_obj", list(range(10)))),
            (cluster.get, ("test_obj",)),
            (cluster.keys, ()),
            (cluster.rename, ("test_obj", "test_obj2")),
        ]

        for method, args in cluster_methods:
            try:
                method(*args)
            except Exception as e:
                if valid_token:
                    raise e
                else:
                    assert "Error calling" in str(e)

    @pytest.mark.level("local")
    def test_calling_shared_resource(self, resource):
        cluster = resource.system

        # Run commands on cluster with current token
        return_codes = cluster.run_python(["import numpy", "print(numpy.__version__)"])
        assert return_codes[0][0] == 0

        # Call function with current token via CURL
        cluster_token = rns_client.cluster_token(cluster.rns_address)
        res = self.call_func_with_curl(
            cluster, resource.name, cluster_token, **{"a": 1, "b": 2}
        )

        # Note: kwarg vals should no longer be concatenated once schema support is addde
        assert "12" in res.stdout

        # Reload the shared function and call it
        reloaded_func = rh.function(name=resource.rns_address)
        assert reloaded_func(1, 2) == 3

    @pytest.mark.level("local")
    def test_use_cluster_apis_for_shared_resource(self, resource):
        # Should be able to use the shared cluster APIs if given access
        current_token = rh.configs.token

        # Confirm we can perform cluster actions with the current token
        cluster = resource.system
        self.call_cluster_methods(cluster, valid_token=True)

        # Should not be able to install packages on the cluster with read access
        try:
            cluster.install_packages(["numpy", "pandas"])
        except Exception as e:
            assert "No read or write access to requested resource" in str(e)

        # Confirm we cannot perform actions on the cluster with an invalid token
        rh.configs.token = "abc123"
        self.call_cluster_methods(cluster, valid_token=False)

        # Reset back to valid token
        rh.configs.token = current_token

    @pytest.mark.level("local")
    def test_use_resource_apis(self, resource):
        current_token = rh.configs.token
        cluster = resource.system

        # Call the function with current valid token
        assert resource(2, 2) == 4

        # Reload the shared function and call it
        reloaded_func = rh.function(name=resource.rns_address)
        assert reloaded_func(1, 2) == 3

        # Use invalid token to confirm no function access
        rh.configs.token = "abc123"
        with pytest.raises(Exception):
            # cluster will throw error since the cluster token is invalid
            # note: use the "call" method directly in order to pass new request headers with invalid token
            cluster._http_client.call(
                key=reloaded_func.name,
                method_name="call",
                headers=rns_client.request_headers(resource.rns_address),
            )

        # Reset back to valid token and confirm we can call function again
        rh.configs.token = current_token
        cluster_token = rns_client.cluster_token(cluster.rns_address)

        res = self.call_func_with_curl(
            cluster, resource.name, cluster_token, **{"a": 1, "b": 2}
        )
        assert "12" in res.stdout

    @pytest.mark.level("local")
    def test_calling_resource_with_cluster_write_access(self, resource):
        """Check that a user with write access to a cluster can call a function on that cluster, even without having
        explicit access to the function."""
        current_username = rh.configs.username
        cluster = resource.system

        cluster_uri = rns_client.resource_uri(cluster.rns_address)
        resource_uri = rns_client.resource_uri(resource.rns_address)

        with friend_account():
            friend_account_request_headers = rns_client.request_headers()
            # Give user write access to cluster from test account
            resp = requests.put(
                f"{rns_client.api_server_url}/resource/{cluster_uri}/users/access",
                data=json.dumps(
                    {
                        "users": [current_username],
                        "access_level": "write",
                        "notify_users": False,
                    }
                ),
                headers=friend_account_request_headers,
            )
            if resp.status_code != 200:
                assert (
                    False
                ), f"Failed to grant user {current_username} write access to the cluster: {resp.text}"

            # Delete user access to function
            requests.delete(
                f"{rns_client.api_server_url}/resource/{resource_uri}/user/{current_username}",
                headers=friend_account_request_headers,
            )

        # Confirm user can still call the function with write access to the cluster
        cluster_token = rns_client.cluster_token(cluster.rns_address)
        res = self.call_func_with_curl(
            cluster,
            resource.name,
            cluster_token,
            **{"a": 1, "b": 2, "serialization": "none"},
        )

        assert "12" in res.stdout

        assert resource(1, 2) == 3

    @pytest.mark.level("local")
    @pytest.mark.skip("Breaking in CI - investigate further")
    def test_calling_resource_with_no_cluster_access(self, resource):
        """Check that a user with no access to the cluster can still call a function on that cluster if they were
        given explicit access to the function."""
        current_username = rh.configs.username
        cluster = resource.system

        cluster_uri = rns_client.resource_uri(cluster.rns_address)

        with friend_account():
            # Delete user access to cluster using the test account
            resp = requests.delete(
                f"{rns_client.api_server_url}/resource/{cluster_uri}/user/{current_username}",
                headers=rns_client.request_headers(),
            )
            if resp.status_code != 200:
                assert (
                    False
                ), f"Failed to remove access to the cluster for user: {current_username}: {resp.text}"

        # Confirm current user can still call the function (which they still have explicit access to)
        cluster_token = rns_client.cluster_token(cluster.rns_address)
        res = self.call_func_with_curl(
            cluster, resource.name, cluster_token, **{"a": 1, "b": 2}
        )
        assert "12" in res.stdout

        assert resource(1, 2) == 3

    @pytest.mark.level("local")
    @pytest.mark.skip("Breaking in CI - investigate further")
    def test_calling_resource_with_cluster_read_access(self, resource):
        """Check that a user with read only access to the cluster cannot call a function on that cluster if they do not
        explicitly have access to the function itself."""
        current_username = rh.configs.username
        cluster = resource.system

        cluster_uri = rns_client.resource_uri(cluster.rns_address)
        resource_uri = rns_client.resource_uri(resource.rns_address)

        with friend_account():
            friend_account_request_headers = rns_client.request_headers()

            # Delete user access to the function
            resp = requests.delete(
                f"{rns_client.api_server_url}/resource/{resource_uri}/user/{current_username}",
                headers=friend_account_request_headers,
            )
            if resp.status_code != 200:
                assert (
                    False
                ), f"Failed to remove access to resource for user: {current_username}: {resp.text}"

            # Ensure cluster access for the current user is set to read only
            resp = requests.put(
                f"{rns_client.api_server_url}/resource/{cluster_uri}/users/access",
                data=json.dumps(
                    {
                        "users": [current_username],
                        "access_level": "read",
                        "notify_users": False,
                    }
                ),
                headers=friend_account_request_headers,
            )
            if resp.status_code != 200:
                assert (
                    False
                ), f"Failed to grant user {current_username} read access to cluster: {resp.text}"

        # Flush auth cache
        cluster.enable_den_auth(flush=True)

        # Confirm user can no longer call the function with read only access to the cluster and no function access
        cluster_token = rns_client.cluster_token(cluster.rns_address)
        res = self.call_func_with_curl(
            cluster, resource.name, cluster_token, **{"a": 1, "b": 2}
        )
        assert json.loads(res.stdout)["output_type"] == "exception"

        with pytest.raises(PermissionError):
            resource(1, 2)
