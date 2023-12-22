import json
import subprocess
import unittest

import pytest
import requests

import runhouse as rh
from runhouse.globals import rns_client

from tests.utils import test_account


def call_func_with_curl(ip_address, func_name, token, *args):
    cmd = f"""curl -k -X POST "https://{ip_address}/call/{func_name}/call?serialization=None" -d '{{"args": {list(args)}}}' -H "Content-Type: application/json" -H "Authorization: Bearer {token}" """  # noqa
    res = subprocess.run(
        cmd,
        shell=True,
        check=True,
        capture_output=True,
        text=True,
    )
    return res


def update_cluster_auth_cache(cluster, token):
    """Refresh cache on cluster for current user to reflect any Den updates made in the test."""
    refresh_cmd = f"update_cache_for_user('{token}')"
    cluster.run_python(
        ["from runhouse.servers.http.auth import update_cache_for_user", refresh_cmd]
    )


def call_cluster_methods(cluster, test_env, valid_token):
    cluster_methods = [
        (cluster.put, ("test_obj", list(range(10)))),
        (cluster.get, ("test_obj",)),
        (cluster.keys, ()),
        (cluster.rename, ("test_obj", "test_obj2")),
        (cluster.add_secrets, ({"aws": "abc123"},)),
        (cluster.put_resource, (test_env,)),
    ]

    for method, args in cluster_methods:
        try:
            method(*args)
        except Exception as e:
            if valid_token:
                raise e
            else:
                assert "Error calling" in str(e)


@pytest.mark.clustertest
def test_cluster_sharing(shared_cluster, shared_function):
    current_token = rh.configs.get("token")
    # Run commands on cluster with current token
    return_codes = shared_cluster.run_python(
        ["import numpy", "print(numpy.__version__)"]
    )
    assert return_codes[0][0] == 0

    # Call function with current token via CURL
    res = call_func_with_curl(
        shared_cluster.address, shared_function.name, current_token, 1, 2
    )
    assert "3" in res.stdout

    # Reload the shared function and call it
    reloaded_func = rh.function(name=shared_function.rns_address)
    assert reloaded_func(1, 2) == 3


@pytest.mark.clustertest
def test_use_shared_cluster_apis(shared_cluster, shared_function, test_env):
    # Should be able to use the shared cluster APIs if given access
    current_token = rh.configs.get("token")

    # Confirm we can perform cluster actions with the current token
    call_cluster_methods(shared_cluster, test_env, valid_token=True)

    # Should not be able to install packages on the cluster with read access
    try:
        shared_cluster.install_packages(["numpy", "pandas"])
    except Exception as e:
        assert "No read or write access to requested resource" in str(e)

    # Confirm we cannot perform actions on the cluster with an invalid token
    rh.configs.set("token", "abc123")
    call_cluster_methods(shared_cluster, test_env, valid_token=False)

    # Reset back to valid token
    rh.configs.set("token", current_token)


@pytest.mark.clustertest
def test_use_shared_function_apis(shared_cluster, shared_function):
    current_token = rh.configs.get("token")

    # Call the function with current valid token
    assert shared_function(2, 2) == 4

    # Reload the shared function and call it
    reloaded_func = rh.function(name=shared_function.rns_address)
    assert reloaded_func(1, 2) == 3

    # Use invalid token to confirm no function access
    rh.configs.set("token", "abc123")
    try:
        shared_function(2, 2) == 4
    except Exception as e:
        assert "Error calling call on server" in str(e)

    # Reset back to valid token and confirm we can call function again
    rh.configs.set("token", current_token)
    res = call_func_with_curl(
        shared_cluster.address, shared_function.name, current_token, 1, 2
    )
    assert "3" in res.stdout


@pytest.mark.clustertest
def test_running_func_with_cluster_read_access(shared_cluster, shared_function):
    """Check that a user with read only access to the cluster cannot call a function on that cluster if they do not
    explicitly have access to the function."""
    current_username = rh.configs.get("username")
    current_token = rh.configs.get("token")

    with test_account():
        # Delete user access to the function
        resource_uri = rns_client.resource_uri(shared_function.rns_address)

        resp = requests.delete(
            f"{rns_client.api_server_url}/resource/{resource_uri}/user/{current_username}",
            headers=rns_client.request_headers,
        )
        if resp.status_code != 200:
            assert False, f"Failed to delete user access to resource: {resp.text}"

        update_cluster_auth_cache(shared_cluster, current_token)

    # Confirm user can no longer call the function since only has read access to the cluster
    res = call_func_with_curl(
        shared_cluster.address, shared_function.name, current_token, 1, 2
    )
    assert "Internal Server Error" in res.stdout

    try:
        shared_function(1, 2) == 3
    except Exception as e:
        assert "No read or write access to requested resource" in str(e)


@pytest.mark.clustertest
def test_running_func_with_cluster_write_access(shared_cluster, shared_function):
    """Check that a user with write access to a cluster can call a function on that cluster, even without having
    explicit access to the function."""
    current_username = rh.configs.get("username")
    current_token = rh.configs.get("token")

    cluster_uri = rns_client.resource_uri(shared_cluster.rns_address)

    with test_account():
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
            headers=rns_client.request_headers,
        )
        if resp.status_code != 200:
            assert False, f"Failed to give write access to cluster: {resp.text}"

        # Delete user access to function
        resource_uri = rns_client.resource_uri(shared_function.rns_address)

        resp = requests.delete(
            f"{rns_client.api_server_url}/resource/{resource_uri}/user/{current_username}",
            headers=rns_client.request_headers,
        )
        if resp.status_code != 200:
            assert False, f"Failed to delete user access to resource: {resp.text}"

        update_cluster_auth_cache(shared_cluster, current_token)

    # Confirm user can still call the function with write access to the cluster
    res = call_func_with_curl(
        shared_cluster.address, shared_function.name, current_token, 1, 2
    )
    assert res.stdout == "3"

    assert shared_function(1, 2) == 3


@pytest.mark.clustertest
def test_running_func_with_no_cluster_access(shared_cluster, shared_function):
    """Check that a user with no access to the cluster can still call a function on that cluster if they were
    given explicit access to the function."""
    current_username = rh.configs.get("username")
    current_token = rh.configs.get("token")

    with test_account():
        # Delete user access to cluster using the test account
        cluster_uri = rns_client.resource_uri(shared_cluster.rns_address)
        resp = requests.delete(
            f"{rns_client.api_server_url}/resource/{cluster_uri}/user/{current_username}",
            headers=rns_client.request_headers,
        )
        if resp.status_code != 200:
            assert False, f"Failed to delete user access to cluster: {resp.text}"

        update_cluster_auth_cache(shared_cluster, current_token)

    # Confirm current user can still call the function
    res = call_func_with_curl(
        shared_cluster.address, shared_function.name, current_token, 1, 2
    )
    assert res.stdout == "3"

    assert shared_function(1, 2) == 3


if __name__ == "__main__":
    unittest.main()
