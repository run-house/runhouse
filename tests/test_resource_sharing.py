import os
import subprocess
import unittest

import pytest

import runhouse as rh


@pytest.mark.clustertest
def test_create_shared_resources(shared_resources):
    from runhouse.globals import configs

    current_token = configs.get("token")
    current_username = configs.get("username")
    current_folder = configs.get("default_folder")

    shared_cluster, shared_function = shared_resources
    test_username = os.getenv("TEST_USERNAME")
    assert shared_cluster.rns_address == f"/{test_username}/{shared_cluster.name}"
    assert shared_function.rns_address == f"/{test_username}/{shared_function.name}"

    # Share cluster & function with other user from the test account
    # Update configs before sharing to assume the role of the test account
    configs.set("username", test_username)
    configs.set("token", os.getenv("TEST_TOKEN"))
    configs.set("default_folder", f"/{test_username}")

    shared_cluster.share(current_username, access_type="read")
    shared_function.share(current_username, access_type="read")

    # reset configs to the current account after sharing from the test account
    configs.set("username", current_username)
    configs.set("token", current_token)
    configs.set("default_folder", current_folder)


@pytest.mark.clustertest
def test_cluster_sharing(ondemand_https_cluster_with_auth):
    from runhouse.globals import configs

    current_token = configs.get("token")

    # Load shared cluster by name from the current account that it was shared with
    test_username = os.getenv("TEST_USERNAME")
    shared_cluster = rh.ondemand_cluster(name=f"/{test_username}/rh-cpu-shared")
    return_codes = shared_cluster.run_python(
        ["import numpy", "print(numpy.__version__)"]
    )
    assert return_codes[0][0] == 0

    shared_func_name = "summer_func_shared"

    # Run a function on the shared cluster
    curl_command = f"""curl -k -X POST "https://{shared_cluster.address}/call/{shared_func_name}/call?serialization=None" -d '{{"args": [1, 2]}}' -H "Content-Type: application/json" -H "Authorization: Bearer {current_token}" """  # noqa

    res = subprocess.run(
        curl_command,
        shell=True,
        check=True,
        capture_output=True,
        text=True,
    )
    if res.returncode != 0:
        assert False, f"Failed to run function on shared cluster: {res.stderr}"

    assert "3" in res.stdout

    # Run the re-loaded func that was shared
    loaded_func = rh.function(name=f"/{test_username}/{shared_func_name}")
    assert loaded_func(1, 2) == 3


@pytest.mark.clustertest
@unittest.skip("Not fully implemented yet.")
def test_use_shared_cluster_apis():
    # Should be able to use the shared cluster APIs if given access
    shared_cluster = rh.ondemand_cluster(
        name=f"/{os.getenv('TEST_USERNAME')}/rh-cpu-shared"
    )

    shared_cluster.put("test_obj", list(range(10)))
    assert shared_cluster.get("test_obj") == list(range(10))


if __name__ == "__main__":
    unittest.main()
