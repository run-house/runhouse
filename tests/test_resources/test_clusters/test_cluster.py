import os
import subprocess
import time
from threading import Thread

import pytest
import requests

import runhouse as rh

from runhouse.constants import (
    DEFAULT_HTTP_PORT,
    DEFAULT_HTTPS_PORT,
    DEFAULT_SERVER_PORT,
    LOCALHOST,
)

from runhouse.resources.hardware.utils import ResourceServerStatus

import tests.test_resources.test_resource
from tests.conftest import init_args
from tests.test_resources.test_envs.test_env import _get_env_var_value
from tests.utils import (
    friend_account,
    friend_account_in_org,
    get_random_str,
    remove_config_keys,
)

""" TODO:
1) In subclasses, test factory methods create same type as parent
2) In subclasses, use monkeypatching to make sure `up()` is called for various methods if the server is not up
3) Test AWS, GCP, and Azure static clusters separately
"""


def load_shared_resource_config(resource_class_name, address):
    resource_class = getattr(rh, resource_class_name)
    loaded_resource = resource_class.from_name(address, dryrun=True)
    return loaded_resource.config()


def summer(a: int, b: int):
    return a + b


def sub(a: int, b: int):
    return a - b


def cluster_keys(cluster):
    return cluster.keys()


def cluster_config():
    return rh.here.config()


def assume_caller_and_get_token():
    token_default = rh.configs.token
    with rh.as_caller():
        token_as_caller = rh.configs.token
    return token_default, token_as_caller


def sleep_fn(secs):
    import time

    time.sleep(secs)


def import_env():
    import pandas  # noqa
    import pytest  # noqa

    return "success"


def run_in_no_env(cmd):
    return rh.here.run(cmd)


def run_node_all(cmd):
    # This forces `cluster.run` to use ssh instead of calling an env run
    return rh.here.run(cmd, node="all")


def sort_env_servlet_processes(env_servlet_processes: dict):
    """helping function for the test_send_status_to_db test, sort the env_servlet_processed dict (including its sub
    dicts) by their keys."""
    keys = list(env_servlet_processes.keys())
    keys.sort()
    sorted_env_servlet_processes = {}
    for k in keys:
        sub_keys = list(env_servlet_processes[k].keys())
        sub_keys.sort()
        nested_dict = {i: env_servlet_processes[k][i] for i in sub_keys}
        sorted_env_servlet_processes[k] = nested_dict
    return sorted_env_servlet_processes


class TestCluster(tests.test_resources.test_resource.TestResource):
    MAP_FIXTURES = {"resource": "cluster"}

    UNIT = {"cluster": ["named_cluster"]}
    LOCAL = {
        "cluster": [
            "docker_cluster_pk_ssh_no_auth",  # Represents private dev use case
            "docker_cluster_pk_ssh_den_auth",  # Helps isolate Auth issues
            "docker_cluster_pk_tls_den_auth",  # Represents public app use case
            "docker_cluster_pk_http_exposed",  # Represents within VPC use case
            "docker_cluster_pwd_ssh_no_auth",
        ],
    }
    MINIMAL = {"cluster": ["static_cpu_pwd_cluster"]}
    RELEASE = {
        "cluster": [
            "static_cpu_pwd_cluster",
        ]
    }
    MAXIMAL = {
        "cluster": [
            "docker_cluster_pk_ssh_no_auth",
            "docker_cluster_pk_ssh_den_auth",
            "docker_cluster_pwd_ssh_no_auth",
            "static_cpu_pwd_cluster",
            "multinode_cpu_docker_conda_cluster",
        ]
    }

    @pytest.mark.level("unit")
    @pytest.mark.clustertest
    def test_cluster_factory_and_properties(self, cluster):
        assert isinstance(cluster, rh.Cluster)
        args = init_args[id(cluster)]
        if "ips" in args:
            # Check that it's a Cluster and not a subclass
            assert cluster.__class__.name == "Cluster"
            assert cluster.ips == args["ips"]
            assert cluster.address == args["ips"][0]

        if "ssh_creds" in args:
            cluster_creds = cluster.creds_values
            if "ssh_private_key" in cluster_creds:
                # this means that the secret was created by accessing an ssh-key file
                cluster_creds.pop("private_key", None)
                cluster_creds.pop("public_key", None)
            assert cluster_creds == args["ssh_creds"]

        if "server_host" in args:
            assert cluster.server_host == args["server_host"]
        else:
            assert cluster.server_host is None

        if "ssl_keyfile" in args:
            assert cluster.cert_config.key_path == args["ssl_keyfile"]

        if "ssl_certfile" in args:
            assert cluster.cert_config.cert_path == args["ssl_certfile"]

    @pytest.mark.level("local")
    @pytest.mark.clustertest
    def test_cluster_recreate(self, cluster):
        # Create underlying ssh connection if not already
        cluster.run(["echo hello"])
        num_open_tunnels = len(rh.globals.ssh_tunnel_cache)

        # Create a new cluster object for the same remote cluster
        cluster.save()
        new_cluster = rh.cluster(cluster.rns_address)
        new_cluster.run(["echo hello"])
        # Check that the same underlying ssh connection was used
        assert len(rh.globals.ssh_tunnel_cache) == num_open_tunnels

    @pytest.mark.level("local")
    @pytest.mark.clustertest
    def test_cluster_endpoint(self, cluster):
        if not cluster.address:
            assert cluster.endpoint() is None
            return

        endpoint = cluster.endpoint()
        if cluster.server_connection_type == "ssh":
            assert cluster.endpoint(external=True) is None
            assert endpoint == f"http://{LOCALHOST}:{cluster.client_port}"
        else:
            url_base = "https" if cluster.server_connection_type == "tls" else "http"
            if cluster.client_port not in [DEFAULT_HTTP_PORT, DEFAULT_HTTPS_PORT]:
                assert (
                    endpoint
                    == f"{url_base}://{cluster.server_address}:{cluster.client_port}"
                )
            else:
                assert endpoint == f"{url_base}://{cluster.server_address}"

        # Try to curl docs
        verify = cluster.client.verify
        r = requests.get(
            f"{endpoint}/status",
            verify=verify,
            headers=rh.globals.rns_client.request_headers(),
        )
        assert r.status_code == 200
        status_data = r.json()[
            0
        ]  # getting the first element because the endpoint returns the status + response to den.
        assert status_data["cluster_config"]["resource_type"] == "cluster"
        assert status_data["env_servlet_processes"]
        assert isinstance(status_data["server_cpu_utilization"], float)
        assert status_data["server_memory_usage"]
        assert not status_data.get("server_gpu_usage", None)

    @pytest.mark.level("local")
    @pytest.mark.clustertest
    def test_cluster_request_timeout(self, docker_cluster_pk_ssh_no_auth):
        cluster = docker_cluster_pk_ssh_no_auth
        with pytest.raises(requests.exceptions.ReadTimeout):
            cluster._http_client.request_json(
                endpoint="/status",
                req_type="get",
                timeout=0.005,
                headers=rh.globals.rns_client.request_headers(),
            )

        status_res = cluster._http_client.request_json(
            endpoint="/status",
            req_type="get",
            headers=rh.globals.rns_client.request_headers(),
        )
        assert status_res

    @pytest.mark.level("local")
    @pytest.mark.clustertest
    def test_cluster_objects(self, cluster):
        k1 = get_random_str()
        k2 = get_random_str()
        cluster.put(k1, "v1")
        cluster.put(k2, "v2")
        assert k1 in cluster.keys()
        assert k2 in cluster.keys()
        assert cluster.get(k1) == "v1"
        assert cluster.get(k2) == "v2"

        # Make new env
        rh.env(reqs=["numpy"], name="numpy_env").to(cluster)
        assert "numpy_env" in cluster.keys()

        k3 = get_random_str()
        cluster.put(k3, "v3", env="numpy_env")
        assert k3 in cluster.keys()
        assert cluster.get(k3) == "v3"

    @pytest.mark.level("local")
    @pytest.mark.clustertest
    def test_cluster_delete_env(self, cluster):
        env1 = rh.env(reqs=["pytest"], name="env1").to(cluster)
        env2 = rh.env(reqs=["pytest"], name="env2").to(cluster)
        env3 = rh.env(reqs=["pytest"], name="env3")

        cluster.put("k1", "v1", env=env1.name)
        cluster.put("k2", "v2", env=env2.name)
        cluster.put_resource(env3, env=env1.name)

        # test delete env2
        assert cluster.get(env2.name)
        assert cluster.get("k2")

        cluster.delete(env2.name)
        assert not cluster.get(env2.name)
        assert not cluster.get("k2")

        # test delete env3, which doesn't affect env1
        assert cluster.get(env3.name)

        cluster.delete(env3.name)
        assert not cluster.get(env3.name)
        assert cluster.get(env1.name)
        assert cluster.get("k1")

    @pytest.mark.level("local")
    @pytest.mark.clustertest
    def test_condensed_config_for_cluster(self, cluster):
        remote_cluster_config = rh.function(cluster_config).to(cluster)
        on_cluster_config = remote_cluster_config()
        local_cluster_config = cluster.config()

        keys_to_skip = [
            "creds",
            "client_port",
            "server_host",
            "api_server_url",
            "ssl_keyfile",
            "ssl_certfile",
        ]
        on_cluster_config = remove_config_keys(on_cluster_config, keys_to_skip)
        local_cluster_config = remove_config_keys(local_cluster_config, keys_to_skip)

        if local_cluster_config.get("stable_internal_external_ips", False):
            cluster_ips = local_cluster_config.pop(
                "stable_internal_external_ips", None
            )[0]
            on_cluster_ips = on_cluster_config.pop(
                "stable_internal_external_ips", None
            )[0]
            assert tuple(cluster_ips) == tuple(on_cluster_ips)

        assert on_cluster_config == local_cluster_config

    @pytest.mark.level("local")
    @pytest.mark.clustertest
    def test_sharing(self, cluster, friend_account_logged_in_docker_cluster_pk_ssh):
        # Skip this test for ondemand clusters, because making
        # it compatible with ondemand_cluster requires changes
        # that break CI.
        # TODO: Remove this by doing some CI-specific logic.
        if cluster.__class__.__name__ == "OnDemandCluster":
            return

        if cluster.rns_address.startswith("~"):
            # For `local_named_resource` resolve the rns address so it can be shared and loaded
            from runhouse.globals import rns_client

            cluster.rns_address = rns_client.local_to_remote_address(
                cluster.rns_address
            )

        cluster.share(
            users=["info@run.house"],
            access_level="read",
            notify_users=False,
        )

        # First try loading in same process/filesystem because it's more debuggable, but not as thorough
        resource_class_name = cluster.config().get("resource_type").capitalize()
        config = cluster.config()

        with friend_account():
            curr_config = load_shared_resource_config(
                resource_class_name, cluster.rns_address
            )
            new_creds = curr_config.get("creds", None)
            assert f'{config["name"]}-ssh-secret' in new_creds
            assert curr_config == config

        # TODO: If we are testing with an ondemand_cluster we to
        # sync sky key so loading ondemand_cluster from config works
        # Also need aws secret to load availability zones
        # secrets=["sky", "aws"],
        load_shared_resource_config_cluster = rh.function(
            load_shared_resource_config
        ).to(friend_account_logged_in_docker_cluster_pk_ssh)
        new_config = load_shared_resource_config_cluster(
            resource_class_name, cluster.rns_address
        )
        new_creds = curr_config.get("creds", None)
        assert f'{config["name"]}-ssh-secret' in new_creds
        assert new_config == config

    @pytest.mark.level("local")
    @pytest.mark.clustertest
    def test_access_to_shared_cluster(self, cluster):
        # TODO: Remove this by doing some CI-specific logic.
        if cluster.__class__.__name__ == "OnDemandCluster":
            return

        if cluster.rns_address.startswith("~"):
            # For `local_named_resource` resolve the rns address so it can be shared and loaded
            from runhouse.globals import rns_client

            cluster.rns_address = rns_client.local_to_remote_address(
                cluster.rns_address
            )

        cluster.share(
            users=["support@run.house"],
            access_level="write",
            notify_users=False,
        )

        cluster_name = cluster.rns_address
        cluster_creds = cluster.creds_values
        cluster_creds.pop("private_key", None)
        cluster_creds.pop("public_key", None)

        with friend_account_in_org():
            shared_cluster = rh.cluster(name=cluster_name)
            assert shared_cluster.rns_address == cluster_name
            assert shared_cluster.creds_values.keys() == cluster_creds.keys()
            echo_msg = "hello from shared cluster"
            run_res = shared_cluster.run([f"echo {echo_msg}"])
            assert echo_msg in run_res[0][1]
            # First element, return code
            assert shared_cluster.run(["echo hello"])[0][0] == 0

    @pytest.mark.level("local")
    @pytest.mark.clustertest
    def test_changing_name_and_saving_in_between(self, cluster):
        remote_summer = rh.function(summer).to(cluster)
        assert remote_summer(3, 4) == 7
        old_name = cluster.name

        cluster.save(name="new_testing_name")

        assert remote_summer(3, 4) == 7
        remote_sub = rh.function(sub).to(cluster)
        assert remote_sub(3, 4) == -1

        cluster_keys_remote = rh.function(cluster_keys).to(cluster)

        # If save did not update the name, this will attempt to create a connection
        # when the cluster is used remotely. However, if you update the name, `on_this_cluster` will
        # work correctly and then the remote function will just call the object store when it calls .keys()
        assert cluster.keys() == cluster_keys_remote(cluster)

        # Restore the state?
        cluster.save(name=old_name)

    @pytest.mark.level("local")
    @pytest.mark.clustertest
    def test_caller_token_propagated(self, cluster):
        remote_assume_caller_and_get_token = rh.function(
            assume_caller_and_get_token
        ).to(cluster)

        remote_assume_caller_and_get_token.share(
            users=["info@run.house"], notify_users=False
        )
        with friend_account():
            unassumed_token, assumed_token = remote_assume_caller_and_get_token()
            # "Local token" is the token the cluster accesses in rh.configs.token; this is what will be used
            # in subsequent rns_client calls
            assert unassumed_token != rh.configs.token

            # Both tokens should be valid for the cluster
            assert rh.globals.rns_client.validate_cluster_token(
                assumed_token, cluster.rns_address
            )
            assert rh.globals.rns_client.validate_cluster_token(
                unassumed_token, cluster.rns_address
            )

        # Docker clusters are logged out, ondemand clusters are logged in
        output = cluster.run("sed -n 's/.*token: *//p' ~/.rh/config.yaml")
        # No config file
        if output[0][0] == 2:
            assert unassumed_token is None
        elif output[0][0] == 0:
            assert unassumed_token == output[0][1].strip()

    ####################################################################################################
    # Status tests
    ####################################################################################################

    @pytest.mark.level("local")
    @pytest.mark.clustertest
    def test_rh_status_pythonic(self, cluster):
        sleep_remote = rh.function(sleep_fn).to(
            cluster, env=rh.env(reqs=["pytest", "pandas"], name="worker_env")
        )
        cluster.put(key="status_key1", obj="status_value1", env="worker_env")
        # Run these in a separate thread so that the main thread can continue
        call_threads = [Thread(target=sleep_remote, args=[3]) for _ in range(3)]
        for call_thread in call_threads:
            call_thread.start()

        # Wait a second so the calls can start
        time.sleep(1)
        cluster_data = cluster.status()

        expected_cluster_status_data_keys = [
            "env_servlet_processes",
            "server_pid",
            "runhouse_version",
            "cluster_config",
        ]

        actual_cluster_status_data_keys = list(cluster_data.keys())

        for key in expected_cluster_status_data_keys:
            assert key in actual_cluster_status_data_keys

        res = cluster_data.get("cluster_config")

        # test cluster config info
        assert res.get("creds") is None
        assert res.get("server_port") == (cluster.server_port or DEFAULT_SERVER_PORT)
        assert res.get("server_connection_type") == cluster.server_connection_type
        assert res.get("den_auth") == cluster.den_auth
        assert res.get("resource_type") == cluster.RESOURCE_TYPE
        assert res.get("ips") == cluster.ips

        assert "worker_env" in cluster_data.get("env_servlet_processes").keys()
        assert "status_key1" in cluster_data.get("env_servlet_processes").get(
            "worker_env"
        ).get("env_resource_mapping")
        assert {
            "resource_type": "str",
            "active_function_calls": [],
        } == cluster_data.get("env_servlet_processes").get("worker_env").get(
            "env_resource_mapping"
        ).get(
            "status_key1"
        )
        sleep_calls = (
            cluster_data.get("env_servlet_processes")
            .get("worker_env")
            .get("env_resource_mapping")
            .get("sleep_fn")
            .get("active_function_calls")
        )

        assert len(sleep_calls) == 3
        assert sleep_calls[0]["key"] == "sleep_fn"
        assert sleep_calls[0]["method_name"] == "call"
        assert sleep_calls[0]["request_id"]
        assert sleep_calls[0]["start_time"]

        # wait for threads to finish
        for call_thread in call_threads:
            call_thread.join()
        updated_status = cluster.status()
        # Check that the sleep calls are no longer active
        assert (
            updated_status.get("env_servlet_processes")
            .get("worker_env")
            .get("env_resource_mapping")
            .get("sleep_fn")
            .get("active_function_calls")
            == []
        )

        # test memory usage info
        expected_env_servlet_keys = [
            "env_cpu_usage",
            "env_gpu_usage",
            "env_resource_mapping",
            "node_index",
            "node_ip",
            "node_name",
            "pid",
        ]
        envs_names = list(cluster_data.get("env_servlet_processes").keys())
        envs_names.sort()
        assert "env_servlet_processes" in cluster_data.keys()
        env_servlets_info = cluster_data.get("env_servlet_processes")
        env_actors_keys = list(env_servlets_info.keys())
        env_actors_keys.sort()
        assert envs_names == env_actors_keys
        for env_name in envs_names:
            env_servlet_info = env_servlets_info.get(env_name)
            env_servlet_info_keys = list(env_servlet_info.keys())
            env_servlet_info_keys.sort()
            assert env_servlet_info_keys == expected_env_servlet_keys

    def status_cli_test_logic(self, cluster, status_cli_command: str):
        default_env_name = cluster.default_env.name

        cluster.put(key="status_key2", obj="status_value2")
        status_output_response = cluster.run(
            [status_cli_command], _ssh_mode="non_interactive"
        )[0]
        assert status_output_response[0] == 0
        status_output_string = status_output_response[1]
        # The string that's returned is utf-8 with the literal escape characters mixed in.
        # We need to convert the escape characters to their actual values to compare the strings.
        status_output_string = status_output_string.encode("utf-8").decode(
            "unicode_escape"
        )
        status_output_string = status_output_string.replace("\n", "")
        assert "Runhouse Daemon is running" in status_output_string
        assert f"Runhouse v{rh.__version__}" in status_output_string
        assert f"server port: {cluster.server_port}" in status_output_string
        assert (
            f"server connection type: {cluster.server_connection_type}"
            in status_output_string
        )
        assert f"den auth: {str(cluster.den_auth)}" in status_output_string
        assert (
            f"resource subtype: {cluster.config().get('resource_subtype')}"
            in status_output_string
        )
        assert f"ips: {str(cluster.ips)}" in status_output_string
        assert "Serving " in status_output_string
        assert (
            f"{default_env_name} (runhouse.Env)" in status_output_string
            or f"{default_env_name} (runhouse.CondaEnv)" in status_output_string
        )
        assert "status_key2 (str)" in status_output_string
        assert "creds" not in status_output_string

        # checking the memory info is printed correctly
        assert "CPU: " in status_output_string
        assert status_output_string.count("CPU: ") >= 1
        assert "pid: " in status_output_string
        assert status_output_string.count("pid: ") >= 1
        assert "node: " in status_output_string
        assert status_output_string.count("node: ") >= 1

        cloud_properties = cluster.config().get("launched_properties", None)
        if cloud_properties:
            properties_to_check = ["cloud", "instance_type", "region", "cost_per_hour"]
            for p in properties_to_check:
                property_value = cloud_properties.get(p)
                assert property_value in status_output_string

    @pytest.mark.level("local")
    @pytest.mark.clustertest
    def test_rh_status_cmd_with_no_den_ping(self, cluster):
        self.status_cli_test_logic(
            cluster=cluster, status_cli_command="runhouse status"
        )

    @pytest.mark.level("local")
    @pytest.mark.clustertest
    def test_rh_status_cmd_with_den_ping(self, cluster):
        self.status_cli_test_logic(
            cluster=cluster, status_cli_command="runhouse status --send-to-den"
        )

    @pytest.mark.skip("Restarting the server mid-test causes some errors, need to fix")
    @pytest.mark.level("local")
    @pytest.mark.clustertest
    # TODO: once fixed, extend this tests for gpu clusters as well.
    def test_rh_status_cli_not_in_cluster(self, cluster):
        default_env_name = cluster.default_env.name

        cluster.put(key="status_key3", obj="status_value3")
        res = str(
            subprocess.check_output(["runhouse", "status", f"{cluster.name}"]), "utf-8"
        )
        assert "😈 Runhouse Daemon is running 🏃" in res
        assert f"server port: {cluster.server_port}" in res
        assert f"server connection_type: {cluster.server_connection_type}" in res
        assert f"den auth: {str(cluster.den_auth)}" in res
        assert f"resource subtype: {cluster.RESOURCE_TYPE.capitalize()}" in res
        assert f"ips: {str(cluster.ips)}" in res
        assert "Serving 🍦 :" in res
        assert f"{default_env_name} (runhouse.Env)" in res
        assert "status_key3 (str)" in res
        assert "ssh certs" not in res

    @pytest.mark.skip("Restarting the server mid-test causes some errors, need to fix")
    @pytest.mark.level("local")
    @pytest.mark.clustertest
    # TODO: once fixed, extend this tests for gpu clusters as well.
    def test_rh_status_stopped(self, cluster):
        try:
            cluster_name = cluster.name
            cluster.run(["runhouse stop"])
            res = subprocess.check_output(["runhouse", "status", cluster_name]).decode(
                "utf-8"
            )
            assert "Runhouse Daemon is not running" in res
            res = subprocess.check_output(
                ["runhouse", "status", f"{cluster_name}_dont_exist"]
            ).decode("utf-8")
            error_txt = (
                f"Cluster {cluster_name}_dont_exist is not found in Den. Please save it, in order to get "
                f"its status"
            )
            assert error_txt in res
        finally:
            cluster.run(["runhouse restart"])

    @pytest.mark.level("local")
    @pytest.mark.clustertest
    def test_send_status_to_db(self, cluster):
        import json

        status = cluster.status()
        env_servlet_processes = status.pop("env_servlet_processes")
        status_data = {
            "status": ResourceServerStatus.running,
            "resource_type": status.get("cluster_config").get("resource_type"),
            "resource_info": status,
            "env_servlet_processes": env_servlet_processes,
        }
        cluster_uri = rh.globals.rns_client.format_rns_address(cluster.rns_address)
        headers = rh.globals.rns_client.request_headers()
        api_server_url = rh.globals.rns_client.api_server_url
        post_status_data_resp = requests.post(
            f"{api_server_url}/resource/{cluster_uri}/cluster/status",
            data=json.dumps(status_data),
            headers=headers,
        )
        assert post_status_data_resp.status_code in [200, 422]
        get_status_data_resp = requests.get(
            f"{api_server_url}/resource/{cluster_uri}/cluster/status?limit=1",
            headers=headers,
        )
        assert get_status_data_resp.status_code == 200
        get_status_data = get_status_data_resp.json()["data"][0]
        assert get_status_data["resource_type"] == status.get("cluster_config").get(
            "resource_type"
        )
        assert get_status_data["status"] == ResourceServerStatus.running

        assert get_status_data["resource_info"] == status
        for k in env_servlet_processes:
            if env_servlet_processes[k]["env_gpu_usage"] == {}:
                env_servlet_processes[k]["env_gpu_usage"] = {
                    "used_memory": None,
                    "utilization_percent": None,
                    "total_memory": None,
                }
        env_servlet_processes = sort_env_servlet_processes(env_servlet_processes)
        get_status_data["env_servlet_processes"] = sort_env_servlet_processes(
            get_status_data["env_servlet_processes"]
        )
        assert get_status_data["env_servlet_processes"] == env_servlet_processes

        status_data["status"] = ResourceServerStatus.terminated
        post_status_data_resp = requests.post(
            f"{api_server_url}/resource/{cluster_uri}/cluster/status",
            data=json.dumps(status_data),
            headers=headers,
        )
        assert post_status_data_resp.status_code == 200
        get_status_data_resp = requests.get(
            f"{api_server_url}/resource/{cluster_uri}/cluster/status?limit=1",
            headers=headers,
        )
        assert (
            get_status_data_resp.json()["data"][0]["status"]
            == ResourceServerStatus.terminated
        )

        # setting the status to running again, so it won't mess with the following tests
        # (when running all release suite at once, for example)
        post_status_data_resp = requests.post(
            f"{api_server_url}/resource/{cluster_uri}/cluster/status",
            data=json.dumps(status_data),
            headers=headers,
        )
        assert post_status_data_resp.status_code in [200, 422]

    ####################################################################################################
    # Default env tests
    ####################################################################################################

    @pytest.mark.level("local")
    @pytest.mark.clustertest
    def test_default_env_in_status(self, cluster):
        res = cluster.status()
        assert cluster.default_env.name in res.get("env_servlet_processes")

    @pytest.mark.level("local")
    @pytest.mark.clustertest
    def test_put_in_default_env(self, cluster):
        k1 = get_random_str()
        cluster.put(k1, "v1")

        assert k1 in cluster.keys(env=cluster.default_env.name)
        cluster.delete(k1)

    @pytest.mark.level("local")
    @pytest.mark.clustertest
    def test_fn_to_default_env(self, cluster):
        remote_summer = rh.function(summer).to(cluster)

        assert remote_summer.name in cluster.keys(env=cluster.default_env.name)
        assert remote_summer(3, 4) == 7

        # Test function with non-trivial imports
        fn = rh.function(import_env).to(cluster)
        assert fn() == "success"

    @pytest.mark.level("local")
    @pytest.mark.clustertest
    def test_run_in_default_env(self, cluster):
        for req in cluster.default_env.reqs:
            if isinstance(req, str) and "_" in req:
                # e.g. pytest_asyncio
                req = req.replace("_", "-")
                assert cluster.run(f"pip freeze | grep {req}")[0][0] == 0

    @pytest.mark.level("local")
    @pytest.mark.clustertest
    def test_default_conda_env_created(self, cluster):
        if not isinstance(cluster.default_env, rh.CondaEnv):
            pytest.skip("Default env is not a CondaEnv")

        assert cluster.default_env.env_name in cluster.run("conda info --envs")[0][1]
        assert isinstance(cluster.get(cluster.default_env.name), rh.CondaEnv)

    @pytest.mark.level("local")
    @pytest.mark.clustertest
    def test_default_env_var_run(self, cluster):
        env_vars = cluster.default_env.env_vars
        if not env_vars:
            pytest.skip("No env vars in default env")

        assert env_vars
        for var in env_vars.keys():
            res = cluster.run([f"echo ${var}"], env=cluster.default_env)
            assert res[0][0] == 0
            assert env_vars[var] in res[0][1]

        get_env_var_cpu = rh.function(_get_env_var_value).to(system=cluster)
        for var in env_vars.keys():
            assert get_env_var_cpu(var) == env_vars[var]

    @pytest.mark.level("local")
    @pytest.mark.clustertest
    def test_cluster_run_within_cluster(self, cluster):
        remote_run = rh.function(run_in_no_env).to(cluster)
        res = remote_run("echo hello")
        exp = cluster.run("echo hello")

        assert res[0][0] == 0
        assert res[0][1] == exp[0][1]

    @pytest.mark.level("local")
    @pytest.mark.clustertest
    def test_cluster_run_within_cluster_node_all(self, cluster):
        remote_run = rh.function(run_node_all).to(cluster)
        # Can't run on a node that is on the cluster
        with pytest.raises(Exception):
            remote_run("echo hello")[0]

    @pytest.mark.level("local")
    @pytest.mark.clustertest
    def test_cluster_put_and_get(self, cluster):
        cluster._folder_mkdir(path="~/.rh/new-folder")
        folder_contents: list = cluster._folder_ls(path="~/.rh")
        assert "new-folder" in [os.path.basename(f) for f in folder_contents]

        cluster._folder_put(
            path="~/.rh/new-folder",
            contents={"sample.txt": "Hello World!"},
            overwrite=True,
        )

        file_contents = cluster._folder_get(path="~/.rh/new-folder/sample.txt")
        assert file_contents == "Hello World!"

        # Should not be able to put to an existing file unless `overwrite=True`
        with pytest.raises(ValueError):
            cluster._folder_put(
                path="~/.rh/new-folder",
                contents={"sample.txt": "Hello World!"},
            )

    @pytest.mark.level("local")
    @pytest.mark.clustertest
    def test_cluster_put_and_get_serialization_methods(self, cluster):
        from runhouse.servers.http.http_utils import deserialize_data

        raw_data = [1, 2, 3]
        pickle_serialization = "pickle"

        cluster._folder_put(
            path="~/.rh/new-folder",
            contents={"sample.pickle": raw_data},
            overwrite=True,
            serialization=pickle_serialization,
        )

        file_contents = cluster._folder_get(path="~/.rh/new-folder/sample.pickle")
        assert deserialize_data(file_contents, pickle_serialization) == raw_data

        json_serialization = "json"
        cluster._folder_put(
            path="~/.rh/new-folder",
            contents={"sample.text": raw_data},
            overwrite=True,
            serialization=json_serialization,
        )

        file_contents = cluster._folder_get(path="~/.rh/new-folder/sample.text")
        assert deserialize_data(file_contents, json_serialization) == raw_data

        with pytest.raises(AttributeError):
            # No serialization specified, default mode of "wb" used which is not supported for a list
            cluster._folder_put(
                path="~/.rh/new-folder",
                contents={"sample.pickle": raw_data},
                overwrite=True,
            )

        # with no serialization specified, but with "w" mode
        cluster._folder_put(
            path="~/.rh/new-folder",
            contents={"sample.txt": raw_data},
            overwrite=True,
            mode="w",
        )

        file_contents = cluster._folder_get(path="~/.rh/new-folder/sample.text")
        assert deserialize_data(file_contents, json_serialization) == raw_data

    @pytest.mark.level("local")
    @pytest.mark.clustertest
    def test_cluster_mkdir_mv_and_rm(self, cluster):
        cluster._folder_mkdir(path="~/.rh/new-folder")

        cluster._folder_mv(path="~/.rh/new-folder", dest_path="~/new-folder")
        file_contents = cluster._folder_ls(path="~")

        assert "new-folder" in [os.path.basename(f) for f in file_contents]

        # Should not be able to mv to an existing directory if `overwrite=False`
        cluster._folder_mkdir(path="~/.rh/another-new-folder")
        with pytest.raises(Exception):
            cluster._folder_mv(
                path="~/.rh/another-new-folder",
                dest_path="~/new-folder",
                overwrite=False,
            )

        # Delete folder contents and directory itself
        cluster._folder_rm(path="~/new-folder", recursive=True)

        assert not cluster._folder_exists(path="~/new-folder")

    @pytest.mark.level("release")
    @pytest.mark.clustertest
    def test_switch_default_env(self, cluster):
        # test setting a new default env, w/o restarting the runhouse server
        test_env = cluster.default_env
        new_env = rh.conda_env(name="new_conda_env", reqs=["diffusers"])
        cluster.default_env = new_env

        # check cluster attr set, and  new env exists on the system
        assert new_env.env_name in cluster.run("conda info --envs")[0][1]
        assert cluster.default_env.name == new_env.name
        assert new_env.name in cluster.status().get("env_servlet_processes")

        # check that env defaults to new default env for run/put
        assert cluster.run("pip freeze | grep diffusers")[0][0] == 0

        k1 = get_random_str()
        cluster.put(k1, "v1")
        assert k1 in cluster.keys(env=new_env.env_name)

        # set it back
        cluster.default_env = test_env
        cluster.delete(new_env.name)
