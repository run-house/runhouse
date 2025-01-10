import json
import os
import subprocess
import time

from datetime import datetime, timezone
from threading import Thread

import pytest
import requests

import runhouse as rh

from runhouse.constants import (
    DEFAULT_HTTP_PORT,
    DEFAULT_HTTPS_PORT,
    DEFAULT_PROCESS_NAME,
    DEFAULT_SERVER_PORT,
    HOUR,
    LAST_ACTIVE_AT_TIMEFRAME,
    LOCALHOST,
    MINUTE,
)
from runhouse.globals import rns_client

from runhouse.resources.hardware.cluster import Cluster
from runhouse.resources.hardware.utils import ClusterStatus, RunhouseDaemonStatus

from runhouse.resources.images.image import ImageSetupStepType

from runhouse.utils import _process_env_vars

import tests.test_resources.test_resource
from tests.conftest import init_args
from tests.fixtures.resource_fixtures import create_folder_path
from tests.utils import (
    _get_env_var_value,
    friend_account,
    friend_account_in_org,
    get_random_str,
    keep_config_keys,
    org_friend_account,
    set_daemon_and_cluster_status,
    set_output_env_vars,
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
    return rh.here.run_bash(cmd)


def run_node_all(cmd):
    # This forces `cluster.run` to use ssh instead of calling an env run
    return rh.here.run_bash(cmd, node="all")


def sort_env_servlet_processes(env_servlet_processes: dict):
    """helping function for the test_send_status_to_db test, sort the servlet_processed dict (including its sub
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
            "docker_cluster_pk_http_exposed",  # Represents within VPC use case
            "docker_cluster_pwd_ssh_no_auth",
        ],
    }
    MINIMAL = {"cluster": ["static_cpu_pwd_cluster"]}
    RELEASE = {
        "cluster": ["static_cpu_pwd_cluster", "static_cpu_pwd_cluster_den_launcher"]
    }
    MAXIMAL = {
        "cluster": [
            "docker_cluster_pk_ssh_no_auth",
            "docker_cluster_pk_ssh_den_auth",
            "docker_cluster_pwd_ssh_no_auth",
            "static_cpu_pwd_cluster",
            "multinode_cpu_docker_conda_cluster",
            "static_gpu_pwd_cluster_den_launcher",
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
            assert cluster.head_ip == args["ips"][0]

        if "ssh_creds" in args:
            args_creds = args["ssh_creds"]
            args_creds_values = (
                args_creds.values if isinstance(args_creds, rh.Secret) else args_creds
            )

            cluster_creds = cluster.creds_values
            if "ssh_private_key" in cluster_creds:
                # this means that the secret was created by accessing an ssh-key file
                cluster_creds.pop("private_key", None)
                cluster_creds.pop("public_key", None)
            assert cluster_creds == args_creds_values

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
        # # Create underlying ssh connection if not already
        if cluster.ips == cluster.internal_ips != ["localhost"]:
            cluster.run_bash_over_ssh(["echo hello"])
        else:
            cluster.run_bash(["echo hello"])

        num_open_tunnels = len(rh.globals.ssh_tunnel_cache)

        # Create a new cluster object for the same remote cluster
        cluster.save()
        new_cluster = rh.cluster(cluster.rns_address)
        if new_cluster.ips == new_cluster.internal_ips != ["localhost"]:
            new_cluster.run_bash_over_ssh(["echo hello"])
        else:
            new_cluster.run_bash(["echo hello"])

        # Check that the same underlying ssh connection was used
        assert len(rh.globals.ssh_tunnel_cache) == num_open_tunnels

    @pytest.mark.level("local")
    @pytest.mark.clustertest
    def test_cluster_endpoint(self, cluster):
        if not cluster.ips:
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
        assert status_data.get("cluster_config").get(
            "resource_subtype"
        ) == cluster.config().get("resource_subtype")
        assert status_data.get("env_servlet_processes", None)
        assert isinstance(status_data.get("server_cpu_utilization", None), float)
        assert status_data.get("server_memory_usage", None)
        assert not status_data.get("server_gpu_usage", None)

    @pytest.mark.level("local")
    @pytest.mark.clustertest
    def test_cluster_request_timeout(self, cluster):
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
        cluster.ensure_process_created("numpy_env")

        k3 = get_random_str()
        cluster.put(k3, "v3", process="numpy_env")
        assert k3 in cluster.keys()
        assert cluster.get(k3) == "v3"

    @pytest.mark.skip("pending cluster.kill functionality")
    @pytest.mark.level("local")
    @pytest.mark.clustertest
    def test_cluster_delete_process(self, cluster):
        proc1 = cluster.ensure_process_created("p1")
        proc2 = cluster.ensure_process_created("p2")

        cluster.put("k1", "v1", process=proc1)
        cluster.put("k2", "v2", process=proc2)

        # test delete env2
        assert cluster.get(proc2)
        assert cluster.get("k2")

        cluster.delete(proc2)
        assert not cluster.get(proc2)
        assert not cluster.get("k2")

        assert cluster.get(proc1)
        assert cluster.get("k1")

    @pytest.mark.level("local")
    @pytest.mark.clustertest
    def test_condensed_config_for_cluster(self, cluster):
        remote_cluster_config = rh.function(cluster_config).to(cluster)
        on_cluster_config = remote_cluster_config()
        local_cluster_config = cluster.config()

        keys_to_keep = [
            "name",
            "is_gpu",
            "resource_subtype",
            "den_auth",
            "server_port",
            "server_connection_type",
            "autostop_mins",
            "domain",
        ]
        if local_cluster_config.get("resource_subtype", None) == "OnDemandCluster":
            keys_to_keep.append("compute_properties")
        else:
            keys_to_keep.append("ips")
        on_cluster_config = keep_config_keys(on_cluster_config, keys_to_keep)
        local_cluster_config = keep_config_keys(local_cluster_config, keys_to_keep)

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

        sky_secret = "ssh-sky-key"
        generated_secret = f'{config["name"]}-ssh-secret'
        # CI testing can be flakey depending on den_tester user creds and overwriting: test against both values
        # expected_creds = (
        #     sky_secret
        #     if isinstance(cluster._creds, SSHSecret)
        #     else generated_secret
        # )

        with friend_account():
            curr_config = load_shared_resource_config(
                resource_class_name, cluster.rns_address
            )
            new_creds = curr_config.get("creds", None)
            assert sky_secret in new_creds or generated_secret in new_creds
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
        assert sky_secret in new_creds or generated_secret in new_creds
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

        with friend_account_in_org():
            shared_cluster = rh.cluster(name=cluster_name)
            assert shared_cluster.rns_address == cluster_name
            assert shared_cluster.creds_values.keys() == cluster_creds.keys()
            echo_msg = "hello from shared cluster"

            if shared_cluster.ips == shared_cluster.internal_ips != ["localhost"]:
                run_res = shared_cluster.run_bash_over_ssh([f"echo {echo_msg}"])
            else:
                run_res = shared_cluster.run_bash([f"echo {echo_msg}"])

            assert echo_msg in run_res[0][1]
            # First element, return code
            if shared_cluster.ips == shared_cluster.internal_ips != ["localhost"]:
                assert shared_cluster.run_bash_over_ssh(["echo hello"])[0][0] == 0
            else:
                assert shared_cluster.run_bash(["echo hello"])[0][0] == 0

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
        if cluster.ips == cluster.internal_ips != ["localhost"]:
            output = cluster.run_bash_over_ssh(
                "sed -n 's/.*token: *//p' ~/.rh/config.yaml"
            )
        else:
            output = cluster.run_bash("sed -n 's/.*token: *//p' ~/.rh/config.yaml")
        # No config file
        if output[0] == 2:
            assert unassumed_token is None
        elif output[0] == 0:
            assert unassumed_token == output[1].strip()

    ####################################################################################################
    # Status tests
    ####################################################################################################

    @pytest.mark.level("local")
    @pytest.mark.clustertest
    def test_rh_status_pythonic(self, cluster):
        process = cluster.ensure_process_created("worker")
        sleep_remote = rh.function(sleep_fn).to(cluster, process=process)
        cluster.put(key="status_key1", obj="status_value1", process=process)
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
        assert res.get("server_port") == (cluster.server_port or DEFAULT_SERVER_PORT)
        assert res.get("server_connection_type") == cluster.server_connection_type
        assert res.get("den_auth") == cluster.den_auth
        assert res.get("resource_type") == cluster.RESOURCE_TYPE
        if res.get("resource_subtype") == "Cluster":
            assert res.get("ips") == cluster.ips
        else:
            assert res.get("compute_properties").get("ips") == cluster.ips

        assert process in cluster_data.get("env_servlet_processes").keys()
        assert "status_key1" in cluster_data.get("env_servlet_processes").get(
            process
        ).get("env_resource_mapping")
        assert {
            "resource_type": "str",
            "active_function_calls": [],
        } == cluster_data.get("env_servlet_processes").get(process).get(
            "env_resource_mapping"
        ).get(
            "status_key1"
        )
        sleep_calls = (
            cluster_data.get("env_servlet_processes")
            .get(process)
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
            .get(process)
            .get("env_resource_mapping")
            .get("sleep_fn")
            .get("active_function_calls")
            == []
        )

        # test memory usage info
        expected_servlet_keys = [
            "env_cpu_usage",
            "env_gpu_usage",
            "env_resource_mapping",
            "node_index",
            "node_ip",
            "node_name",
            "pid",
        ]
        process_names = list(cluster_data.get("env_servlet_processes").keys())
        process_names.sort()
        assert "env_servlet_processes" in cluster_data.keys()
        servlets_info = cluster_data.get("env_servlet_processes")
        actors_keys = list(servlets_info.keys())
        actors_keys.sort()
        assert process_names == actors_keys
        for process_name in process_names:
            servlet_info = servlets_info.get(process_name)
            servlet_info_keys = list(servlet_info.keys())
            servlet_info_keys.sort()
            assert servlet_info_keys == expected_servlet_keys

    @pytest.mark.skip("pending cluster.kill functionality")
    @pytest.mark.level("local")
    @pytest.mark.clustertest
    def test_rh_status_pythonic_delete_process(self, cluster):
        process = cluster.ensure_process_created("process_name")
        summer_temp = rh.function(summer).to(system=cluster, process=process)
        call_summer_temp = summer_temp(1, 3)
        assert call_summer_temp == 4

        # make sure status is calculated properly before process deletion.
        self.test_rh_status_pythonic(cluster=cluster)

        cluster.delete(process)
        # make sure status is calculated properly after process deletion.
        self.test_rh_status_pythonic(cluster=cluster)

    def status_cli_test_logic(self, cluster, status_cli_command: str):
        default_process_name = DEFAULT_PROCESS_NAME

        cluster.put(key="status_key2", obj="status_value2")

        if cluster.ips == cluster.internal_ips != ["localhost"]:
            status_output_response = cluster.run_bash_over_ssh([status_cli_command])[0]
        else:
            status_output_response = cluster.run_bash([status_cli_command])[0]
        assert status_output_response[0] == 0
        status_output_string = status_output_response[1]
        # The string that's returned is utf-8 with the literal escape characters mixed in.
        # We need to convert the escape characters to their actual values to compare the strings.
        status_output_string = status_output_string.encode("utf-8").decode(
            "unicode_escape"
        )
        status_output_string = status_output_string.replace("\n", "")
        assert "Runhouse server is running" in status_output_string
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
        assert f"{default_process_name}" in status_output_string
        assert "status_key2 (str)" in status_output_string
        assert "creds" not in status_output_string

        # checking the memory info is printed correctly
        assert "CPU: " in status_output_string
        assert status_output_string.count("CPU: ") >= 1
        assert "pid: " in status_output_string
        assert status_output_string.count("pid: ") >= 1
        assert "node: " in status_output_string
        assert status_output_string.count("node: ") >= 1

        cloud_properties = cluster.config().get("compute_properties", None)
        if cloud_properties:
            properties_to_check = ["cloud", "instance_type", "region", "cost_per_hour"]
            for p in properties_to_check:
                property_value = cloud_properties.get(p)
                assert property_value in status_output_string

    @pytest.mark.level("local")
    @pytest.mark.clustertest
    def test_rh_status_cmd_with_no_den_ping(self, cluster):
        self.status_cli_test_logic(
            cluster=cluster, status_cli_command="runhouse cluster status"
        )

    @pytest.mark.level("local")
    @pytest.mark.clustertest
    def test_rh_status_cmd_with_den_ping(self, cluster):
        self.status_cli_test_logic(
            cluster=cluster, status_cli_command="runhouse cluster status --send-to-den"
        )

    @pytest.mark.skip("Restarting the server mid-test causes some errors, need to fix")
    @pytest.mark.level("local")
    @pytest.mark.clustertest
    # TODO: once fixed, extend this tests for gpu clusters as well.
    def test_rh_status_cli_not_in_cluster(self, cluster):
        default_process_name = DEFAULT_PROCESS_NAME

        cluster.put(key="status_key3", obj="status_value3")
        res = str(
            subprocess.check_output(
                ["runhouse", "cluster", "status", f"{cluster.name}"]
            ),
            "utf-8",
        )
        assert "😈 Runhouse server is running 🏃" in res
        assert f"server port: {cluster.server_port}" in res
        assert f"server connection_type: {cluster.server_connection_type}" in res
        assert f"den auth: {str(cluster.den_auth)}" in res
        assert f"resource subtype: {cluster.RESOURCE_TYPE.capitalize()}" in res
        assert f"ips: {str(cluster.ips)}" in res
        assert "Serving 🍦 :" in res
        assert f"{default_process_name} (runhouse.Env)" in res
        assert "status_key3 (str)" in res
        assert "ssh certs" not in res

    @pytest.mark.skip("Restarting the server mid-test causes some errors, need to fix")
    @pytest.mark.level("local")
    @pytest.mark.clustertest
    # TODO: once fixed, extend this tests for gpu clusters as well.
    def test_rh_status_stopped(self, cluster):
        try:
            cluster_name = cluster.name
            cluster.run_bash(["runhouse server stop"])
            res = subprocess.check_output(["runhouse", "status", cluster_name]).decode(
                "utf-8"
            )
            assert "Runhouse Daemon is not running" in res
            res = subprocess.check_output(
                ["runhouse", "cluster", "status", f"{cluster_name}_dont_exist"]
            ).decode("utf-8")
            error_txt = (
                f"Cluster {cluster_name}_dont_exist is not found in Den. Please save it, in order to get "
                f"its status"
            )
            assert error_txt in res
        finally:
            cluster.run_bash(["runhouse server restart"])

    @pytest.mark.level("local")
    @pytest.mark.clustertest
    def test_send_status_to_db(self, cluster):

        status = cluster.status()
        env_servlet_processes = status.pop("env_servlet_processes")
        status_data = {
            "daemon_status": RunhouseDaemonStatus.RUNNING,
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
        assert get_status_data["daemon_status"] == RunhouseDaemonStatus.RUNNING

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

        status_data["daemon_status"] = RunhouseDaemonStatus.TERMINATED
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
            get_status_data_resp.json()["data"][0]["daemon_status"]
            == RunhouseDaemonStatus.TERMINATED
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
    # Default process tests
    ####################################################################################################

    @pytest.mark.level("local")
    @pytest.mark.clustertest
    def test_default_process_in_status(self, cluster):
        res = cluster.status()
        assert DEFAULT_PROCESS_NAME in res.get("env_servlet_processes")

    @pytest.mark.level("local")
    @pytest.mark.clustertest
    def test_put_in_default_process(self, cluster):
        k1 = get_random_str()
        cluster.put(k1, "v1")

        assert k1 in cluster.keys(process=DEFAULT_PROCESS_NAME)
        cluster.delete(k1)

    @pytest.mark.level("local")
    @pytest.mark.clustertest
    def test_fn_to_default_process(self, cluster):
        remote_summer = rh.function(summer).to(cluster)

        assert remote_summer.name in cluster.keys(process=DEFAULT_PROCESS_NAME)
        assert remote_summer(3, 4) == 7

        # Test function with non-trivial imports
        fn = rh.function(import_env).to(cluster)
        assert fn() == "success"

    @pytest.mark.level("local")
    @pytest.mark.clustertest
    def test_run_in_default_process(self, cluster):
        reqs = []
        if cluster.image:
            for step in cluster.image.setup_steps:
                if step.step_type == ImageSetupStepType.PACKAGES:
                    reqs += step.kwargs.get("reqs")
        for req in reqs:
            if isinstance(req, str) and "_" in req:
                # e.g. pytest_asyncio
                req = req.replace("_", "-")
                assert cluster.run_bash(f"pip freeze | grep {req}")[0] == 0

    @pytest.mark.level("local")
    @pytest.mark.clustertest
    def test_default_conda_env_created(self, cluster):
        if not cluster.image or not cluster.image.conda_env_name:
            pytest.skip("Default process is not in a conda env")

        assert cluster.image.conda_env_name in cluster.run_bash("conda info --envs")[1]

    @pytest.mark.level("local")
    @pytest.mark.clustertest
    def test_default_process_env_var_run(self, cluster):
        env_vars = {}
        if cluster.image:
            for setup_step in cluster.image.setup_steps:
                if setup_step.step_type == ImageSetupStepType.SET_ENV_VARS:
                    image_env_vars = _process_env_vars(
                        setup_step.kwargs.get("env_vars")
                    )
                    env_vars.update(image_env_vars)
        if not env_vars:
            pytest.skip("No env vars in cluster image")

        assert env_vars
        for var in env_vars.keys():
            res = cluster.run_bash([f"echo ${var}"])
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

        if cluster.ips == cluster.internal_ips != ["localhost"]:
            exp = cluster.run_bash_over_ssh(["echo hello"])
        else:
            exp = cluster.run_bash(["echo hello"])

        # multinode case; check that the call passes on all nodes
        if len(cluster.ips) > 1:
            assert len(res) == len(exp)
            for node_id in range(len(res)):
                assert res[node_id][0] == 0
                assert res[node_id][1].strip() == exp[node_id][1].strip()
        else:
            # single-node case
            assert res[0] == 0
            assert res[1].strip() == exp[0][1].strip()

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

    @pytest.mark.level("local")
    @pytest.mark.clustertest
    def test_observability_enabled_by_default_on_cluster(self, cluster):
        # Disable observability locally, which will be reflected on the cluster once the server is restarted
        if cluster.image:
            rh.configs.disable_observability()
            cluster.restart_server()

            res = cluster.run_bash(["echo $disable_observability"])
            assert "True" in res[0][1]

    ####################################################################################################
    # Cluster list test
    ####################################################################################################
    @pytest.mark.level("local")
    @pytest.mark.clustertest
    def test_cluster_list_default_pythonic(self, cluster):
        original_username = rns_client.username
        new_username = (
            "test-org"
            if cluster.rns_address.startswith("/test-org/")
            else original_username
        )

        with org_friend_account(
            new_username=new_username,
            token=rns_client.token,
            original_username=original_username,
        ):
            default_clusters = Cluster.list().get("den_clusters", {})
            assert len(default_clusters) > 0
            assert [
                den_cluster.get("Status") == ClusterStatus.RUNNING
                for den_cluster in default_clusters
            ]
            assert any(
                den_cluster
                for den_cluster in default_clusters
                if den_cluster.get("Name") == cluster.name
            )

    @pytest.mark.level("local")
    @pytest.mark.clustertest
    def test_cluster_list_all_pythonic(self, cluster):
        original_username = rns_client.username
        new_username = (
            "test-org"
            if cluster.rns_address.startswith("/test-org/")
            else original_username
        )

        with org_friend_account(
            new_username=new_username,
            token=rns_client.token,
            original_username=original_username,
        ):
            # create dummy terminated cluster - set the daemon status to terminated, which will also
            # update the cluster status. Making sure that its name is in the same format as all testing clusters
            # (contains timestamp and uuid)
            terminated_cluster = rh.cluster(
                name=f"{create_folder_path()}_terminated-cluster",
                server_connection_type="ssh",
            ).save()
            set_daemon_and_cluster_status(
                terminated_cluster,
                daemon_status=RunhouseDaemonStatus.TERMINATED,
                cluster_status=ClusterStatus.TERMINATED,
            )

            all_clusters = Cluster.list(show_all=True).get("den_clusters", {})
            present_statuses = set(
                [den_cluster.get("Status") for den_cluster in all_clusters]
            )
            assert len(present_statuses) > 1
            assert ClusterStatus.RUNNING in present_statuses
            assert ClusterStatus.TERMINATED in present_statuses

            test_cluster = [
                den_cluster
                for den_cluster in all_clusters
                if den_cluster.get("Name") == cluster.name
            ][0]
            assert test_cluster.get("Status") == ClusterStatus.RUNNING

    @pytest.mark.level("local")
    @pytest.mark.clustertest
    def test_cluster_list_status_pythonic(self, cluster):
        original_username = rns_client.username
        new_username = (
            "test-org"
            if cluster.rns_address.startswith("/test-org/")
            else original_username
        )

        with org_friend_account(
            new_username=new_username,
            token=rns_client.token,
            original_username=original_username,
        ):
            for status in [ClusterStatus.RUNNING, ClusterStatus.TERMINATED]:
                # check that filtered requests contains only specific status
                filtered_clusters = Cluster.list(status=status).get("den_clusters", {})
                if filtered_clusters:
                    filtered_statuses = set(
                        [cluster.get("Status") for cluster in filtered_clusters]
                    )
                    assert status in filtered_statuses

    @pytest.mark.level("local")
    @pytest.mark.clustertest
    def test_cluster_list_since_pythonic(self, cluster):
        cluster.save()  # tls exposed local cluster is not saved by default

        original_username = rns_client.username
        new_username = (
            "test-org"
            if cluster.rns_address.startswith("/test-org/")
            else original_username
        )

        with org_friend_account(
            new_username=new_username,
            token=rns_client.token,
            original_username=original_username,
        ):
            minutes_time_filter = 10
            clusters = Cluster.list(since=f"{minutes_time_filter}m")
            recent_clusters = clusters.get("den_clusters", {})

            clusters_last_active_timestamps = set(
                [
                    den_cluster.get("Last Active (UTC)")
                    for den_cluster in recent_clusters
                ]
            )

            assert len(clusters_last_active_timestamps) >= 1
            current_utc_time = datetime.utcnow().replace(tzinfo=timezone.utc)

            for timestamp in clusters_last_active_timestamps:
                # Convert the timestamp string to a naive datetime object
                timestamp_obj = datetime.strptime(timestamp, "%m/%d/%Y, %H:%M:%S")
                timestamp_obj = timestamp_obj.replace(tzinfo=timezone.utc)
                assert (
                    current_utc_time - timestamp_obj
                ).total_seconds() <= minutes_time_filter * MINUTE

    @pytest.mark.level("local")
    @pytest.mark.clustertest
    def test_cluster_list_cmd_output_no_filters(self, capsys, cluster):
        import re
        import subprocess

        original_username = rns_client.username
        new_username = (
            "test-org"
            if cluster.rns_address.startswith("/test-org/")
            else original_username
        )

        with org_friend_account(
            new_username=new_username,
            token=rns_client.token,
            original_username=original_username,
        ):
            env = set_output_env_vars()

            process = subprocess.Popen(
                "runhouse cluster list",
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
            )
            process.wait()
            stdout = process.communicate()[0]
            capsys.readouterr()
            cmd_stdout = stdout.decode("utf-8")

            assert cmd_stdout

            # The output is printed as a table.
            # testing that the table name is printed correctly
            regex = f".*Clusters for {rh.configs.username}.*\(Running: .*/.*, Total Displayed: .*/.*\).*"
            assert re.search(regex, cmd_stdout)

            # testing that the table column names is printed correctly
            col_names = ["┃ Name", "┃ Cluster Type", "┃ Status", "┃ Last Active (UTC)"]
            for name in col_names:
                assert name in cmd_stdout
            assert (
                f"Showing clusters that were active in the last {int(LAST_ACTIVE_AT_TIMEFRAME / HOUR)} hours."
                in cmd_stdout
            )
            assert cluster.name in cmd_stdout

    @pytest.mark.level("local")
    @pytest.mark.clustertest
    def test_cluster_list_cmd_output_with_filters(self, capsys, cluster):

        import re
        import subprocess

        original_username = rns_client.username
        new_username = (
            "test-org"
            if cluster.rns_address.startswith("/test-org/")
            else original_username
        )

        with org_friend_account(
            new_username=new_username,
            token=rns_client.token,
            original_username=original_username,
        ):
            cluster.save()  # tls exposed local cluster is not saved by default

            subprocess_env = set_output_env_vars()

            for status in [ClusterStatus.RUNNING, ClusterStatus.TERMINATED]:
                process = subprocess.Popen(
                    f"runhouse cluster list --status {status}",
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=subprocess_env,
                )
                process.wait()
                stdout = process.communicate()[0]
                capsys.readouterr()
                cmd_stdout = stdout.decode("utf-8")
                assert cmd_stdout

                # The output is printed as a table.
                # testing that the table name is printed correctly

                regex = ".*Clusters for.*\(Running: .*/.*, Total Displayed: .*/.*\).*"
                assert re.search(regex, cmd_stdout)

                # testing that the table column names is printed correctly
                col_names = [
                    "┃ Name",
                    "┃ Cluster Type",
                    "┃ Status",
                    "┃ Last Active (UTC)",
                ]
                for name in col_names:
                    assert name in cmd_stdout

                assert (
                    "Note: the above clusters have registered activity in the last 24 hours."
                    not in cmd_stdout
                )

                if status == ClusterStatus.RUNNING:
                    assert cluster.name in cmd_stdout

                # Check other statuses not found in output
                cmd_stdout = cmd_stdout.replace("Running:", "")
                statuses = [s.lower() for s in list(ClusterStatus.__members__.keys())]
                statuses.remove(status)

                for status in statuses:
                    assert status.capitalize() not in cmd_stdout

    @pytest.mark.level("local")
    @pytest.mark.clustertest
    def test_cluster_list_and_create_process(self, cluster):
        assert DEFAULT_PROCESS_NAME in cluster.list_processes()

        # create a process manually with the create_process functionality
        cluster.ensure_process_created(name="new_test_process_created_with_utility")
        assert "new_test_process_created_with_utility" in cluster.list_processes()

    @pytest.mark.level("local")
    @pytest.mark.clustertest
    def test_cluster_run_bash_in_process(self, cluster):
        process = cluster.ensure_process_created(name="test_process")
        res = cluster.run_bash("echo hello", process=process)
        assert res[0] == 0
        assert res[1].strip() == "hello"

    @pytest.mark.level("local")
    @pytest.mark.clustertest
    def test_cluster_kill_process(self, cluster):
        process = cluster.ensure_process_created(name="new_test_process")
        assert process in cluster.list_processes()
        cluster.put(key="new_key", obj="val", process=process)
        assert cluster.get("new_key") == "val"
        cluster.kill_process(process)
        assert cluster.get("new_key") is None
        assert process not in cluster.list_processes()
