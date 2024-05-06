import subprocess

import pandas as pd
import pytest
import requests

import runhouse as rh

from runhouse.constants import (
    DEFAULT_HTTP_PORT,
    DEFAULT_HTTPS_PORT,
    DEFAULT_SERVER_PORT,
    LOCALHOST,
)

import tests.test_resources.test_resource
from tests.conftest import init_args
from tests.utils import friend_account_in_org, get_random_str, remove_config_keys

""" TODO:
1) In subclasses, test factory methods create same type as parent
2) In subclasses, use monkeypatching to make sure `up()` is called for various methods if the server is not up
3) Test AWS, GCP, and Azure static clusters separately
"""


def load_shared_resource_config(resource_class_name, address):
    resource_class = getattr(rh, resource_class_name)
    loaded_resource = resource_class.from_name(address, dryrun=True)
    return loaded_resource.config()


def save_resource_and_return_config():
    df = pd.DataFrame(
        {"id": [1, 2, 3, 4, 5, 6], "grade": ["a", "b", "b", "a", "a", "e"]}
    )
    table = rh.table(df, name="test_table")
    return table.config()


def test_table_to_rh_here():
    df = pd.DataFrame(
        {"id": [1, 2, 3, 4, 5, 6], "grade": ["a", "b", "b", "a", "a", "e"]}
    )
    rh.table(df, name="test_table").to(rh.here)
    assert rh.here.get("test_table") is not None


def summer(a: int, b: int):
    return a + b


def sub(a: int, b: int):
    return a - b


def cluster_keys(cluster):
    return cluster.keys()


def cluster_config():
    return rh.here.config()


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
        ]
    }
    MINIMAL = {"cluster": ["static_cpu_cluster"]}
    RELEASE = {
        "cluster": [
            "static_cpu_cluster",
            "password_cluster",
        ]
    }
    MAXIMAL = {
        "cluster": [
            "docker_cluster_pk_ssh_no_auth",
            "docker_cluster_pk_ssh_den_auth",
            "docker_cluster_pwd_ssh_no_auth",
            "static_cpu_cluster",
            "password_cluster",
            "multinode_cpu_cluster",
        ]
    }

    @pytest.mark.level("unit")
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
    def test_docker_cluster_fixture_is_logged_out(self, docker_cluster_pk_ssh_no_auth):
        save_resource_and_return_config_cluster = rh.function(
            save_resource_and_return_config,
            name="save_resource_and_return_config_cluster",
        ).to(
            system=docker_cluster_pk_ssh_no_auth,
        )
        saved_config_on_cluster = save_resource_and_return_config_cluster()
        # This cluster was created without any logged in Runhouse config. Make sure that the simple resource
        # created on the cluster starts with "~", which is the prefix that local Runhouse configs are saved with.
        assert ("/" not in saved_config_on_cluster["name"]) or (
            saved_config_on_cluster["name"].startswith("~")
        )

    @pytest.mark.level("local")
    def test_cluster_recreate(self, cluster):
        # Create underlying ssh connection if not already
        cluster.run(["echo hello"])
        num_open_tunnels = len(rh.globals.sky_ssh_runner_cache)

        # Create a new cluster object for the same remote cluster
        cluster.save()
        new_cluster = rh.cluster(cluster.rns_address)
        new_cluster.run(["echo hello"])
        # Check that the same underlying ssh connection was used
        assert len(rh.globals.sky_ssh_runner_cache) == num_open_tunnels

    @pytest.mark.level("local")
    def test_cluster_endpoint(self, cluster):
        if not cluster.address:
            assert cluster.endpoint() is None
            return

        endpoint = cluster.endpoint()
        if cluster.server_connection_type in ["ssh", "aws_ssm"]:
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
        assert r.json()["resource_type"] == "cluster"

    @pytest.mark.level("local")
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
    def test_cluster_delete_env(self, cluster):
        env1 = rh.env(reqs=[], working_dir="./", name="env1").to(cluster)
        env2 = rh.env(reqs=[], working_dir="./", name="env2").to(cluster)
        env3 = rh.env(reqs=[], working_dir="./", name="env3")

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
    @pytest.mark.skip(reason="TODO")
    def test_rh_here_objects(self, cluster):
        save_test_table_remote = rh.function(test_table_to_rh_here, system=cluster)
        save_test_table_remote()
        assert "test_table" in cluster.keys()
        assert isinstance(cluster.get("test_table"), rh.Table)

    @pytest.mark.level("local")
    def test_rh_status_pythonic(self, cluster):
        cluster.put(key="status_key1", obj="status_value1", env="numpy_env")
        res = cluster.status()
        assert res.get("creds") is None
        assert res.get("server_port") == (cluster.server_port or DEFAULT_SERVER_PORT)
        assert res.get("server_connection_type") == cluster.server_connection_type
        assert res.get("den_auth") == cluster.den_auth
        assert res.get("resource_type") == cluster.RESOURCE_TYPE
        assert res.get("ips") == cluster.ips
        assert "numpy_env" in res.get("envs")
        assert {"name": "status_key1", "resource_type": "str"} in res.get("envs")[
            "numpy_env"
        ]

    @pytest.mark.level("local")
    def test_rh_status_cli_in_cluster(self, cluster):
        default_env_name = cluster.default_env.name

        cluster.put(key="status_key2", obj="status_value2")
        status_output_string = cluster.run(
            ["runhouse status"], _ssh_mode="non_interactive"
        )[0][1]
        # The string that's returned is utf-8 with the literal escape characters mixed in.
        # We need to convert the escape characters to their actual values to compare the strings.
        status_output_string = status_output_string.encode("utf-8").decode(
            "unicode_escape"
        )
        status_output_string = status_output_string.replace("\n", "")
        assert "Runhouse Daemon is running" in status_output_string
        assert f"server_port: {cluster.server_port}" in status_output_string
        assert (
            f"server_connection_type: {cluster.server_connection_type}"
            in status_output_string
        )
        assert f"den_auth: {str(cluster.den_auth)}" in status_output_string
        assert f"resource_type: {cluster.RESOURCE_TYPE.lower()}" in status_output_string
        assert f"ips: {str(cluster.ips)}" in status_output_string
        assert "Serving " in status_output_string
        assert f"{default_env_name}" in status_output_string
        assert "status_key2 (str)" in status_output_string
        assert "creds" not in status_output_string

    @pytest.mark.skip("Restarting the server mid-test causes some errors, need to fix")
    @pytest.mark.level("local")
    def test_rh_status_cli_not_in_cluster(self, cluster):
        # TODO -- check this base_env
        cluster.put(key="status_key3", obj="status_value3")
        res = str(
            subprocess.check_output(["runhouse", "status", f"{cluster.name}"]), "utf-8"
        )
        assert "😈 Runhouse Daemon is running 🏃" in res
        assert f"server_port: {cluster.server_port}" in res
        assert f"server_connection_type: {cluster.server_connection_type}" in res
        assert f"den_auth: {str(cluster.den_auth)}" in res
        assert f"resource_type: {cluster.RESOURCE_TYPE.lower()}" in res
        assert f"ips: {str(cluster.ips)}" in res
        assert "Serving 🍦 :" in res
        assert "base_env (runhouse.resources.envs.env.Env):" in res
        assert "status_key3 (str)" in res
        assert "ssh_certs" not in res

    @pytest.mark.skip("Restarting the server mid-test causes some errors, need to fix")
    @pytest.mark.level("local")
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
            users=["support@run.house"],
            access_level="read",
            notify_users=False,
        )

        # First try loading in same process/filesystem because it's more debuggable, but not as thorough
        resource_class_name = cluster.config().get("resource_type").capitalize()
        config = cluster.config()

        with friend_account_in_org():
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
        # work correclty and then the remote function will just call the object store when it calls .keys()
        assert cluster.keys() == cluster_keys_remote(cluster)

        # Restore the state?
        cluster.save(name=old_name)
