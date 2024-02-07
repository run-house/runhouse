import subprocess

import pandas as pd
import pytest
import requests

import runhouse as rh

from runhouse.constants import LOCALHOST

import tests.test_resources.test_resource
from tests.conftest import init_args
from tests.utils import get_random_str

""" TODO:
1) In subclasses, test factory methods create same type as parent
2) In subclasses, use monkeypatching to make sure `up()` is called for various methods if the server is not up
3) Test AWS, GCP, and Azure static clusters separately
"""


def save_resource_and_return_config():
    df = pd.DataFrame(
        {"id": [1, 2, 3, 4, 5, 6], "grade": ["a", "b", "b", "a", "a", "e"]}
    )
    table = rh.table(df, name="test_table")
    return table.config_for_rns


def test_table_to_rh_here():
    df = pd.DataFrame(
        {"id": [1, 2, 3, 4, 5, 6], "grade": ["a", "b", "b", "a", "a", "e"]}
    )
    rh.table(df, name="test_table").to(rh.here)
    assert rh.here.get("test_table") is not None


class TestCluster(tests.test_resources.test_resource.TestResource):
    MAP_FIXTURES = {"resource": "cluster"}

    UNIT = {"cluster": ["named_cluster"]}
    LOCAL = {
        "cluster": [
            "docker_cluster_pk_ssh_no_auth",
            "docker_cluster_pk_ssh_den_auth",
            "docker_cluster_pwd_ssh_no_auth",
        ]
    }
    MINIMAL = {"cluster": ["static_cpu_cluster"]}
    THOROUGH = {
        "cluster": ["static_cpu_cluster", "password_cluster", "multinode_cpu_cluster"]
    }
    MAXIMAL = {
        "cluster": [
            "docker_cluster_pk_ssh_no_auth",
            "docker_cluster_pk_ssh_den_auth",
            "docker_cluster_pwd_ssh_no_auth",
            "docker_cluster_pk_ssh_telemetry",
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
            assert cluster.ssh_creds == args["ssh_creds"]

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
        num_open_tunnels = len(rh.globals.ssh_tunnel_cache)

        # Create a new cluster object for the same remote cluster
        cluster.save()
        new_cluster = rh.cluster(cluster.name)
        new_cluster.run(["echo hello"])
        # Check that the same underlying ssh connection was used
        assert len(rh.globals.ssh_tunnel_cache) == num_open_tunnels

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
            assert endpoint == f"{url_base}://{cluster.address}:{cluster.server_port}"

        # Try to curl docs
        r = requests.get(
            f"{endpoint}/docs",
            verify=False,
            headers=rh.globals.rns_client.request_headers(),
        )
        assert r.status_code == 200
        assert "FastAPI" in r.text

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
        env1 = rh.env(reqs=["numpy"], name="env1").to(cluster)
        env2 = rh.env(reqs=["numpy"], name="env2").to(cluster)
        env3 = rh.env(reqs=["numpy"], name="env3")

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
        assert res.get("server_port") == cluster.server_port
        assert res.get("server_connection_type") == cluster.server_connection_type
        assert res.get("den_auth") == cluster.den_auth
        assert res.get("resource_type") == cluster.RESOURCE_TYPE
        assert res.get("ips") == cluster.ips
        assert "numpy_env" in res.get("envs")
        assert {"name": "status_value1", "resource_type": "str"} in res.get("envs")[
            "numpy_env"
        ]

    @pytest.mark.level("local")
    def test_rh_status_cli(self, cluster):
        cluster.put(key="status_key2", obj="status_value2", env="base_env")
        res = cluster.run(["runhouse status"])[0][1]
        assert "😈 Runhouse Daemon is running 🏃" in res
        assert f"server_port: {cluster.server_port}" in res
        assert f"server_connection_type: {cluster.server_connection_type}" in res
        assert f"den_auth: {str(cluster.den_auth)}" in res
        assert f"resource_type: {cluster.RESOURCE_TYPE.lower()}" in res
        assert f"ips: {str(cluster.ips)}" in res
        assert "Serving 🍦 :" in res
        assert "base_env (runhouse.resources.envs.env.Env):" in res
        assert "status_value2 (str)" in res

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
