import pytest
import requests

import runhouse as rh

import tests.test_resources.test_resource
from tests.conftest import init_args

""" TODO:
1) In subclasses, test factory methods create same type as parent
2) In subclasses, use monkeypatching to make sure `up()` is called for various methods if the server is not up
3) Test AWS, GCP, and Azure static clusters separately
"""


def save_resource_and_return_config():
    import pandas as pd

    df = pd.DataFrame(
        {"id": [1, 2, 3, 4, 5, 6], "grade": ["a", "b", "b", "a", "a", "e"]}
    )
    table = rh.table(df, name="test_table")
    return table.config_for_rns


class TestCluster(tests.test_resources.test_resource.TestResource):

    MAP_FIXTURES = {"resource": "cluster"}

    UNIT = {"cluster": ["named_cluster"]}
    LOCAL = {
        "cluster": [
            "local_docker_cluster_public_key_logged_in",
            "local_docker_cluster_public_key_logged_out",
            "local_docker_cluster_passwd",
        ]
    }
    MINIMAL = {"cluster": ["static_cpu_cluster"]}
    THOROUGH = {"cluster": ["static_cpu_cluster", "password_cluster"]}
    MAXIMAL = {
        "cluster": [
            "local_docker_cluster_public_key_logged_in",
            "local_docker_cluster_public_key_logged_out",
            "local_docker_cluster_telemetry_public_key",
            "local_docker_cluster_passwd",
            "static_cpu_cluster",
            "password_cluster",
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
            assert cluster.ssh_creds() == args["ssh_creds"]

        if "server_host" in args:
            assert cluster.server_host == args["server_host"]
        else:
            # TODO: Test default behavior
            pass

        if "server_port" in args:
            assert cluster.server_port == args["server_port"]
        else:
            # TODO: Test default behavior
            pass

        if "server_connection_type" in args:
            assert cluster.server_connection_type == args["server_connection_type"]
        else:
            # TODO: Test default behavior
            assert cluster.server_connection_type == "ssh"

        if "ssl_keyfile" in args:
            assert cluster.cert_config.key_path == args["ssl_keyfile"]

        if "ssl_certfile" in args:
            assert cluster.cert_config.cert_path == args["ssl_certfile"]

        if "den_auth" in args:
            assert cluster.den_auth == args["den_auth"]
        else:
            # TODO: Test default behavior
            pass

    @pytest.mark.level("local")
    def test_logged_in_local_cluster(self, local_docker_cluster_public_key_logged_in):
        save_resource_and_return_config_cluster = rh.function(
            save_resource_and_return_config,
            name="save_resource_and_return_config_cluster",
            system=local_docker_cluster_public_key_logged_in,
        )
        saved_config_on_cluster = save_resource_and_return_config_cluster()
        # This cluster was created using our own logged in Runhouse config. Make sure that the simple resource
        # created on the cluster starts with /default_folder, e.g. /rohinb2/<resource>s
        assert saved_config_on_cluster["name"].startswith(
            rh.configs.defaults_cache["default_folder"]
        )

    @pytest.mark.level("local")
    def test_logged_out_local_cluster(self, local_docker_cluster_public_key_logged_out):
        save_resource_and_return_config_cluster = rh.function(
            save_resource_and_return_config,
            name="save_resource_and_return_config_cluster",
            system=local_docker_cluster_public_key_logged_out,
        )
        saved_config_on_cluster = save_resource_and_return_config_cluster()
        # This cluster was created without any logged in Runhouse config. Make sure that the simple resource
        # created on the cluster starts with "~", which is the prefix that local Runhouse configs are saved with.
        assert saved_config_on_cluster["name"].startswith("~")

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
            assert endpoint == f"http://{rh.Cluster.LOCALHOST}:{cluster.client_port}"
        else:
            url_base = "https" if cluster.server_connection_type == "tls" else "http"
            assert endpoint == f"{url_base}://{cluster.address}:{cluster.server_port}"

        # Try to curl docs
        r = requests.get(
            f"{endpoint}/docs",
            verify=False,
            headers=rh.globals.rns_client.request_headers,
        )
        assert r.status_code == 200
        assert "FastAPI" in r.text
