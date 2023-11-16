import runhouse as rh

import tests.test_resources.test_resource
from tests.conftest import init_args

from .conftest import (
    local_docker_cluster_passwd,
    local_docker_cluster_public_key,
    local_logged_out_docker_cluster,
    # local_test_account_cluster_public_key,
    named_cluster,
    password_cluster,
    static_cpu_cluster,
)

""" TODO:
1) In subclasses, test factory methods create same type as parent
2) In subclasses, use monkeypatching to make sure `up()` is called for various methods if the server is not up
3) Test AWS, GCP, and Azure static clusters separately
"""


class TestCluster(tests.test_resources.test_resource.TestResource):

    MAP_FIXTURES = {"resource": "cluster"}

    UNIT = {"cluster": [local_docker_cluster_public_key]}
    LOCAL = {
        "cluster": [
            local_docker_cluster_public_key,
            local_docker_cluster_passwd,
            local_logged_out_docker_cluster,
            # local_test_account_cluster_public_key,
        ]
    }
    MINIMAL = {"cluster": [static_cpu_cluster]}
    THOROUGH = {"cluster": [named_cluster, password_cluster]}
    MAXIMAL = {"cluster": [named_cluster, password_cluster]}

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
