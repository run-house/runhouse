import runhouse as rh

from tests.test_resources.test_resource import TestResource
from .conftest import (
    local_docker_cluster_passwd,
    local_docker_cluster_public_key,
    local_logged_out_docker_cluster,
    local_test_account_cluster_public_key,
    named_cluster,
    password_cluster,
    static_cpu_cluster,
    unnamed_cluster,
)

""" TODO:
1) In subclasses, test factory methods create same type as parent
2) In subclasses, use monkeypatching to make sure `up()` is called for various methods if the server is not up
3) Test AWS, GCP, and Azure static clusters separately
"""


class TestCluster(TestResource):

    UNIT = {"resource": [local_docker_cluster_public_key]}
    LOCAL = {
        "resource": [
            unnamed_cluster,
            named_cluster,
            local_docker_cluster_public_key,
            local_docker_cluster_passwd,
            local_logged_out_docker_cluster,
            local_test_account_cluster_public_key,
        ]
    }
    REMOTE = {"resource": [static_cpu_cluster]}
    FULL = {"resource": [unnamed_cluster, named_cluster, password_cluster]}
    ALL = {"resource": [unnamed_cluster, named_cluster, password_cluster]}

    def test_factory_methods(self, cluster):
        assert isinstance(cluster, Cluster)
        assert isinstance(cluster, Resource)
        assert isinstance(cluster, rh.Cluster)
        assert isinstance(cluster, rh.Resource)
        assert isinstance(cluster, rh.resources.cluster.Cluster)
