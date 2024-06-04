import pytest

import tests.test_resources.test_clusters.test_cluster
from tests.utils import friend_account


class TestOnDemandCluster(tests.test_resources.test_clusters.test_cluster.TestCluster):

    MAP_FIXTURES = {"resource": "cluster"}

    UNIT = {"cluster": []}
    LOCAL = {"cluster": []}
    MINIMAL = {
        "cluster": [
            "ondemand_aws_cluster",
            "ondemand_gcp_cluster",
        ]
    }
    RELEASE = {
        "cluster": [
            "ondemand_aws_cluster",
            "ondemand_gcp_cluster",
            # "ondemand_k8s_cluster",  # tested in test_kubernetes_cluster.py
            "ondemand_aws_https_cluster_with_auth",
        ]
    }
    MAXIMAL = {
        "cluster": [
            "ondemand_aws_cluster",
            "ondemand_gcp_cluster",
            "ondemand_k8s_cluster",
            "ondemand_aws_https_cluster_with_auth",
            "v100_gpu_cluster",
            "k80_gpu_cluster",
            "a10g_gpu_cluster",
            "static_cpu_cluster",
            "password_cluster",
            "multinode_cpu_cluster",
            "multinode_gpu_cluster",
        ]
    }

    @pytest.mark.level("minimal")
    def test_restart_does_not_change_config_yaml(self, cluster):
        assert cluster.up_if_not()
        config_yaml_res = cluster.run("cat ~/.rh/config.yaml")
        assert config_yaml_res[0][0] == 0
        config_yaml_content = config_yaml_res[0][1]

        cluster.share(
            users=["info@run.house"],
            access_level="read",
            notify_users=False,
        )

        with friend_account():
            cluster.restart_server()
            config_yaml_res_after_restart = cluster.run("cat ~/.rh/config.yaml")
            assert config_yaml_res_after_restart[0][0] == 0
            config_yaml_content_after_restart = config_yaml_res[0][1]
            assert config_yaml_content_after_restart == config_yaml_content
