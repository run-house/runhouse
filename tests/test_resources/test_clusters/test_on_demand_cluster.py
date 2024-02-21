import tests.test_resources.test_clusters.test_cluster


class TestOnDemandCluster(tests.test_resources.test_clusters.test_cluster.TestCluster):

    MAP_FIXTURES = {"resource": "cluster"}

    UNIT = {"cluster": []}
    LOCAL = {"cluster": []}
    MAXIMAL = {
        "cluster": [
            "ondemand_aws_cluster",
            "ondemand_gcp_cluster",
            "v100_gpu_cluster",
            "k80_gpu_cluster",
            "a10g_gpu_cluster",
            "static_cpu_cluster",
            "password_cluster",
            "multinode_cpu_cluster",
        ]
    }
