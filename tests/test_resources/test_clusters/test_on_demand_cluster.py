import tests.test_resources.test_clusters.test_cluster


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
        ]
    }
