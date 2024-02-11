import tests.test_resources.test_clusters.test_cluster


class TestOnDemandCluster(tests.test_resources.test_clusters.test_cluster.TestCluster):

    MAP_FIXTURES = {"resource": "cluster"}

    UNIT = {"cluster": []}
    LOCAL = {"cluster": []}
    MINIMAL = {"cluster": ["ondemand_cpu_cluster"]}
    THOROUGH = {"cluster": ["ondemand_cpu_cluster", "ondemand_https_cluster_with_auth"]}
    MAXIMAL = {
        "cluster": [
            "ondemand_cpu_cluster",
            "v100_gpu_cluster",
            "k80_gpu_cluster",
            "a10g_gpu_cluster",
            "static_cpu_cluster",
            "password_cluster",
            "multinode_cpu_cluster",
        ]
    }

    def test_sasha(self):
        import runhouse as rh

        my_c = rh.ondemand_cluster(name="sasha-ondemand-cluster")
        my_c.up_if_not()
        a = my_c.status()
        ips = a["handle"].stable_internal_external_ips
        assert len(ips) > 0
        b = my_c.name
        assert b == "sasha-ondemand-cluster"
        my_c.save()
