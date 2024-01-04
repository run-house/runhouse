import pytest

import runhouse as rh

from runhouse.resources.hardware import KubernetesCluster

import tests.test_resources.test_clusters.test_cluster

from tests.conftest import init_args


class TestKubernetesCluster(
    tests.test_resources.test_clusters.test_cluster.TestCluster
):

    MAP_FIXTURES = {"resource": "cluster"}

    UNIT = {"cluster": []}
    LOCAL = {"cluster": []}
    MINIMAL = {"cluster": ["kubernetes_cpu_cluster"]}

    # @pytest.mark.level("minimal")
    # def test_read_shared_k8s_cluster(kubernetes_cpu_cluster):
    #     res = kubernetes_cpu_cluster.run_python(["import numpy", "print(numpy.__version__)"])
    #     assert res[0][1]

    # # def num_cpus():
    # #     import multiprocessing
    # #     return f"Num cpus: {multiprocessing.cpu_count()}"

    # @pytest.mark.level("minimal")
    # def test_function_on_k8s_cluster(kubernetes_cpu_cluster):

    #     def num_cpus():
    #         import multiprocessing
    #         return f"Num cpus: {multiprocessing.cpu_count()}"

    #     num_cpus_cluster = rh.function(name="num_cpus_cluster", fn=num_cpus).to(system=kubernetes_cpu_cluster, reqs=["./"])
    #     assert num_cpus_cluster() == 'Num cpus: 4'
