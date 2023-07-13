import copy
import unittest

import pytest

import runhouse as rh
from runhouse.rns.hardware import OnDemandCluster

from .conftest import parametrize_cpu_clusters


def is_on_cluster(cluster):
    return cluster.on_this_cluster()


@pytest.mark.clustertest
def test_cluster_config(cpu_cluster):
    config = cpu_cluster.config_for_rns
    cluster2 = OnDemandCluster.from_config(config)
    assert cluster2.address == cpu_cluster.address


@pytest.mark.clustertest
@pytest.mark.rnstest
def test_cluster_sharing(cpu_cluster):
    cpu_cluster.share(
        users=["donny@run.house", "josh@run.house"],
        access_type="write",
        notify_users=False,
    )
    assert True


@pytest.mark.clustertest
def test_read_shared_cluster(cpu_cluster):
    res = cpu_cluster.run_python(["import numpy", "print(numpy.__version__)"])
    assert res[0][1]


@pytest.mark.clustertest
@parametrize_cpu_clusters
def test_install(cluster):
    cluster.install_packages(
        [
            "./",
            "torch==1.12.1",
            # 'conda:jupyterlab',
            # 'gh:pytorch/vision'
        ]
    )


@pytest.mark.clustertest
@parametrize_cpu_clusters
def test_basic_run(cluster):
    # Create temp file where fn's will be stored
    test_cmd = "echo hi"
    res = cluster.run(commands=[test_cmd])
    assert "hi" in res[0][1]


@pytest.mark.clustertest
@parametrize_cpu_clusters
def test_restart_server(cluster):
    cluster.up_if_not()
    codes = cluster.restart_server(resync_rh=False)
    assert codes


@pytest.mark.clustertest
@parametrize_cpu_clusters
def test_on_same_cluster(cluster):
    hw_copy = copy.copy(cluster)

    func_hw = rh.function(is_on_cluster).to(cluster)
    assert func_hw(cluster)
    assert func_hw(hw_copy)


@pytest.mark.clustertest
@parametrize_cpu_clusters
def test_on_diff_cluster(cluster, byo_cpu):
    func_hw = rh.function(is_on_cluster).to(cluster)
    assert not func_hw(byo_cpu)


if __name__ == "__main__":
    unittest.main()
