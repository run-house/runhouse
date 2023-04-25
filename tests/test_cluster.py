import unittest

import pytest

import runhouse as rh

from runhouse.rns.hardware import OnDemandCluster


def is_on_cluster(cluster):
    return cluster.on_this_cluster()


@pytest.mark.clustertest
def test_cluster_config(cpu):
    config = cpu.config_for_rns
    cluster2 = OnDemandCluster.from_config(config)
    assert cluster2.address == cpu.address


@pytest.mark.clustertest
def test_cluster_sharing(cpu):
    cpu.share(
        users=["donny@run.house", "josh@run.house"],
        access_type="write",
        notify_users=False,
    )
    assert True


@pytest.mark.clustertest
def test_read_shared_cluster(cpu):
    res = cpu.run_python(["import numpy", "print(numpy.__version__)"])
    assert res[0][1]


@pytest.mark.clustertest
def test_install(cpu):
    cpu.install_packages(
        [
            "./",
            "torch==1.12.1",
            # 'conda:jupyterlab',
            # 'gh:pytorch/vision'
        ]
    )


@pytest.mark.clustertest
def test_basic_run(cpu):
    # Create temp file where fn's will be stored
    test_cmd = "echo hi"
    cpu.up_if_not()
    res = cpu.run(commands=[test_cmd])
    assert "hi" in res[0][1]


@pytest.mark.clustertest
def test_restart_grpc(cpu):
    cpu.up_if_not()
    codes = cpu.restart_grpc_server(resync_rh=False)
    assert codes


@pytest.mark.clustertest
def test_on_same_cluster(cpu):
    hw_copy = cpu.copy()
    cpu.restart_grpc_server()
    cpu.up_if_not()

    func_hw = rh.function(is_on_cluster).to(cpu)
    assert func_hw(cpu)
    assert func_hw(hw_copy)


@pytest.mark.clustertest
def test_on_diff_cluster(cpu, a10g):
    diff_hw = a10g

    func_hw = rh.function(is_on_cluster).to(cpu)
    assert not func_hw(diff_hw)


if __name__ == "__main__":
    unittest.main()
