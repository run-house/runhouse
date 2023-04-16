import unittest

import runhouse as rh

from runhouse.rns.hardware import cluster, OnDemandCluster


def is_on_cluster(cluster):
    return cluster.on_this_cluster()


def test_cluster_config():
    rh_cpu = cluster(name="^rh-cpu")
    if not rh_cpu.is_up():
        rh_cpu.up()
    config = rh_cpu.config_for_rns
    cluster2 = OnDemandCluster.from_config(config)
    assert cluster2.address == rh_cpu.address


def test_cluster_sharing():
    c = cluster(name="^rh-cpu").up_if_not().save()
    c.share(
        users=["donny@run.house", "josh@run.house"],
        access_type="write",
        notify_users=False,
    )
    assert True


def test_read_shared_cluster():
    c = cluster(name="@/rh-cpu")
    res = c.run_python(["import numpy", "print(numpy.__version__)"])
    assert res[0][1]


def test_install():
    c = cluster(name="^rh-cpu")
    c.install_packages(
        [
            "./",
            "torch==1.12.1",
            # 'conda:jupyterlab',
            # 'gh:pytorch/vision'
        ]
    )


def test_basic_run():
    # Create temp file where fn's will be stored
    test_cmd = "echo hi"
    hw = cluster(name="^rh-cpu")
    hw.up_if_not()
    res = hw.run(commands=[test_cmd])
    assert "hi" in res[0][1]


def test_restart_grpc():
    hw = cluster(name="^rh-cpu")
    hw.up_if_not()
    codes = hw.restart_grpc_server(resync_rh=False)
    assert codes


def test_on_same_cluster():
    hw = cluster(name="^rh-cpu").up_if_not()
    hw.restart_grpc_server()
    hw.up_if_not()

    hw_copy = cluster(name="^rh-cpu")

    func_hw = rh.function(is_on_cluster).to(hw)
    assert func_hw(hw)
    assert func_hw(hw_copy)


def test_on_diff_cluster():
    hw = cluster(name="^rh-cpu").up_if_not()
    diff_hw = rh.cluster(name="test-byo-cluster").up_if_not()

    func_hw = rh.function(is_on_cluster).to(hw)
    assert not func_hw(diff_hw)


if __name__ == "__main__":
    unittest.main()
