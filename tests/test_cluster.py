import unittest

from runhouse.rns.hardware import cluster, OnDemandCluster


def test_cluster_config():
    rh_cpu = cluster(name="^rh-cpu", dryrun=False)
    if not rh_cpu.is_up():
        rh_cpu.up()
    config = rh_cpu.config_for_rns
    cluster2 = OnDemandCluster.from_config(config)
    assert cluster2.address == rh_cpu.address


def test_cluster_sharing():
    c = cluster(name="^rh-cpu").up_if_not().save()
    c.share(users=["donny@run.house", "josh@run.house"], access_type="write")
    assert True


def test_read_shared_cluster():
    c = cluster(name="/jlewitt1/rh-cpu")
    res = c.run_python(["import numpy", "print(numpy.__version__)"])
    assert res[0][1]


def test_install():
    c = cluster(name="^rh-cpu")
    c.install_packages(
        [
            "./",
            "torch==1.12.1",
            # 'conda:jupyterlab',  # TODO [DG] make this actually work
            # 'gh:pytorch/vision'  # TODO [DG] make this actually work
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


if __name__ == "__main__":
    unittest.main()
