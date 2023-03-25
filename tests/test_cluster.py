import os
import unittest

import runhouse as rh

from runhouse.rns.hardware import cluster, OnDemandCluster


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
    c = cluster(name="/jlewitt1/rh-cpu")
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


def test_same_cluster():
    hw = cluster(name="^rh-cpu")
    hw.up_if_not()

    hw_copy = cluster(name="^rh-cpu")

    def dummy_func(a):
        return a

    func_hw = rh.function(dummy_func).to(hw)
    assert hw.on_same_cluster(func_hw)
    assert hw_copy.on_same_cluster(func_hw)


def test_diff_cluster():
    hw = cluster(name="^rh-cpu")
    hw.up_if_not()

    def dummy_func(a):
        return a

    func_hw = rh.function(dummy_func).to(hw)

    new_hw = cluster(name="diff-cpu")
    assert not new_hw.on_same_cluster(func_hw)


def test_submit_job_on_slurm_cluster():
    sc = rh.cluster(
        name="my_slurm_cluster",
        url=os.getenv("SLURM_URL"),
        auth_user=os.getenv("SLURM_USER"),
        jwt_token=os.getenv("SLURM_JWT"),
    ).save()

    job_payload = {
        "job": {
            "name": "test",
            "ntasks": 1,
            "nodes": 1,
            "current_working_directory": "/home/ubuntu/test",
            "standard_input": "/dev/null",
            "standard_output": "/home/ubuntu/test/test.out",
            "standard_error": "/home/ubuntu/test/test_error.out",
            "environment": {
                "PATH": "/bin:/usr/bin/:/usr/local/bin/",
                "LD_LIBRARY_PATH": "/lib/:/lib64/:/usr/local/lib",
            },
        },
        "script": "#!/bin/bash\necho 'Hello world, I am running on node' $HOSTNAME\nsleep 10\ndate",
    }
    node_ip = sc.submit_job(payload=job_payload)
    assert node_ip


if __name__ == "__main__":
    unittest.main()
