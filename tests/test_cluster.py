import os
import unittest

import pytest

import runhouse as rh

from runhouse.rns.hardware import OnDemandCluster


def summer(a, b):
    return a + b


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
def test_install(cpu_cluster):
    cpu_cluster.install_packages(
        [
            "./",
            "torch==1.12.1",
            # 'conda:jupyterlab',
            # 'gh:pytorch/vision'
        ]
    )


@pytest.mark.clustertest
def test_basic_run(cpu_cluster):
    # Create temp file where fn's will be stored
    test_cmd = "echo hi"
    cpu_cluster.up_if_not()
    res = cpu_cluster.run(commands=[test_cmd])
    assert "hi" in res[0][1]


@pytest.mark.clustertest
def test_restart_grpc(cpu_cluster):
    cpu_cluster.up_if_not()
    codes = cpu_cluster.restart_grpc_server(resync_rh=False)
    assert codes


@pytest.mark.clustertest
def test_on_same_cluster(cpu_cluster):
    hw_copy = cpu_cluster.copy()
    cpu_cluster.up_if_not()

    func_hw = rh.function(is_on_cluster).to(cpu_cluster)
    assert func_hw(cpu_cluster)
    assert func_hw(hw_copy)


@pytest.mark.clustertest
def test_on_diff_cluster(cpu_cluster, cpu_cluster_2):
    diff_hw = cpu_cluster_2

    func_hw = rh.function(is_on_cluster).to(cpu_cluster)
    assert not func_hw(diff_hw)


@pytest.mark.slurmtest
def test_submit_job_to_slurm_via_rest():
    # Initialize Slurm cluster with REST API attributes
    sc = rh.cluster(
        name="my_slurm_cluster",
        api_url=os.getenv("SLURM_URL"),
        api_auth_user=os.getenv("SLURM_USER"),
        api_jwt_token=os.getenv("SLURM_JWT"),
    )

    job_payload = {
        "job": {
            "name": "test",
            "ntasks": 1,
            "nodes": 1,
            "current_working_directory": "/home/ubuntu/tests",
            "standard_input": "/dev/null",
            "standard_output": "/home/ubuntu/tests/%j.out",
            "standard_error": "/home/ubuntu/tests/%j.out",
            "environment": {
                "PATH": "/bin:/usr/bin/:/usr/local/bin/",
                "LD_LIBRARY_PATH": "/lib/:/lib64/:/usr/local/lib",
            },
        },
        "script": "#!/bin/bash\necho 'Hello world, I am running on node' $HOSTNAME\nsleep 10\ndate",
    }

    job_id = sc.submit_job(payload=job_payload)

    assert isinstance(job_id, int)


@pytest.mark.slurmtest
def test_submit_job_to_slurm_via_ssh():
    # Initialize Slurm cluster with SSH attributes
    sc = rh.cluster(
        name="my_slurm_cluster",
        ssh_creds={
            "ssh_user": "ubuntu",
            "ssh_private_key": "~/.ssh/runhouse-auth.pem",
        },
        ips=[os.getenv("SLURM_NODE_IP")],
        log_folder="tests",
        partition="rhcluster",
    )

    job_id = sc.submit_job(summer, a=1, b=2)

    assert isinstance(job_id, int)


if __name__ == "__main__":
    unittest.main()
