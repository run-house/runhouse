import copy
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
def test_restart_server(cpu_cluster):
    cpu_cluster.up_if_not()
    codes = cpu_cluster.restart_server(resync_rh=False)
    assert codes


@pytest.mark.clustertest
def test_on_same_cluster(cpu_cluster):
    hw_copy = copy.copy(cpu_cluster)
    cpu_cluster.up_if_not()

    func_hw = rh.function(is_on_cluster).to(cpu_cluster)
    assert func_hw(cpu_cluster)
    assert func_hw(hw_copy)


@pytest.mark.clustertest
def test_on_diff_cluster(cpu_cluster, byo_cpu):
    func_hw = rh.function(is_on_cluster).to(cpu_cluster)
    assert not func_hw(byo_cpu)


@pytest.mark.slurmtest
def test_submit_job_to_slurm(slurm_cluster, request):
    submitted_job = slurm_cluster.submit_job(summer, a=1, b=2)
    assert type(submitted_job).__name__ == "SlurmJob"

    job_id: str = submitted_job.job_id

    # Temporarily store job_id in pytest cache to reference this job id in subsequent tests
    request.config.cache.set("job_id", job_id)

    assert isinstance(job_id, str)


@pytest.mark.slurmtest
def test_get_slurm_job_result(slurm_cluster, request):
    job_id = request.config.cache.get("job_id", None)
    res = slurm_cluster.result(job_id)
    assert int(res) == 3


@pytest.mark.slurmtest
def test_get_slurm_stdout(slurm_cluster, request):
    job_id = request.config.cache.get("job_id", None)
    stdout = slurm_cluster.stdout(job_id)
    assert "3" in stdout


@pytest.mark.slurmtest
def test_get_slurm_stderr(slurm_cluster, request):
    job_id = request.config.cache.get("job_id", None)
    stderr = slurm_cluster.stderr(job_id)
    assert stderr == ""


@pytest.mark.rnstest
@pytest.mark.slurmtest
def test_reload_ssh_slurm_cluster_from_rns():
    sc = rh.SlurmCluster.from_name("ssh_slurm_cluster")
    assert isinstance(sc, rh.SlurmCluster)


if __name__ == "__main__":
    unittest.main()
