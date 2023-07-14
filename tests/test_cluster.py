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
def test_submit_job_to_slurm(slurm_cluster):
    # Run a slurm command on the jumpbox to submit a job
    job_name = "test_job"
    sample_command = f"""srun --job-name={job_name} bash -c "echo Hello, world!" && hostname && sleep 10"""
    worker_ips = slurm_cluster.submit_job(job_name=job_name, commands=[sample_command])
    assert worker_ips

    print(f"Submitted job, running on worker nodes with IPs: {', '.join(worker_ips)}")

    slurm_cluster.ssh_tunnel_to_target_host(target_host=worker_ips[0])

    stdout, stderr = slurm_cluster.run_commands_on_target_host(["ls -l"])
    assert stdout


@unittest.skip("Not implemented yet")
@pytest.mark.slurmtest
def test_get_slurm_job_result(slurm_cluster, request):
    job_id = request.config.cache.get("job_id", None)
    res = slurm_cluster.result(job_id)
    assert int(res) == 3


@unittest.skip("Not implemented yet")
@pytest.mark.slurmtest
def test_get_slurm_stdout(slurm_cluster, request):
    job_id = request.config.cache.get("job_id", None)
    stdout = slurm_cluster.stdout(job_id)
    assert "3" in stdout


@unittest.skip("Not implemented yet")
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
