import copy
import os
import unittest
from typing import List

import pytest

import runhouse as rh
from runhouse.rns.hardware import OnDemandCluster

from .conftest import cpu_clusters, sagemaker_clusters, summer


def is_on_cluster(cluster):
    return cluster.on_this_cluster()


def np_array(num_list: List[int]):
    import numpy as np

    return np.array(num_list)


def sd_generate_image(prompt):
    from diffusers import StableDiffusionPipeline

    model = StableDiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-base"
    ).to("cuda")
    return model(prompt).images[0]


@pytest.mark.clustertest
def test_cluster_config(ondemand_cpu_cluster):
    config = ondemand_cpu_cluster.config_for_rns
    cluster2 = OnDemandCluster.from_config(config)
    assert cluster2.address == ondemand_cpu_cluster.address


@pytest.mark.clustertest
@pytest.mark.rnstest
def test_cluster_sharing(ondemand_cpu_cluster):
    ondemand_cpu_cluster.share(
        users=["donny@run.house", "josh@run.house"],
        access_type="write",
        notify_users=False,
    )
    assert True


@pytest.mark.clustertest
def test_read_shared_cluster(ondemand_cpu_cluster):
    res = ondemand_cpu_cluster.run_python(["import numpy", "print(numpy.__version__)"])
    assert res[0][1]


@pytest.mark.clustertest
@cpu_clusters
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
@cpu_clusters
def test_basic_run(cluster):
    # Create temp file where fn's will be stored
    test_cmd = "echo hi"
    res = cluster.run(commands=[test_cmd])
    assert "hi" in res[0][1]


@pytest.mark.clustertest
@cpu_clusters
def test_restart_server(cluster):
    cluster.up_if_not()
    codes = cluster.restart_server(resync_rh=False)
    assert codes


@pytest.mark.clustertest
@cpu_clusters
def test_on_same_cluster(cluster):
    hw_copy = copy.copy(cluster)

    func_hw = rh.function(is_on_cluster).to(cluster)
    assert func_hw(cluster)
    assert func_hw(hw_copy)


@pytest.mark.clustertest
def test_on_diff_cluster(ondemand_cpu_cluster, byo_cpu):
    func_hw = rh.function(is_on_cluster).to(ondemand_cpu_cluster)
    assert not func_hw(byo_cpu)


@pytest.mark.clustertest
def test_byo_cluster(byo_cpu, local_folder):
    assert byo_cpu.is_up()

    summer_func = rh.function(summer).to(byo_cpu)
    assert summer_func(1, 2) == 3

    byo_cpu.put("test_obj", list(range(10)))
    assert byo_cpu.get("test_obj") == list(range(10))

    local_folder = local_folder.to(byo_cpu)
    assert "sample_file_0.txt" in local_folder.ls(full_paths=False)


@pytest.mark.clustertest
def test_byo_proxy(byo_cpu, local_folder):
    rh.rh_config.open_cluster_tunnels.pop(byo_cpu.address)
    byo_cpu.client = None
    # byo_cpu._rpc_tunnel.close()
    byo_cpu._rpc_tunnel = None

    byo_cpu._ssh_creds["ssh_host"] = "127.0.0.1"
    byo_cpu._ssh_creds.update(
        {"ssh_proxy_command": "ssh -W %h:%p ubuntu@test-byo-cluster"}
    )
    assert byo_cpu.up_if_not()

    status, stdout, _ = byo_cpu.run(["echo hi"])[0]
    assert status == 0
    assert stdout == "hi\n"

    summer_func = rh.function(summer, env=rh.env(working_dir="local:./")).to(byo_cpu)
    assert summer_func(1, 2) == 3

    byo_cpu.put("test_obj", list(range(10)))
    assert byo_cpu.get("test_obj") == list(range(10))

    # TODO: uncomment out when in-mem lands
    # local_folder = local_folder.to(byo_cpu)
    # assert "sample_file_0.txt" in local_folder.ls(full_paths=False)


@unittest.skip("Support for multiple live clusters not yet implemented")
def test_connections_to_multiple_sm_clusters(sm_cluster):
    assert sm_cluster.is_up()

    np_func = rh.function(np_array).to(sm_cluster, env=["./", "numpy", "pytest"])

    # Run function on SageMaker compute
    my_list = [1, 2, 3]
    res = np_func(my_list)
    assert res.tolist() == my_list


@pytest.mark.sagemakertest
def test_launch_and_connect_to_sagemaker(sm_cluster):
    assert sm_cluster.is_up()

    # Run func on the cluster
    np_func = rh.function(np_array).to(sm_cluster, env=["./", "numpy"])
    my_list = [1, 2, 3]
    res = np_func(my_list)
    assert res.tolist() == my_list

    # Use cluster object store
    test_list = list(range(5, 50, 2)) + ["a string"]
    sm_cluster.put("my_list", test_list)
    ret = sm_cluster.get("my_list")
    assert ret == test_list

    # Run CLI commands
    return_codes = sm_cluster.run(commands=["ls -la"])
    assert return_codes[0][0] == 0


@pytest.mark.sagemakertest
def test_create_and_run_sagemaker_training_job(sm_source_dir, sm_entry_point):
    import dotenv
    from sagemaker.pytorch import PyTorch

    dotenv.load_dotenv()

    cluster_name = "rh-sagemaker-training"

    # https://sagemaker.readthedocs.io/en/stable/api/training/estimators.html#sagemaker.estimator.EstimatorBase
    estimator = PyTorch(
        entry_point=sm_entry_point,
        # Estimator requires a role ARN (can't be a profile)
        role=os.getenv("AWS_ROLE_ARN"),
        # Script can sit anywhere in the file system
        source_dir=sm_source_dir,
        framework_version="1.9.1",
        py_version="py38",
        instance_count=1,
        instance_type="ml.m5.large",
        # https://docs.aws.amazon.com/sagemaker/latest/dg/train-warm-pools.html
        keep_alive_period_in_seconds=3600,
        # A list of absolute or relative paths to directories with any additional libraries that
        # should be exported to the cluster
        dependencies=[],
    )

    c = rh.sagemaker_cluster(name=cluster_name, estimator=estimator)
    c.save()

    reloaded_cluster = rh.sagemaker_cluster(cluster_name, dryrun=True)
    reloaded_cluster.teardown_and_delete()
    assert not reloaded_cluster.is_up()


@pytest.mark.sagemakertest
def test_stable_diffusion_on_sm_gpu(sm_gpu_cluster):
    # Note: Default image used on the cluster will already have torch installed
    sd_generate = (
        rh.function(sd_generate_image)
        .to(
            sm_gpu_cluster,
            env=[
                "diffusers",
                "transformers",
            ],
        )
        .save("sd_generate")
    )

    # the following runs on our remote SageMaker instance
    img = sd_generate("A hot dog made out of matcha.")
    assert img

    sm_gpu_cluster.teardown_and_delete()
    assert not sm_gpu_cluster.is_up()


if __name__ == "__main__":
    unittest.main()
