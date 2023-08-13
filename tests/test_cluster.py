import copy
import os
import unittest
from typing import List

import pytest

import runhouse as rh
from runhouse.rns.hardware import OnDemandCluster

from .conftest import cpu_clusters


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
@cpu_clusters
def test_on_diff_cluster(cluster, byo_cpu):
    func_hw = rh.function(is_on_cluster).to(cluster)
    assert not func_hw(byo_cpu)


@pytest.mark.clustertest
def test_launch_and_connect_to_sagemaker(sm_cluster):
    # Reload the cluster object and run a command on the cluster
    assert sm_cluster.is_up()

    # Check cluster object store is working
    test_list = list(range(5, 50, 2)) + ["a string"]
    sm_cluster.put("my_list", test_list)
    ret = sm_cluster.get("my_list")
    assert ret == test_list

    # # Test CLI commands
    return_codes = sm_cluster.run(commands=["ls -la"])
    assert return_codes[0][0] == 0


@pytest.mark.clustertest
def test_run_function_on_sagemaker(sm_cluster):
    assert sm_cluster.is_up()

    np_func = rh.function(np_array).to(sm_cluster, env=["./", "numpy", "pytest"])

    # Run function on SageMaker compute
    my_list = [1, 2, 3]
    res = np_func(my_list)

    assert res.tolist() == my_list


@pytest.mark.clustertest
def test_create_sagemaker_training_job(sm_source_dir, sm_entry_point):
    from sagemaker.pytorch import PyTorch

    cluster_name = "rh-sagemaker-training"

    # https://sagemaker.readthedocs.io/en/stable/api/training/estimators.html#sagemaker.estimator.EstimatorBase
    estimator = PyTorch(
        entry_point=sm_entry_point,
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


@pytest.mark.clustertest
def test_stable_diffusion_on_sm_gpu(sm_gpu_cluster):
    sd_generate = (
        rh.function(sd_generate_image)
        .to(
            sm_gpu_cluster,
            env=[
                "torch==2.0.0",
                "diffusers",
                "transformers",
                "accelerate",
                "pytest",
            ],
        )
        .save("sd_generate")
    )

    # the following runs on our remote SageMaker instance
    img = sd_generate("A hot dog made out of matcha.")
    assert img


if __name__ == "__main__":
    unittest.main()
