import copy
import os
import unittest
from typing import List

import pytest

import runhouse as rh

from runhouse.rns.hardware import OnDemandCluster


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


@pytest.mark.clustertest
def test_launch_and_connect_to_sagemaker(sm_cluster):
    # Reload the cluster object and run a command on the cluster
    reloaded_cluster = rh.sagemaker_cluster(sm_cluster.name)
    assert reloaded_cluster.is_up()

    # Check cluster object store is working
    test_list = list(range(5, 50, 2)) + ["a string"]
    reloaded_cluster.put("my_list", test_list)
    ret = reloaded_cluster.get("my_list")
    assert ret == test_list

    # # Test CLI commands
    return_codes = reloaded_cluster.run(commands=["ls -la"])
    assert return_codes[0][0] == 0


@pytest.mark.gputest
def test_sd_generate():
    cluster = rh.sagemaker_cluster(
        name="rh-sagemaker-gpu",
        instance_type="ml.g5.2xlarge",
    ).save()

    sd_generate = (
        rh.function(sd_generate_image)
        .to(
            cluster,
            env=["torch==2.0.0", "diffusers", "transformers", "accelerate", "pytest"],
        )
        .save("sd_generate")
    )

    # the following runs on our remote SageMaker instance
    sd_generate("A hot dog made out of matcha.").show()

    cluster.teardown_and_delete()


@pytest.mark.clustertest
def test_run_function_on_sagemaker():
    # Launch a new SageMaker instance, save the cluster object's metadata to Runhouse Den for re-use and sharing
    cluster = rh.sagemaker_cluster(name="rh-sagemaker", connection_wait_time=0).save()
    assert cluster.is_up()

    np_func = rh.function(np_array).to(cluster, env=["numpy", "pytest"])

    # Run function on SageMaker compute
    my_list = [1, 2, 3]
    res = np_func(my_list)

    assert res.tolist() == my_list


@pytest.mark.clustertest
def test_create_sagemaker_training_job():
    from sagemaker.pytorch import PyTorch

    role = os.getenv("ROLE_ARN")

    estimator = PyTorch(
        entry_point="pytorch_train.py",
        source_dir=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "runhouse/scripts/sagemaker",
        ),
        role=role,
        framework_version="1.9.1",
        py_version="py38",
        instance_count=1,
        instance_type="ml.m5.large",
    )
    c = rh.sagemaker_cluster(
        name="rh-sagemaker-training", role=role, estimator=estimator
    )
    c.save()

    reloaded_cluster = rh.sagemaker_cluster("rh-sagemaker-training", dryrun=True)
    assert reloaded_cluster


if __name__ == "__main__":
    unittest.main()
