import os

import pytest

import runhouse as rh

from tests.test_resources.test_clusters.cluster_tests import np_array, sd_generate_image


@pytest.mark.skip("Support for multiple live clusters not yet implemented")
def test_connections_to_multiple_sm_clusters(sm_cluster):
    assert sm_cluster.is_up()

    np_func = rh.function(np_array).to(sm_cluster, env=["./", "numpy", "pytest"])

    # Run function on SageMaker compute
    my_list = [1, 2, 3]
    res = np_func(my_list)
    assert res.tolist() == my_list


def test_launch_and_connect_to_sagemaker(sm_cluster):
    assert sm_cluster.is_up()

    # Run func on the cluster
    np_func = rh.function(np_array).to(system=sm_cluster, env=["./", "numpy"])
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


def test_create_and_run_sagemaker_training_job(sm_source_dir, sm_entry_point):
    import dotenv
    from sagemaker.pytorch import PyTorch

    dotenv.load_dotenv()

    role_arn = os.getenv("AWS_ROLE_ARN")
    assert role_arn

    cluster_name = "rh-sagemaker-training"

    # https://sagemaker.readthedocs.io/en/stable/api/training/estimators.html#sagemaker.estimator.EstimatorBase
    estimator = PyTorch(
        entry_point=sm_entry_point,
        # Estimator requires a role ARN (can't be a profile)
        role=role_arn,
        # Script can sit anywhere in the file system
        source_dir=sm_source_dir,
        # PyTorch version for executing training code
        framework_version="1.13",
        py_version="py39",
        instance_count=1,
        instance_type="ml.m5.large",
        # https://docs.aws.amazon.com/sagemaker/latest/dg/train-warm-pools.html
        keep_alive_period_in_seconds=3600,
        # A list of absolute or relative paths to directories with any additional libraries that
        # should be exported to the cluster
        dependencies=[],
    )

    rh.sagemaker_cluster(name=cluster_name, estimator=estimator).up_if_not().save()

    reloaded_cluster = rh.sagemaker_cluster(name=cluster_name)
    reloaded_cluster.teardown_and_delete()
    assert not reloaded_cluster.is_up()


def test_stable_diffusion_on_sm_gpu(sm_gpu_cluster):
    # Note: Default image used on the cluster will already have torch installed
    sd_generate = (
        rh.function(sd_generate_image)
        .to(sm_gpu_cluster, env=["diffusers", "transformers"])
        .save()
    )

    # the following runs on our remote SageMaker instance
    img = sd_generate("A hot dog made out of matcha.")
    assert img

    sm_gpu_cluster.teardown_and_delete()
    assert not sm_gpu_cluster.is_up()
