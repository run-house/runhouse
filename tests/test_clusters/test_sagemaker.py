import os
import unittest
from pathlib import Path

import pytest

import runhouse as rh

from .test_cluster import np_array, sd_generate_image


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
        framework_version="2.1.0",
        py_version="py310",
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


@pytest.mark.sagemakertest
def test_sm_cluster_with_https(sm_cluster):
    # After launching the cluster with the existing fixture, restart the server on the cluster using HTTPS
    sm_cluster.server_connection_type = "tls"
    sm_cluster.restart_server()

    local_cert_path = sm_cluster.ssl_certfile
    assert Path(local_cert_path).exists()

    # Confirm we can send https requests to the cluster
    sm_cluster.install_packages(["gradio"])


@pytest.mark.clustertest
def test_restart_sm_cluster_with_den_auth(sm_cluster):
    from runhouse.globals import configs

    sm_cluster.den_auth = True
    sm_cluster.restart_server()

    # Create an invalid token, confirm the server does not accept the request
    orig_token = configs.get("token")

    # Request should return 200 with valid token
    sm_cluster.client.check_server()

    configs.set("token", "abcd123")

    # Request should raise an exception with an invalid token
    try:
        sm_cluster.client.check_server()
    except ValueError as e:
        assert "Invalid or expired token" in str(e)

    configs.set("token", orig_token)


if __name__ == "__main__":
    unittest.main()
