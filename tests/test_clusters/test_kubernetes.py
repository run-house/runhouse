import unittest

import pytest

import runhouse as rh


def np_array(num_list: list):
    import numpy as np

    return np.array(num_list)


@pytest.mark.k8s_test
def test_launch_and_connect_to_sagemaker(k8s_cluster):
    assert k8s_cluster.is_up()

    # Run func on the cluster
    np_func = rh.function(np_array).to(k8s_cluster, env=["./", "numpy"])
    my_list = [1, 2, 3]
    res = np_func(my_list)
    assert res.tolist() == my_list

    # Use cluster object store
    test_list = list(range(5, 50, 2)) + ["a string"]
    k8s_cluster.put("my_list", test_list)
    ret = k8s_cluster.get("my_list")
    assert ret == test_list

    # Run CLI commands
    return_codes = k8s_cluster.run(commands=["ls -la"])
    assert return_codes[0][0] == 0


if __name__ == "__main__":
    unittest.main()
