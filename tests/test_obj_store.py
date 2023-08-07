import inspect
import logging
import time
import unittest

import numpy as np

import pytest

import runhouse as rh
from runhouse import Package

from tests.test_function import multiproc_torch_sum

TEMP_FILE = "my_file.txt"
TEMP_FOLDER = "~/runhouse-tests"

logger = logging.getLogger(__name__)


def do_printing_and_logging(steps=3):
    for i in range(steps):
        # Wait to make sure we're actually streaming
        time.sleep(1)
        print(f"Hello from the cluster stdout! {i}")
        logger.info(f"Hello from the cluster logs! {i}")
    return list(range(50))


def do_tqdm_printing_and_logging(steps=6):
    from tqdm.auto import tqdm  # progress bar

    progress_bar = tqdm(range(steps))
    for i in range(steps):
        # Wait to make sure we're actually streaming
        time.sleep(0.1)
        progress_bar.update(1)
    return list(range(50))


@pytest.mark.clustertest
def test_stream_logs(ondemand_cpu_cluster):
    print_fn = rh.function(fn=do_printing_and_logging, system=ondemand_cpu_cluster)
    res = print_fn(stream_logs=True)
    # TODO [DG] assert that the logs are streamed
    assert res == list(range(50))


@pytest.mark.clustertest
def test_get_from_cluster(ondemand_cpu_cluster):
    print_fn = rh.function(fn=do_printing_and_logging, system=ondemand_cpu_cluster)
    print(print_fn())
    res = print_fn.remote()
    assert isinstance(res, rh.Blob)
    assert res.name in ondemand_cpu_cluster.list_keys()

    assert res.fetch.data == list(range(50))
    res = ondemand_cpu_cluster.get(res.name)
    assert res == list(range(50))


@pytest.mark.clustertest
def test_put_and_get_on_cluster(ondemand_cpu_cluster):
    test_list = list(range(5, 50, 2)) + ["a string"]
    ondemand_cpu_cluster.put("my_list", test_list)
    ret = ondemand_cpu_cluster.get("my_list")
    assert all(a == b for (a, b) in zip(ret, test_list))


@pytest.mark.clustertest
@pytest.mark.parametrize("env", [None, "base", "pytorch"])
def test_call_module_method(ondemand_cpu_cluster, env):
    ondemand_cpu_cluster.put("numpy_pkg", Package.from_string("numpy"), env=env)

    # Test for method
    res = ondemand_cpu_cluster.call_module_method(
        "numpy_pkg", "_detect_cuda_version_or_cpu"
    )
    assert res == "cpu"

    # Test for property
    res = ondemand_cpu_cluster.call_module_method("numpy_pkg", "config_for_rns")
    numpy_config = Package.from_string("numpy").config_for_rns
    assert res
    assert isinstance(res, dict)
    assert res == numpy_config

    # Test iterator
    ondemand_cpu_cluster.put("config_dict", list(numpy_config.keys()), env=env)
    res = ondemand_cpu_cluster.call_module_method(
        "config_dict", "__iter__", stream_logs=True
    )
    # Checks that all the keys in numpy_config were returned
    inspect.isgenerator(res)
    for key in res:
        assert key
        numpy_config.pop(key)
    assert not numpy_config


class slow_numpy_array:
    def __init__(self, size=5):
        self.size = size
        self.arr = np.zeros(self.size)

    def slow_get_array(self):
        for i in range(self.size):
            time.sleep(1)
            logger.info(f"Hello from the cluster logs! {i}")
            self.arr[i] = i
            yield f"Hello from the cluster! {self.arr}"


@pytest.mark.clustertest
@pytest.mark.parametrize("env", [None])
def test_stateful_generator(ondemand_cpu_cluster, env):
    # We need this here just to make sure the "tests" module is synced over
    # TODO remove when we add support for rh.Module
    rh.function(fn=do_printing_and_logging, system=ondemand_cpu_cluster)
    ondemand_cpu_cluster.put("slow_numpy_array", slow_numpy_array(), env=env)
    for val in ondemand_cpu_cluster.call_module_method(
        "slow_numpy_array", "slow_get_array", stream_logs=True
    ):
        assert val
        print(val)


def pinning_helper(key=None):
    from_obj_store = rh.here.get(key + "_inside")
    if from_obj_store:
        return "Found in obj store!"

    rh.blob(name=key + "_inside", data=["put within fn"] * 5).pin()
    return ["fn result"] * 3


@pytest.mark.clustertest
def test_pinning_and_arg_replacement(ondemand_cpu_cluster):
    ondemand_cpu_cluster.delete_keys()
    pin_fn = rh.function(pinning_helper).to(ondemand_cpu_cluster)

    # First run should pin "run_pin" and "run_pin_inside"
    pin_fn.remote(key="run_pin", run_name="pinning_test")
    print(ondemand_cpu_cluster.list_keys())
    assert ondemand_cpu_cluster.get("pinning_test") == ["fn result"] * 3
    assert ondemand_cpu_cluster.get("run_pin_inside").data == ["put within fn"] * 5

    # When we just ran with the arg "run_pin", we put a new pin called "pinning_test_inside"
    # from within the fn. Running again should return it.
    assert pin_fn("run_pin") == "Found in obj store!"

    put_pin_value = ["put_pin_value"] * 4
    ondemand_cpu_cluster.put("put_pin_inside", put_pin_value)
    assert pin_fn("put_pin") == "Found in obj store!"


@pytest.mark.clustertest
def test_put_resource(ondemand_cpu_cluster, test_env):
    test_env.name = "test_env"
    ondemand_cpu_cluster.put_resource(test_env)
    assert (
        ondemand_cpu_cluster.get("test_env").config_for_rns == test_env.config_for_rns
    )

    assert (
        ondemand_cpu_cluster.call_module_method(
            "test_env", "config_for_rns", stream_logs=True
        )
        == test_env.config_for_rns
    )
    assert (
        ondemand_cpu_cluster.call_module_method("test_env", "name", stream_logs=True)
        == "test_env"
    )


@pytest.mark.clustertest
def test_fault_tolerance(ondemand_cpu_cluster):
    ondemand_cpu_cluster.delete_keys()
    ondemand_cpu_cluster.put("my_list", list(range(5, 50, 2)) + ["a string"])
    ondemand_cpu_cluster.restart_server(restart_ray=False, resync_rh=False)
    ret = ondemand_cpu_cluster.get("my_list")
    assert all(a == b for (a, b) in zip(ret, list(range(5, 50, 2)) + ["a string"]))


def serialization_helper_1():
    import torch

    tensor = torch.zeros(100).cuda()
    rh.pin_to_memory("torch_tensor", tensor)


def serialization_helper_2():
    tensor = rh.get_pinned_object("torch_tensor")
    return tensor.device()  # Should succeed if array hasn't been serialized


@unittest.skip
@pytest.mark.clustertest
@pytest.mark.gputest
def test_pinning_to_gpu(k80_gpu_cluster):
    # Based on the following quirk having to do with Numpy objects becoming immutable if they're serialized:
    # https://docs.ray.io/en/latest/ray-core/objects/serialization.html#fixing-assignment-destination-is-read-only
    k80_gpu_cluster.delete_keys()
    fn_1 = rh.function(serialization_helper_1).to(k80_gpu_cluster)
    fn_2 = rh.function(serialization_helper_2).to(k80_gpu_cluster)
    fn_1()
    fn_2()


def np_serialization_helper_1():
    print(time.time())
    rh.blob(data=np.zeros(100), name="np_arr").pin()


def np_serialization_helper_2():
    print(time.time())
    arr = rh.blob(name="np_arr").data
    arr[0] = 1
    return arr


@pytest.mark.clustertest
def test_pinning_in_memory(ondemand_cpu_cluster):
    # Based on the following quirk having to do with Numpy objects becoming immutable if they're serialized:
    # https://docs.ray.io/en/latest/ray-core/objects/serialization.html#fixing-assignment-destination-is-read-only
    ondemand_cpu_cluster.delete_keys()
    fn_1 = rh.function(np_serialization_helper_1).to(ondemand_cpu_cluster)
    fn_1()
    fn_2 = rh.function(np_serialization_helper_2).to(ondemand_cpu_cluster)
    res = fn_2()
    assert res[0] == 1
    assert res[1] == 0


@pytest.mark.clustertest
def test_multiprocessing_streaming(ondemand_cpu_cluster):
    re_fn = rh.function(
        multiproc_torch_sum, system=ondemand_cpu_cluster, env=["./", "torch==1.12.1"]
    )
    summands = list(zip(range(5), range(4, 9)))
    res = re_fn(summands)
    assert res == [4, 6, 8, 10, 12]


@pytest.mark.clustertest
def test_tqdm_streaming(ondemand_cpu_cluster):
    # Note, this doesn't work properly in PyCharm due to incomplete
    # support for carriage returns in the PyCharm console.
    print_fn = rh.function(fn=do_tqdm_printing_and_logging, system=ondemand_cpu_cluster)
    res = print_fn(steps=40, stream_logs=True)
    assert res == list(range(50))


@pytest.mark.clustertest
def test_cancel_run(ondemand_cpu_cluster):
    print_fn = rh.function(fn=do_printing_and_logging, system=ondemand_cpu_cluster)
    run_key = print_fn.run(10)
    run_key2 = print_fn.run()

    # TODO if you look at screen on the cluster, the job is continuing
    ondemand_cpu_cluster.cancel(run_key, force=True)
    with pytest.raises(Exception) as e:
        ondemand_cpu_cluster.get(run_key, stream_logs=True)
    assert "task was cancelled" in str(e.value)

    # Check that another job in the same env isn't affected
    res = ondemand_cpu_cluster.get(run_key2)
    assert res == list(range(50))


if __name__ == "__main__":
    unittest.main()
