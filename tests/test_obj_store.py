import time

import numpy as np

import pytest

import runhouse as rh
from runhouse.logger import get_logger

from tests.test_resources.test_modules.test_functions.test_function import (
    multiproc_np_sum,
)

TEMP_FILE = "my_file.txt"
TEMP_FOLDER = "~/runhouse-tests"

logger = get_logger(__name__)

UNIT = {"cluster": []}
LOCAL = {
    "cluster": [
        "docker_cluster_pwd_ssh_no_auth",
        "docker_cluster_pk_ssh_no_auth",
    ]
}
MINIMAL = {"cluster": ["ondemand_aws_docker_cluster"]}
RELEASE = {
    "cluster": [
        "ondemand_aws_docker_cluster",
        "ondemand_gcp_cluster",
        "ondemand_k8s_cluster",
        "ondemand_k8s_docker_cluster",
        "ondemand_aws_https_cluster_with_auth",
        "static_cpu_pwd_cluster",
    ]
}
MAXIMAL = {
    "cluster": [
        "docker_cluster_pwd_ssh_no_auth",
        "docker_cluster_pk_ssh_no_auth",
        "ondemand_aws_docker_cluster",
        "ondemand_gcp_cluster",
        "ondemand_k8s_cluster",
        "ondemand_k8s_docker_cluster",
        "ondemand_aws_https_cluster_with_auth",
        "multinode_cpu_docker_conda_cluster",
        "static_cpu_pwd_cluster",
    ]
}


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


def test_stream_logs(cluster):
    print_fn = rh.function(fn=do_printing_and_logging, system=cluster)
    res = print_fn(stream_logs=True)
    # TODO [DG] assert that the logs are streamed
    assert res == list(range(50))


def test_get_from_cluster(cluster):
    print_fn = rh.function(fn=do_printing_and_logging, system=cluster)
    print(print_fn())
    res = print_fn.call.remote()
    assert isinstance(res, rh.Blob)
    assert res.name in cluster.keys()

    assert res.fetch() == list(range(50))
    res = cluster.get(res.name).resolved_state()
    assert res == list(range(50))


def test_put_and_get_on_cluster(cluster):
    test_list = list(range(5, 50, 2)) + ["a string"]
    cluster.put("my_list", test_list)
    ret = cluster.get("my_list")
    assert all(a == b for (a, b) in zip(ret, test_list))

    # Test that NOT_FOUND error is raised when key doesn't exist
    with pytest.raises(KeyError) as e:
        cluster.get("nonexistent_key", default=KeyError)
    assert "key nonexistent_key not found" in str(e.value)


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


@pytest.mark.parametrize("env", [None])
def test_stateful_generator(cluster, env):
    # We need this here just to make sure the "tests" module is synced over
    rh.function(fn=do_printing_and_logging, system=cluster)
    cluster.put("slow_numpy_array", slow_numpy_array(), env=env)
    for val in cluster.call("slow_numpy_array", "slow_get_array", stream_logs=True):
        assert val
        print(val)


def pinning_helper(key=None):
    from_obj_store = rh.here.get(key + "_inside")
    if from_obj_store:
        return "Found in obj store!"

    rh.blob(name=key + "_inside", data=["put within fn"] * 5).pin()
    return ["fn result"] * 3


def test_pinning_and_arg_replacement(cluster):
    cluster.clear()
    pin_fn = rh.function(pinning_helper).to(cluster)

    # First run should pin "run_pin" and "run_pin_inside"
    pin_fn.call.remote(key="run_pin", run_name="pinning_test")
    print(cluster.keys())
    assert cluster.get("pinning_test").fetch() == ["fn result"] * 3
    assert cluster.get("run_pin_inside").data == ["put within fn"] * 5

    # When we just ran with the arg "run_pin", we put a new pin called "pinning_test_inside"
    # from within the fn. Running again should return it.
    assert pin_fn("run_pin") == "Found in obj store!"

    put_pin_value = ["put_pin_value"] * 4
    cluster.put("put_pin_inside", put_pin_value)
    assert pin_fn("put_pin") == "Found in obj store!"


def test_put_resource(cluster, test_env):
    test_env.name = "~/test_env"
    cluster.put_resource(test_env)
    assert cluster.call("test_env", "config", stream_logs=True) == test_env.config()
    assert cluster.get("test_env").config() == test_env.config()


def serialization_helper_1():
    import torch

    tensor = torch.zeros(100).cuda()
    rh.pin_to_memory("torch_tensor", tensor)


def serialization_helper_2():
    tensor = rh.get_pinned_object("torch_tensor")
    return tensor.device()  # Should succeed if array hasn't been serialized


@pytest.mark.skip
def test_pinning_to_gpu(k80_gpu_cluster):
    # Based on the following quirk having to do with Numpy objects becoming immutable if they're serialized:
    # https://docs.ray.io/en/latest/ray-core/objects/serialization.html#fixing-assignment-destination-is-read-only
    k80_gpu_cluster.delete()
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


def test_pinning_in_memory(cluster):
    # Based on the following quirk having to do with Numpy objects becoming immutable if they're serialized:
    # https://docs.ray.io/en/latest/ray-core/objects/serialization.html#fixing-assignment-destination-is-read-only
    cluster.clear()
    fn_1 = rh.function(np_serialization_helper_1).to(cluster)
    fn_1()
    fn_2 = rh.function(np_serialization_helper_2).to(cluster)
    res = fn_2()
    assert res[0] == 1
    assert res[1] == 0


def test_multiprocessing_streaming(cluster):
    re_fn = rh.function(multiproc_np_sum, system=cluster, env=["numpy"])
    summands = list(zip(range(5), range(4, 9)))
    res = re_fn(summands)
    assert res == [4, 6, 8, 10, 12]


def test_tqdm_streaming(cluster):
    # Note, this doesn't work properly in PyCharm due to incomplete
    # support for carriage returns in the PyCharm console.
    print_fn = rh.function(fn=do_tqdm_printing_and_logging).to(cluster, env=["tqdm"])
    res = print_fn(steps=40, stream_logs=True)
    assert res == list(range(50))
