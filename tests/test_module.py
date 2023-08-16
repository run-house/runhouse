import inspect
import logging
import os
import time
import unittest

import numpy as np
import pandas as pd

import pytest

import runhouse as rh
from runhouse import Package


logger = logging.getLogger(__name__)


""" Tests for runhouse.Module. Structure:
    - Test call_module_method rpc, with various envs
    - Test creating module from class
    - Test creating module as rh.Module subclass
    - Test calling Module methods async
"""


@pytest.mark.clustertest
# @pytest.mark.parametrize("env", [None, "base", "pytorch"])
@pytest.mark.parametrize("env", [None])
def test_call_module_method(ondemand_cpu_cluster, env):
    ondemand_cpu_cluster.put("numpy_pkg", Package.from_string("numpy"), env=env)

    # Test for method
    res = ondemand_cpu_cluster.call_module_method(
        "numpy_pkg", "_detect_cuda_version_or_cpu", stream_logs=True
    )
    assert res == "cpu"

    # Test for property
    res = ondemand_cpu_cluster.call_module_method(
        "numpy_pkg", "config_for_rns", stream_logs=True
    )
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


class SlowNumpyArray:
    def __init__(self, size=5):
        self.size = size
        self.arr = None
        self._hidden_1 = "hidden"

    def remote_init(self):
        self.arr = np.zeros(self.size)
        self._hidden_2 = "hidden"

    def slow_iter(self):
        if not self._hidden_1 and self._hidden_2:
            raise ValueError("Hidden attributes not set")
        for i in range(self.size):
            time.sleep(1)
            print(f"Hello from the cluster stdout! {i}")
            logger.info(f"Hello from the cluster logs! {i}")
            self.arr[i] = i
            yield f"Hello from the cluster! {self.arr}"

    def cpu_count(self, local=True):
        return os.cpu_count()


@pytest.mark.clustertest
# @pytest.mark.parametrize("env", [None, "base", "pytorch"])
@pytest.mark.parametrize("env", [None])
def test_module_from_class(ondemand_cpu_cluster, env):
    RemoteClass = rh.module(SlowNumpyArray).to(ondemand_cpu_cluster)
    remote_array = RemoteClass(size=3)
    assert remote_array.system == ondemand_cpu_cluster
    results = []
    out = ""
    with rh.capture_stdout() as stdout:
        for i, val in enumerate(remote_array.slow_iter()):
            assert val
            print(val)
            results += [val]
            out = out + str(stdout)
    assert len(results) == 3

    # Check that stdout was captured. Skip the last result because sometimes we
    # don't catch it and it makes the test flaky.
    for i in range(remote_array.size - 1):
        assert f"Hello from the cluster stdout! {i}" in out
        assert f"Hello from the cluster logs! {i}" in out

    print(remote_array.cpu_count())
    assert remote_array.cpu_count() == os.cpu_count()
    print(remote_array.cpu_count(local=False))
    assert remote_array.cpu_count(local=False) == 2

    # Properties
    print(remote_array.arr)
    # We expect the arr to be populated, because we instantiated the class locally before sending to cluster,
    # but it should reflect the remote updates we made.
    assert remote_array.arr is None
    arr = remote_array.fetch.arr
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (3,)
    assert arr[0] == 0
    assert arr[2] == 2

    remote_array.size = 20
    assert remote_array.fetch.size == 20


class SlowPandas(rh.Module):
    def __init__(self, size=5):
        super().__init__()
        self.size = size
        self.df = None

    def remote_init(self):
        self.df = pd.DataFrame(np.zeros((self.size, self.size)))

    def slow_iter(self):
        for i in range(self.size):
            time.sleep(1)
            print(f"Hello from the cluster stdout! {i}")
            logger.info(f"Hello from the cluster logs! {i}")
            self.df[i] = i
            yield f"Hello from the cluster! {self.df.loc[[i]]}"

    async def slow_iter_async(self):
        for i in range(self.size):
            time.sleep(1)
            print(f"Hello from the cluster stdout! {i}")
            logger.info(f"Hello from the cluster logs! {i}")
            self.df[i] = i
            yield f"Hello from the cluster! {self.df.loc[[i]]}"

    def cpu_count(self, local=True):
        return os.cpu_count()

    async def cpu_count_async(self, local=True):
        return os.cpu_count()


@pytest.mark.clustertest
# @pytest.mark.parametrize("env", [None, "base", "pytorch"])
@pytest.mark.parametrize("env", [None])
def test_module_from_subclass(ondemand_cpu_cluster, env):
    remote_df = SlowPandas(size=3).to(ondemand_cpu_cluster, env)
    assert remote_df.system == ondemand_cpu_cluster
    results = []
    # Capture stdout to check that it's working
    out = ""
    with rh.capture_stdout() as stdout:
        for i, val in enumerate(remote_df.slow_iter()):
            assert val
            print(val)
            results += [val]
            out = out + str(stdout)
    assert len(results) == 3

    # Check that stdout was captured. Skip the last result because sometimes we
    # don't catch it and it makes the test flaky.
    for i in range(remote_df.size - 1):
        assert f"Hello from the cluster stdout! {i}" in out
        assert f"Hello from the cluster logs! {i}" in out

    print(remote_df.cpu_count())
    assert remote_df.cpu_count() == os.cpu_count()
    print(remote_df.cpu_count(local=False))
    assert remote_df.cpu_count(local=False) == 2

    # Properties
    assert remote_df.df is None
    df = remote_df.fetch.df
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (3, 3)
    assert df.loc[0, 0] == 0
    assert df.loc[2, 2] == 2

    remote_df.size = 20
    assert remote_df.fetch.size == 20

    del remote_df

    # Test get_or_to
    remote_df = SlowPandas(size=3).get_or_to(ondemand_cpu_cluster, env=env, name="SlowPandas")
    assert remote_df.system.config_for_rns == ondemand_cpu_cluster.config_for_rns
    assert remote_df.cpu_count(local=False, stream_logs=False) == 2
    # Check that size is unchanged from when we set it to 20 above
    assert remote_df.fetch.size == 20



@pytest.mark.clustertest
@pytest.mark.asyncio
# @pytest.mark.parametrize("env", [None, "base", "pytorch"])
@pytest.mark.parametrize("env", [None])
async def test_module_from_subclass_async(ondemand_cpu_cluster, env):
    remote_df = SlowPandas(size=3).to(ondemand_cpu_cluster, env)
    assert remote_df.system == ondemand_cpu_cluster
    results = []
    # Capture stdout to check that it's working
    out = ""
    with rh.capture_stdout() as stdout:
        async for val in remote_df.slow_iter_async():
            assert val
            print(val)
            results += [val]
            out = out + str(stdout)
    assert len(results) == 3

    # Check that stdout was captured. Skip the last result because sometimes we
    # don't catch it and it makes the test flaky.
    for i in range(remote_df.size - 1):
        assert f"Hello from the cluster stdout! {i}" in out
        assert f"Hello from the cluster logs! {i}" in out

    print(await remote_df.cpu_count_async())
    assert await remote_df.cpu_count_async() == os.cpu_count()
    print(await remote_df.cpu_count_async(local=False))
    assert await remote_df.cpu_count_async(local=False) == 2

    # Properties
    assert remote_df.df is None
    df = await remote_df.fetch_async("df")
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (3, 3)
    assert df.loc[0, 0] == 0
    assert df.loc[2, 2] == 2

    await remote_df.set_async("size", 20)
    assert remote_df.fetch.size == 20


if __name__ == "__main__":
    unittest.main()
