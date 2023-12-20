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

from tests.utils import test_account

logger = logging.getLogger(__name__)

""" Tests for runhouse.Module. Structure:
    - Test call_module_method rpc, with various envs
    - Test creating module from class
    - Test creating module as rh.Module subclass
    - Test calling Module methods async
"""


def resolve_test_helper(obj):
    return obj


class SlowNumpyArray:
    def __init__(self, size=5):
        self.size = size
        self.arr = np.zeros(self.size)
        self._hidden_1 = "hidden"

    def slow_iter(self):
        self._hidden_2 = "hidden"
        if not self._hidden_1 and self._hidden_2:
            raise ValueError("Hidden attributes not set")
        for i in range(self.size):
            time.sleep(1)
            print(f"Hello from the cluster stdout! {i}")
            logger.info(f"Hello from the cluster logs! {i}")
            self.arr[i] = i
            yield f"Hello from the cluster! {self.arr}"

    @classmethod
    def cpu_count(cls, local=True):
        return os.cpu_count()

    def size_minus_cpus(self):
        return self.size - self.cpu_count()

    @classmethod
    def factory_constructor(cls, size=5):
        return cls(size=size)


class SlowPandas(rh.Module):
    def __init__(self, size=5):
        super().__init__()
        self.size = size
        self.df = pd.DataFrame(np.zeros((self.size, self.size)))
        self._hidden_1 = "hidden"

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


class Calculator:
    importer = "Calculators Inc"

    def __init__(self, owner=None):
        self.model = "Casio"
        self._release_year = 2023
        self.owner = owner

    def summer(self, a: int, b: int):
        return a + b

    def sub(self, a: int, b: int):
        return a - b

    def divider(self, a: int, b: int):
        if b == 0:
            raise ZeroDivisionError
        return round(a / b)

    def mult(self, a: int, b: int):
        return a * b


@pytest.mark.usefixtures("cluster")
class TestModule:

    # --------- integration tests ---------
    @pytest.mark.parametrize("env", [None])
    @pytest.mark.level("local")
    def test_call_module_method(self, cluster, env):
        cluster.put("numpy_pkg", Package.from_string("numpy"), env=env)

        # Test for method
        res = cluster.call("numpy_pkg", "_detect_cuda_version_or_cpu", stream_logs=True)
        assert res == "cpu"

        # Test for property
        res = cluster.call("numpy_pkg", "config_for_rns", stream_logs=True)
        numpy_config = Package.from_string("numpy").config_for_rns
        assert res
        assert isinstance(res, dict)
        assert res == numpy_config

        # Test iterator
        cluster.put("config_dict", list(numpy_config.keys()), env=env)
        res = cluster.call("config_dict", "__iter__", stream_logs=True)
        # Checks that all the keys in numpy_config were returned
        inspect.isgenerator(res)
        for key in res:
            assert key
            numpy_config.pop(key)
        assert not numpy_config

    @pytest.mark.parametrize("env", [None])
    @pytest.mark.level("local")
    def test_module_from_factory(self, cluster, env):
        size = 3
        RemoteClass = rh.module(SlowNumpyArray).to(cluster)
        remote_array = RemoteClass(size=size, name="remote_array1")

        # Test that naming works properly, and "class" module was unaffacted
        assert remote_array.name == "remote_array1"
        assert RemoteClass.name == "SlowNumpyArray"

        # Test that module was initialized correctly on the cluster
        assert remote_array.system == cluster
        assert remote_array.remote.size == size
        assert all(remote_array.remote.arr == np.zeros(size))
        assert remote_array.remote._hidden_1 == "hidden"

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
        for i in range(size - 1):
            assert f"Hello from the cluster stdout! {i}" in out
            assert f"Hello from the cluster logs! {i}" in out

        cluster_cpus = int(
            cluster.run_python(["import os; print(os.cpu_count())"])[0][1]
        )
        # Test classmethod on remote class
        assert RemoteClass.cpu_count() == os.cpu_count()
        assert RemoteClass.cpu_count(local=False) == cluster_cpus

        # Test classmethod on remote instance
        assert remote_array.cpu_count() == os.cpu_count()
        assert remote_array.cpu_count(local=False) == cluster_cpus

        # Test instance method
        assert remote_array.size_minus_cpus() == size - cluster_cpus

        # Test remote getter
        arr = remote_array.remote.arr
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (size,)
        assert arr[0] == 0
        assert arr[2] == 2

        # Test remote setter
        remote_array.remote.size = 20
        assert remote_array.remote.size == 20

        # Test creating a second instance of the same class
        remote_array2 = RemoteClass(size=30, name="remote_array2")
        assert remote_array2.system == cluster
        assert remote_array2.remote.size == 30

        # Test creating a third instance with the factory method
        remote_array3 = RemoteClass.factory_constructor.remote(
            size=40, run_name="remote_array3"
        )
        assert remote_array3.system.config_for_rns == cluster.config_for_rns
        assert remote_array3.remote.size == 40
        assert remote_array3.cpu_count(local=False) == cluster_cpus

        # Make sure first array and class are unaffected by this change
        assert remote_array.remote.size == 20
        assert RemoteClass.cpu_count(local=False) == cluster_cpus

        # Test resolve()
        helper = rh.function(resolve_test_helper).to(cluster, env=rh.Env())
        resolved_obj = helper(remote_array.resolve())
        assert resolved_obj.__class__.__name__ == "SlowNumpyArray"
        assert not hasattr(resolved_obj, "config_for_rns")
        assert resolved_obj.size == 20
        assert list(resolved_obj.arr) == [0, 1, 2]

    @pytest.mark.parametrize("env", [None])
    @pytest.mark.level("local")
    def test_module_from_subclass(self, cluster, env):
        size = 3
        remote_df = SlowPandas(size=size).to(cluster, env)
        assert remote_df.system == cluster

        # Test that module was initialized correctly on the cluster
        assert remote_df.remote.size == size
        assert len(remote_df.remote.df) == size
        assert remote_df.remote._hidden_1 == "hidden"

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

        cpu_count = int(cluster.run_python(["import os; print(os.cpu_count())"])[0][1])
        print(remote_df.cpu_count())
        assert remote_df.cpu_count() == os.cpu_count()
        print(remote_df.cpu_count(local=False))
        assert remote_df.cpu_count(local=False) == cpu_count

        # Test setting and getting properties
        df = remote_df.remote.df
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (3, 3)
        assert df.loc[0, 0] == 0
        assert df.loc[2, 2] == 2

        remote_df.size = 20
        assert remote_df.remote.size == 20

        del remote_df

        # Test get_or_to
        remote_df = SlowPandas(size=3).get_or_to(cluster, env=env, name="SlowPandas")
        assert remote_df.system.config_for_rns == cluster.config_for_rns
        assert remote_df.cpu_count(local=False, stream_logs=False) == cpu_count
        # Check that size is unchanged from when we set it to 20 above
        assert remote_df.remote.size == 20

        # Test that resolve() has no effect
        helper = rh.function(resolve_test_helper).to(cluster, env=rh.Env())
        resolved_obj = helper(remote_df.resolve())
        assert resolved_obj.__class__.__name__ == "SlowPandas"
        assert resolved_obj.size == 20  # resolved_obj.remote.size causing an error
        assert resolved_obj.config_for_rns == remote_df.config_for_rns

    @pytest.mark.asyncio
    @pytest.mark.parametrize("env", [None])
    @pytest.mark.level("local")
    async def test_module_from_subclass_async(self, cluster, env):
        remote_df = SlowPandas(size=3).to(cluster, env)
        assert remote_df.system == cluster
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

        cpu_count = int(cluster.run_python(["import os; print(os.cpu_count())"])[0][1])
        print(await remote_df.cpu_count_async())
        assert await remote_df.cpu_count_async() == os.cpu_count()
        print(await remote_df.cpu_count_async(local=False))
        assert await remote_df.cpu_count_async(local=False) == cpu_count

        # Properties
        df = await remote_df.fetch_async("df")
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (3, 3)
        assert df.loc[0, 0] == 0
        assert df.loc[2, 2] == 2

        await remote_df.set_async("size", 20)
        assert remote_df.remote.size == 20

    @pytest.mark.skip("Not working yet")
    @pytest.mark.level("local")
    def test_hf_autotokenizer(self, cluster):
        from transformers import AutoTokenizer

        AutoTokenizer.from_pretrained("bert-base-uncased")
        RemoteAutoTokenizer = rh.module(AutoTokenizer).to(cluster, env=["transformers"])
        tok = RemoteAutoTokenizer.from_pretrained.remote(
            "bert-base-uncased", run_name="bert-tok"
        )
        # assert tok.remote.pad_token == "<pad>"
        prompt = "Tell me about unified development interfaces into compute and data infrastructure."
        assert tok(prompt, return_tensors="pt").shape == (1, 18)

    # --------- Unittests ---------

    @pytest.mark.usefixtures("cluster")
    @pytest.mark.parametrize("env", [None])
    @pytest.mark.level("local")
    def test_create_and_rename(self, cluster, env):
        RemoteCalc = rh.module(cls=Calculator).to(cluster)
        remote_calc = RemoteCalc(owner="Runhouse", name="Runhouse_calc")
        assert isinstance(RemoteCalc, rh.Module)
        assert RemoteCalc.name == "Calculator"
        assert remote_calc.name == "Runhouse_calc"
        assert remote_calc.summer(2, 3) == 5
        assert remote_calc.mult(3, 7) == 21

        # test that after rename, module stays the same, apart from the name.
        # Moreover, tests that the rename did not affect the existing remote instances, as well as creating new ones.
        RemoteCalc.rename("my_calc_module")
        assert RemoteCalc.name == "my_calc_module"
        remote_calc_renamed = RemoteCalc(owner="Runhouse", name="Runhouse_calc_new")
        assert remote_calc_renamed.summer(4, 6) == 10
        assert remote_calc_renamed.divider(35, 7) == 5
        assert remote_calc.summer(4, 6) == remote_calc_renamed.summer(4, 6)
        assert remote_calc.divider(35, 7) == remote_calc_renamed.divider(35, 7)

    @pytest.mark.usefixtures("cluster")
    @pytest.mark.parametrize("env", [None])
    @pytest.mark.level("local")
    def test_local_remote_properties(self, cluster, env):
        RemoteCalc = rh.module(cls=Calculator).to(cluster)
        remote_calc = RemoteCalc(owner="Runhouse", name="Runhouse_calc")
        assert RemoteCalc.name == "Calculator"
        assert remote_calc.name == "Runhouse_calc"
        assert remote_calc.remote.model == "Casio"
        assert remote_calc.remote._release_year == 2023
        assert remote_calc.local.importer == "Calculators Inc"

    @pytest.mark.parametrize("env", [None])
    @pytest.mark.asyncio
    @pytest.mark.level("local")
    async def test_fetch_class_and_properties(self, cluster, env):
        RemoteCalc = rh.module(cls=Calculator).to(cluster)
        owner = "Runhouse"

        # fetch of a general class (not subclass)
        remote_calc = RemoteCalc(owner=owner)
        model = remote_calc.fetch("model")
        release_year = remote_calc.fetch("_release_year")
        remote_owner = remote_calc.fetch("owner")
        model_async = await remote_calc.fetch_async("model")
        release_year_async = await remote_calc.fetch_async("_release_year")
        owner_async = await remote_calc.fetch_async("owner")
        assert model == "Casio"
        assert model_async == "Casio"
        assert release_year == 2023
        assert release_year == release_year_async
        assert remote_owner == owner_async

        # fetch subclass
        remote_df_sync = SlowPandas(size=4).to(cluster, env).fetch()
        assert remote_df_sync.size == 4
        assert remote_df_sync.cpu_count() == SlowPandas(size=4).cpu_count()

        assert (
            await SlowPandas(size=4).to(system=cluster, env=env).fetch_async("size")
            == 4
        )

    @pytest.mark.parametrize("env", [None])
    @pytest.mark.level("local")
    def test_get_or_to(self, cluster, env):
        RemoteCalcNew = rh.module(Calculator).get_or_to(cluster, name="new_remote_cals")
        remote_calc_new = RemoteCalcNew(owner="Runhouse_admin", name="Admin_calc")
        assert remote_calc_new.mult(5, 9) == 45
        assert remote_calc_new.remote.owner == "Runhouse_admin"
        assert remote_calc_new.name == "Admin_calc"
        RemoteCalcExisting = rh.module(Calculator).get_or_to(
            cluster, name="new_remote_cals"
        )
        remote_calc_existing = RemoteCalcExisting(
            owner="Runhouse_users", name="Users_calc"
        )
        assert remote_calc_existing.mult(6, 2) == 12
        assert remote_calc_existing.remote.owner == "Runhouse_users"
        assert remote_calc_existing.remote.name == "Users_calc"

    @pytest.mark.parametrize("env", [None])
    @pytest.mark.level("local")
    def test_refresh(self, cluster, env):
        RemoteCalc = rh.module(cls=Calculator).to(cluster)
        # Note: by reusing the name, we're overwriting the class module in the cluster's object store with
        # the new instance.
        remote_calc = RemoteCalc(owner="Runhouse", name=RemoteCalc.name)
        assert remote_calc.sub(33, 5) == 28
        assert remote_calc.mult(4, 9) == 36
        # Because we overwrote the class module in the cluster's object store, when we refresh it we should get
        # the instance back.
        remote_calc_refreshed = RemoteCalc.refresh()
        assert remote_calc_refreshed.system.name == cluster.name
        assert remote_calc_refreshed.remote.owner == "Runhouse"
        assert remote_calc_refreshed.mult(4, 9) == 36
        assert remote_calc_refreshed.sub(33, 5) == 28

    def get_class_name(self, cls):
        return cls._name

    def calc_square(self, a: int):
        return a * a

    @pytest.mark.parametrize("env", [None])
    @pytest.mark.level("local")
    def test_resolve(self, cluster, env):
        remote_calc = rh.module(Calculator).to(cluster)

        # resolve method
        assert self.get_class_name(remote_calc.resolve()) == "Calculator"
        assert self.calc_square(remote_calc.resolve().mult(2, 3)) == 36
        assert self.calc_square(remote_calc.resolve().summer(2, 3)) == 25
        assert self.calc_square(remote_calc.resolve().divider(16, 2)) == 64
        assert self.calc_square(remote_calc.resolve().sub(1, 1)) == 0

        # resolved_state() method
        assert self.get_class_name(remote_calc.resolved_state()) == "Calculator"
        assert self.calc_square(remote_calc.resolved_state().mult(2, 3)) == 36
        assert self.calc_square(remote_calc.resolved_state().summer(2, 3)) == 25
        assert self.calc_square(remote_calc.resolved_state().divider(16, 2)) == 64
        assert self.calc_square(remote_calc.resolved_state().sub(1, 1)) == 0

    @pytest.mark.parametrize("env", [None])
    @pytest.mark.level("local")
    def test_save(self, cluster, env):
        # TODO: ask Josh for advice how to share it with a new user each time.
        users = ["josh@run.house"]
        remote_calc = rh.module(Calculator).to(cluster).save(name="rh_remote_calc")
        added_users, new_users = remote_calc.share(
            users=users, notify_users=False, access_level="write"
        )
        assert remote_calc.name == "rh_remote_calc"
        assert added_users == {users[0]: "write"} or added_users == {}
        assert new_users == {}

    @pytest.mark.parametrize("env", [None])
    @pytest.mark.asyncio
    @pytest.mark.level("local")
    async def test_set_async(self, cluster, env):
        RemoteCalc = rh.module(Calculator).to(cluster)
        my_remote_calc = RemoteCalc(owner="Runhouse", name="Runhouse_remote_dev")

        # test before changing the properties values.
        assert my_remote_calc.remote.owner == "Runhouse"
        assert my_remote_calc.remote.name == "Runhouse_remote_dev"
        assert my_remote_calc.remote.model == "Casio"
        assert my_remote_calc.remote._release_year == 2023

        # test after changing the properties values.
        await my_remote_calc.set_async("owner", "Runhouse_eng")
        assert my_remote_calc.remote.owner == "Runhouse_eng"
        await my_remote_calc.set_async("_release_year", 2020)
        assert my_remote_calc.remote._release_year == 2020

        # test that the unchanged properties remained the same.
        assert my_remote_calc.remote.model == "Casio"
        assert my_remote_calc.remote.name == "Runhouse_remote_dev"

    @pytest.mark.level("unit")
    def test_signature(self):

        SlowNumpy = rh.module(SlowNumpyArray)
        assert set(SlowNumpy.signature) == {
            "slow_iter",
            "cpu_count",
            "size_minus_cpus",
            "factory_constructor",
        }
        assert SlowNumpy.signature == {
            "cpu_count": {
                "signature": "(local=True)",
                "property": False,
                "async": False,
                "gen": False,
                "local": True,
            },
            "factory_constructor": {
                "signature": "(size=5)",
                "property": False,
                "async": False,
                "gen": False,
                "local": False,
            },
            "size_minus_cpus": {
                "signature": "(self)",
                "property": False,
                "async": False,
                "gen": False,
                "local": False,
            },
            "slow_iter": {
                "signature": "(self)",
                "property": False,
                "async": False,
                "gen": True,
                "local": False,
            },
        }

        arr = SlowNumpy(size=5)
        assert set(arr.signature) == {
            "size",
            "_hidden_1",
            "slow_iter",
            "size_minus_cpus",
            "arr",
            "cpu_count",
            "factory_constructor",
        }

        df = SlowPandas(size=10)
        assert df.signature == {
            "_hidden_1": {
                "async": False,
                "gen": False,
                "local": False,
                "property": True,
                "signature": None,
            },
            "cpu_count": {
                "async": False,
                "gen": False,
                "local": True,
                "property": False,
                "signature": "(self, local=True)",
            },
            "cpu_count_async": {
                "async": True,
                "gen": False,
                "local": True,
                "property": False,
                "signature": "(self, local=True)",
            },
            "df": {
                "async": False,
                "gen": False,
                "local": False,
                "property": True,
                "signature": None,
            },
            "size": {
                "async": False,
                "gen": False,
                "local": False,
                "property": True,
                "signature": None,
            },
            "slow_iter": {
                "async": False,
                "gen": True,
                "local": False,
                "property": False,
                "signature": "(self)",
            },
            "slow_iter_async": {
                "async": True,
                "gen": True,
                "local": False,
                "property": False,
                "signature": "(self)",
            },
        }

        RemoteCalc = rh.module(Calculator)
        assert set(RemoteCalc.signature.keys()) == {
            "summer",
            "sub",
            "divider",
            "mult",
            "importer",
        }
        assert RemoteCalc.signature == {
            "divider": {
                "signature": "(self, a: int, b: int)",
                "property": False,
                "async": False,
                "gen": False,
                "local": False,
            },
            "importer": {
                "signature": None,
                "property": True,
                "async": False,
                "gen": False,
                "local": False,
            },
            "mult": {
                "signature": "(self, a: int, b: int)",
                "property": False,
                "async": False,
                "gen": False,
                "local": False,
            },
            "sub": {
                "signature": "(self, a: int, b: int)",
                "property": False,
                "async": False,
                "gen": False,
                "local": False,
            },
            "summer": {
                "signature": "(self, a: int, b: int)",
                "property": False,
                "async": False,
                "gen": False,
                "local": False,
            },
        }

    @pytest.mark.level("thorough")
    def test_shared_readonly(
        self, ondemand_https_cluster_with_auth, local_test_account_cluster_public_key
    ):
        if ondemand_https_cluster_with_auth.address == "localhost":
            pytest.skip("Skipping sharing test on local cluster")

        size = 3
        remote_df = SlowPandas(size=size).to(
            ondemand_https_cluster_with_auth, name="remote_df"
        )
        remote_df.share(
            users=["info@run.house"],
            access_level="read",
            notify_users=False,
        )

        with test_account():
            test_load_and_use_readonly_module(
                mod_name=remote_df.rns_address, cpu_count=2, size=size
            )

        cpu_count = int(
            ondemand_https_cluster_with_auth.run_python(
                ["import os; print(os.cpu_count())"]
            )[0][1]
        )
        test_fn = rh.fn(test_load_and_use_readonly_module).to(
            local_test_account_cluster_public_key
        )
        test_fn(mod_name=remote_df.rns_address, cpu_count=cpu_count, size=size)


def test_load_and_use_readonly_module(mod_name, cpu_count, size=3):
    remote_df = rh.module(name=mod_name)
    # Check that module is readonly and cluster is not set
    assert isinstance(remote_df.system, str)
    assert remote_df.access_level == "read"

    assert remote_df.remote.size == size
    assert len(remote_df.remote.df) == size
    assert remote_df.remote._hidden_1 == "hidden"

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
    for i in range(size - 1):
        assert f"Hello from the cluster stdout! {i}" in out
        assert f"Hello from the cluster logs! {i}" in out

    print(remote_df.cpu_count())
    assert remote_df.cpu_count() == os.cpu_count()
    print(remote_df.cpu_count(local=False))
    assert remote_df.cpu_count(local=False) == cpu_count

    # Test setting and getting properties
    df = remote_df.remote.df
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (3, 3)
    assert df.loc[0, 0] == 0
    assert df.loc[2, 2] == 2

    remote_df.size = 20
    assert remote_df.remote.size == 20
    remote_df.size = size  # reset to original value for second test


if __name__ == "__main__":
    unittest.main()
