import inspect
import json
import os
import site
import time

from importlib import reload as importlib_reload

import numpy as np
import pandas as pd
import pytest

import runhouse as rh
from runhouse import Package
from runhouse.constants import TEST_ORG
from runhouse.logger import get_logger
from runhouse.utils import capture_stdout

logger = get_logger(__name__)

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
    def local_home(cls, local=True):
        return os.path.expanduser("~")

    @classmethod
    def home(cls):
        return os.path.expanduser("~")

    @classmethod
    def cpu_count(cls):
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

    @classmethod
    def local_home(cls, local=True):
        return os.path.expanduser("~")

    @classmethod
    def home(cls):
        return os.path.expanduser("~")

    @classmethod
    def cpu_count(cls):
        return os.cpu_count()

    def size_minus_cpus(self):
        return self.size - self.cpu_count()

    async def cpu_count_async(self):
        return os.cpu_count()


class ModuleConstructingOtherModule:
    def construct_and_get_env(self):
        remote_calc = rh.module(Calculator)()
        return remote_calc.env


class Calculator:
    importer = "Calculators Inc"

    def __init__(self, owner=None):
        self.model = "Casio"
        self._release_year = 2023
        self.owner = owner

    async def asummer(self, a: int, b: int):
        return a + b

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


class LotsOfAsync:
    async def asum(self, a: int, b: int):
        return a + b

    def returns_coroutine(self, a: int, b: int):
        return self.asum(a, b)

    async def async_returns_coroutine(self, a: int, b: int):
        return self.asum(a, b)


class ConstructorModule:
    def construct_module_on_cluster(self):
        calc_module = rh.module(Calculator)()
        assert calc_module.summer(2, 3) == 5


def load_and_use_readonly_module(mod_name, cpu_count, size=3):
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
    with capture_stdout() as stdout:
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


async def load_and_use_calculator_module(mod_name):
    remote_calc = rh.module(name=mod_name)

    assert remote_calc.remote.model == "Casio"
    assert remote_calc.remote._release_year == 2023

    assert remote_calc.summer(2, 3) == 5
    assert await remote_calc.asummer(2, 3) == 5
    assert remote_calc.mult(3, 7) == 21

    assert remote_calc.sub(33, 5) == 28

    assert remote_calc.divider(16, 2) == 8


def construct_and_use_calculator_module(mod_name):
    remote_calc_class = rh.module(name=mod_name)
    remote_calc = remote_calc_class()

    assert remote_calc.remote.model == "Casio"
    assert remote_calc.remote._release_year == 2023

    assert remote_calc.summer(2, 3) == 5
    assert remote_calc.mult(3, 7) == 21
    assert remote_calc.sub(33, 5) == 28
    assert remote_calc.divider(16, 2) == 8

    return remote_calc


@pytest.mark.moduletest
class TestModule:

    # --------- integration tests ---------
    @pytest.mark.level("local")
    def test_call_module_method(self, cluster):
        cluster.put("numpy_pkg", Package.from_string("numpy"))

        # Test for property
        res = cluster.call("numpy_pkg", "config", stream_logs=True)
        numpy_config = Package.from_string("numpy").config()
        assert res
        assert isinstance(res, dict)
        assert res == numpy_config

        # Test iterator
        cluster.put("config_dict", list(numpy_config.keys()))
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
        remote_instance = RemoteClass(size=size, name="remote_instance1")

        # Test that naming works properly, and "class" module was unaffacted
        assert remote_instance.name == "remote_instance1"
        assert RemoteClass.name == "SlowNumpyArray"

        # Test that module was initialized correctly on the cluster
        assert remote_instance.system == cluster
        assert remote_instance.remote.size == size
        assert all(remote_instance.remote.arr == np.zeros(size))
        assert remote_instance.remote._hidden_1 == "hidden"

        results = []
        out = ""
        with capture_stdout() as stdout:
            for val in remote_instance.slow_iter():
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

        local_home = os.path.expanduser("~")
        if cluster.on_this_cluster():
            remote_home = local_home
        else:
            remote_home = cluster.run(["echo $HOME"])[0][1].strip()

        # Test classmethod on remote class
        assert RemoteClass.local_home() == local_home
        assert RemoteClass.home() == remote_home

        # Test classmethod on remote instance
        assert remote_instance.local_home() == local_home
        assert remote_instance.home() == remote_home

        # Test instance method
        assert remote_instance.size_minus_cpus() == size - remote_instance.cpu_count()

        # Test remote getter
        arr = remote_instance.remote.arr
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (size,)
        assert arr[0] == 0
        assert arr[2] == 2

        # Test remote setter
        remote_instance.remote.size = 20
        assert remote_instance.remote.size == 20

        # Test creating a second instance of the same class
        remote_instance2 = RemoteClass(size=30, name="remote_instance2")
        assert remote_instance2.system == cluster
        assert remote_instance2.remote.size == 30

        # TODO fix factory method support
        # Test creating a third instance with the factory method
        remote_instance3 = RemoteClass.factory_constructor.remote(
            size=40, run_name="remote_instance3"
        )

        remote_instance_config = remote_instance3.system.config()
        cluster_config = cluster.config()

        assert remote_instance_config == cluster_config
        assert remote_instance3.remote.size == 40
        assert remote_instance3.home() == remote_home

        # Make sure first array and class are unaffected by this change
        assert remote_instance.remote.size == 20
        assert RemoteClass.home() == remote_home

        # Test that resolve() replaces the object properly on the cluster
        # TODO deprecate resolve and test that stub module is deserialized properly on the cluster as the true object
        #  in the obj_store
        # helper = rh.function(resolve_test_helper).to(cluster)
        # resolved_obj = helper(remote_instance.resolve())
        # assert resolved_obj.__class__.__name__ == "SlowNumpyArray"
        # assert not hasattr(resolved_obj, "config")
        # assert resolved_obj.size == 20
        # assert list(resolved_obj.arr) == [0, 1, 2]

    @pytest.mark.parametrize("env", [None])
    @pytest.mark.level("local")
    def test_module_from_subclass(self, cluster, env):
        size = 3
        remote_instance = SlowPandas(size=size).to(cluster, env)
        assert remote_instance.system == cluster

        # Test that module was initialized correctly on the cluster
        assert remote_instance.remote.size == size
        assert len(remote_instance.remote.df) == size
        assert remote_instance.remote._hidden_1 == "hidden"

        results = []
        # Capture stdout to check that it's working
        out = ""
        with capture_stdout() as stdout:
            for i, val in enumerate(remote_instance.slow_iter()):
                assert val
                print(val)
                results += [val]
                out = out + str(stdout)
        assert len(results) == 3

        # Check that stdout was captured. Skip the last result because sometimes we
        # don't catch it and it makes the test flaky.
        for i in range(remote_instance.size - 1):
            assert f"Hello from the cluster stdout! {i}" in out
            assert f"Hello from the cluster logs! {i}" in out

        local_home = os.path.expanduser("~")
        if cluster.on_this_cluster():
            remote_home = local_home
        else:
            remote_home = cluster.run(["echo $HOME"])[0][1].strip()

        # Test classmethod on remote instance
        assert remote_instance.local_home() == local_home
        assert remote_instance.home() == remote_home

        # Test instance method
        assert remote_instance.size_minus_cpus() == size - remote_instance.cpu_count()

        # Test setting and getting properties
        df = remote_instance.remote.df
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (3, 3)
        assert df.loc[0, 0] == 0
        assert df.loc[2, 2] == 2

        remote_instance.remote.size = 20
        assert remote_instance.remote.size == 20

        del remote_instance

        # Test get_or_to
        remote_instance = SlowPandas(size=3).get_or_to(
            cluster, env=env, name="SlowPandas"
        )
        assert remote_instance.system.config() == cluster.config()
        # Check that size is unchanged from when we set it to 20 above
        assert remote_instance.remote.size == 20

        # Test that resolve() has no effect
        # TODO deprecate resolve and test that stub module is deserialized properly on the cluster as the true object
        #  in the obj_store
        # helper = rh.function(resolve_test_helper).to(cluster, env=rh.Env())
        # resolved_obj = helper(remote_instance.resolve())
        # assert resolved_obj.__class__.__name__ == "SlowPandas"
        # assert resolved_obj.size == 20  # resolved_obj.remote.size causing an error
        # assert resolved_obj.config() == remote_instance.config()

    @pytest.mark.parametrize("env", [None])
    @pytest.mark.level("local")
    @pytest.mark.asyncio
    async def test_module_from_subclass_async(self, cluster, env):
        remote_df = SlowPandas(size=3).to(cluster, env)
        assert remote_df.system == cluster
        results = []
        # Capture stdout to check that it's working
        out = ""
        with capture_stdout() as stdout:
            async for val in remote_df.slow_iter_async():
                assert val
                print(val)
                results += [val]
                out = out + str(stdout)
        assert len(results) == 3

        # Check that stdout was captured. Skip the last result because sometimes we
        # don't catch it and it makes the test flaky.
        # for i in range(remote_df.size - 1):
        #     assert f"Hello from the cluster stdout! {i}" in out
        #     assert f"Hello from the cluster logs! {i}" in out

        if cluster.on_this_cluster():
            cpu_count = os.cpu_count()
        else:
            cpu_count = int(
                cluster.run_python(["import os; print(os.cpu_count())"])[0][1]
            )
        assert await remote_df.cpu_count_async() == cpu_count

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
    @pytest.mark.level("local")
    @pytest.mark.asyncio
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

        # TODO: This line used to end in .fetch() . We need to update when we fix the .resolve()/serialization behavior
        remote_df_sync = SlowPandas(size=4).to(cluster, env)
        assert remote_df_sync.size == 4

        if cluster.on_this_cluster():
            cpu_count = os.cpu_count()
        else:
            cpu_count = int(
                cluster.run_python(["import os; print(os.cpu_count())"])[0][1]
            )
        assert remote_df_sync.cpu_count() == cpu_count

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
    @pytest.mark.skip(
        "TODO: Fix module serialization / resolve logic due to breakage "
        "when server is running a different Python version"
    )
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
        if cluster.on_this_cluster():
            pytest.skip("Skipping sharing test on local cluster")

        users = ["josh@run.house"]
        remote_calc = rh.module(Calculator).to(cluster).save(name="rh_remote_calc")
        added_users, new_users, _ = remote_calc.share(
            users=users, notify_users=False, access_level="write"
        )
        assert remote_calc.name == "rh_remote_calc"
        assert added_users == {users[0]: "write"} or added_users == {}
        assert new_users == {}

    @pytest.mark.parametrize("env", [None])
    @pytest.mark.level("local")
    @pytest.mark.asyncio
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
        assert set(SlowNumpy.signature()) == {
            "slow_iter",
            "home",
            "cpu_count",
            "size_minus_cpus",
            "factory_constructor",
        }
        # Check that rich signature is json-able and has the same number of keys as the regular signature
        assert len(json.loads(json.dumps(SlowNumpy.signature(rich=True)))) == 5

        arr = SlowNumpy(size=5)
        assert set(arr.signature()) == {
            "slow_iter",
            "home",
            "cpu_count",
            "size_minus_cpus",
            "factory_constructor",
        }

        df = SlowPandas(size=10)
        assert set(df.signature()) == {
            "slow_iter",
            "slow_iter_async",
            "home",
            "cpu_count",
            "cpu_count_async",
            "size_minus_cpus",
        }
        assert len(json.loads(json.dumps(df.signature(rich=True)))) == 6

        RemoteCalc = rh.module(Calculator)
        assert set(RemoteCalc.signature()) == {
            "summer",
            "asummer",
            "sub",
            "divider",
            "mult",
        }

    # SSH Clusters we can't share read only modules, since we'd have to give them an SSH key
    # in order to access the cluster. TLS clusters we can't connect to the Docker instances
    # easily.
    @pytest.mark.level("local")
    @pytest.mark.skip("Doesn't work with local clusters.")
    @pytest.mark.asyncio
    async def test_share_module_read_only(self, cluster):
        from tests.utils import friend_account

        remote_calc = rh.module(Calculator)().to(cluster, name="remote_calc_instance")

        remote_calc.share(
            users=["info@run.house"],
            access_level="read",
            notify_users=False,
        )

        with friend_account():
            await load_and_use_calculator_module(mod_name=remote_calc.rns_address)

    @pytest.mark.level("release")
    @pytest.mark.asyncio
    async def test_share_module_read_only_inter_cluster(
        self,
        ondemand_aws_https_cluster_with_auth,
        friend_account_logged_in_docker_cluster_pk_ssh,
    ):

        # Make sure cluster is not shared
        ondemand_aws_https_cluster_with_auth.revoke(users=["info@run.house"])

        # Note that this module is already constructed before it is put on the cluster
        remote_calc = (
            rh.module(Calculator)()
            .to(ondemand_aws_https_cluster_with_auth, name="remote_calc_instance")
            .save()
        )

        test_fn_remote = rh.function(load_and_use_calculator_module).to(
            friend_account_logged_in_docker_cluster_pk_ssh
        )

        remote_calc.revoke(users=["info@run.house"])

        # Expect this to raise a ValueError
        with pytest.raises(ValueError):
            await test_fn_remote(remote_calc.rns_address)

        remote_calc.share(
            users=["info@run.house"],
            access_level="read",
            notify_users=False,
        )

        await test_fn_remote(remote_calc.rns_address)

        remote_calc.revoke(users=["info@run.house"])

        # Expect this to raise a ValueError
        with pytest.raises(ValueError):
            await test_fn_remote(remote_calc.rns_address)

    @pytest.mark.level("release")
    def test_share_module_with_generator_read_only_inter_cluster(
        self,
        ondemand_aws_https_cluster_with_auth,
        friend_account_logged_in_docker_cluster_pk_ssh,
    ):
        # from tests.utils import friend_account

        size = 3
        remote_df = SlowPandas(size=size).to(
            ondemand_aws_https_cluster_with_auth, name="remote_df_ondemand_cluster"
        )

        remote_df.share(
            users=["info@run.house"],
            access_level="read",
            notify_users=False,
        )

        cpu_count = int(
            ondemand_aws_https_cluster_with_auth.run_python(
                ["import os; print(os.cpu_count())"]
            )[0][1]
        )
        test_fn = rh.fn(load_and_use_readonly_module).to(
            friend_account_logged_in_docker_cluster_pk_ssh
        )
        test_fn(mod_name=remote_df.rns_address, cpu_count=cpu_count, size=size)

    @pytest.mark.level("local")
    def test_construct_shared_module(
        self,
        cluster,
    ):
        if TEST_ORG in cluster.rns_address:
            pytest.skip(
                "Skipping test if the cluster is in the test org, as we can't share outside of the org."
            )

        from tests.utils import friend_account

        remote_calc_unconstructed = (
            rh.module(Calculator)
            .to(cluster, name="remote_calculator_unconstructed")
            .save()
        )

        cluster.revoke(
            users=["info@run.house"],
        )

        remote_calc_unconstructed.share(
            users=["info@run.house"],
            access_level="read",
            notify_users=False,
        )

        with pytest.raises(ValueError):
            with friend_account():
                construct_and_use_calculator_module(
                    mod_name=remote_calc_unconstructed.rns_address
                )

        cluster.share(
            users=["info@run.house"],
            access_level="write",
            notify_users=False,
        )

        with friend_account():
            construct_and_use_calculator_module(
                mod_name=remote_calc_unconstructed.rns_address
            )

    @pytest.mark.skip(
        "Broken in CI after we allowed higher Ray versions, probably due to Pydantic."
    )
    @pytest.mark.level("unit")
    def test_openapi_spec_generation(self):
        from openapi_core import OpenAPI

        remote_calc = rh.module(Calculator)

        # Normally we'd send the calculator to a system, save it, and then call this
        # But for testing purposes we can just pass in a name so this can run as a unit test
        spec = remote_calc.openapi_spec(spec_name="CalculatorAPI")

        # This automatically validates the spec
        openapi = OpenAPI.from_dict(spec)
        assert openapi

        # Assert that the spec can be converted to json
        assert json.loads(json.dumps(spec))

    @pytest.mark.level("local")
    def test_dependency_exception(self, cluster):
        from .exception_module import ExceptionModule

        with pytest.raises(ModuleNotFoundError) as err:
            exc_module = ExceptionModule()
            exc_module.to(cluster)
        assert "plotly" in str(err.value)

    @pytest.mark.level("local")
    @pytest.mark.asyncio
    async def test_run_async_permutations(self, cluster):
        lots_of_async_remote = rh.module(LotsOfAsync).to(cluster)
        calculator_remote = rh.module(Calculator).to(cluster)

        lots_of_async_remote_instance = lots_of_async_remote()
        calculator_remote_instance = calculator_remote()

        # Test things work natively
        assert await lots_of_async_remote_instance.asum(2, 3) == 5
        assert calculator_remote_instance.summer(2, 3) == 5

        # Test that can call with run_async set
        assert await lots_of_async_remote_instance.asum(2, 3, run_async=True) == 5
        assert lots_of_async_remote_instance.asum(2, 3, run_async=False) == 5
        assert await calculator_remote_instance.summer(2, 3, run_async=True) == 5
        assert calculator_remote_instance.summer(2, 3, run_async=False) == 5

    @pytest.mark.level("local")
    @pytest.mark.asyncio
    async def test_returns_coroutine(self, cluster):
        lots_of_async_remote = rh.module(LotsOfAsync).to(cluster)
        lots_of_async_remote_instance = lots_of_async_remote()

        # Test that can call with run_async set
        future_module = lots_of_async_remote_instance.returns_coroutine(2, 3)
        assert future_module.__class__.__name__ == "FutureModule"
        assert await future_module == 5

        # Test that can call with run_async set to True
        future_module = await lots_of_async_remote_instance.returns_coroutine(
            2, 3, run_async=True
        )
        assert future_module.__class__.__name__ == "FutureModule"
        assert await future_module == 5

        future_module = await lots_of_async_remote_instance.async_returns_coroutine(
            2, 3
        )
        assert future_module.__class__.__name__ == "FutureModule"
        assert await future_module == 5

        # Test that can call with run_async set to False
        future_module = lots_of_async_remote_instance.async_returns_coroutine(
            2, 3, run_async=False
        )
        assert future_module.__class__.__name__ == "FutureModule"
        assert await future_module == 5

    @pytest.mark.level("local")
    def test_construct_module_on_cluster(self, cluster):
        env = rh.env(
            name="test_env",
            reqs=["pandas", "numpy"],
        )
        remote_constructor_module = rh.module(ConstructorModule)().to(cluster, env=env)
        remote_constructor_module.construct_module_on_cluster()

    @pytest.mark.level("local")
    def test_import_editable_package(self, cluster, installed_editable_package):

        importlib_reload(site)

        # Test that the editable package can be imported and used
        from test_fake_package import editable_package_function

        assert editable_package_function() == "Hello from the editable package!"

        # Now send this to the remote cluster and test that it can still be imported and used
        remote_editable_package_fn = rh.function(editable_package_function).to(cluster)
        assert remote_editable_package_fn() == "Hello from the editable package!"

        cluster.delete("editable_package_function")
        cluster.run("pip uninstall -y test_fake_package")

    @pytest.mark.level("local")
    def test_import_editable_package_from_new_env(
        self, cluster, installed_editable_package_copy
    ):
        importlib_reload(site)

        # Test that the editable package can be imported and used
        from test_fake_package_copy import TestModuleFromPackage

        assert (
            TestModuleFromPackage.hello_world()
            == "Hello from the editable package module!"
        )

        # Now send this to the remote cluster and test that it can still be imported and used
        env = rh.env(name="fresh_env", reqs=["numpy"])
        remote_editable_package_module = rh.module(TestModuleFromPackage).to(
            cluster, env=env
        )
        assert (
            remote_editable_package_module.hello_world()
            == "Hello from the editable package module!"
        )

    @pytest.mark.level("local")
    def test_module_constructed_on_cluster_is_in_same_env(self, cluster):
        env = rh.env(
            name="special_env",
            reqs=["pandas", "numpy"],
        )
        remote_module = rh.module(ModuleConstructingOtherModule).to(
            system=cluster, env=env
        )
        assert remote_module.construct_and_get_env().name == env.name
