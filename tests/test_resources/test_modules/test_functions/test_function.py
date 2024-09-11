import multiprocessing
import os
import time

import pytest
import requests

import runhouse as rh
from runhouse.globals import rns_client
from runhouse.logger import get_logger

from tests.utils import friend_account

logger = get_logger(__name__)


def get_remote_func_name(test_folder):
    return f"@/{test_folder}/remote_function"


def call_function(fn, **kwargs):
    return fn(**kwargs)


def summer(a, b):
    return a + b


async def async_summer(a, b):
    return a + b


def returns_coroutine(a, b):
    return async_summer(a, b)


async def async_returns_coroutine(a, b):
    return async_summer(a, b)


def np_array(list):
    import numpy as np

    return np.array(list)


def np_summer(a, b):
    import numpy as np

    print(f"Summing {a} and {b}")
    return int(np.array([a, b]).sum())


def multiproc_np_sum(inputs):
    print(f"CPUs: {os.cpu_count()}")
    # See https://pythonspeed.com/articles/python-multiprocessing/
    # and https://github.com/pytorch/pytorch/issues/3492
    with multiprocessing.get_context("spawn").Pool() as P:
        return P.starmap(np_summer, inputs)


def getpid(a=0):
    return os.getpid() + a


def slow_generator(size):
    logger.info("Hello from the cluster logs!")
    print("Hello from the cluster stdout!")
    arr = []
    for i in range(size):
        time.sleep(1)
        logger.info(f"Hello from the cluster logs! {i}")
        print(f"Hello from the cluster stdout! {i}")
        arr += [i]
        yield f"Hello from the cluster! {arr}"


async def async_slow_generator(size):
    logger.info("Hello from the cluster logs!")
    print("Hello from the cluster stdout!")
    arr = []
    for i in range(size):
        time.sleep(1)
        logger.info(f"Hello from the cluster logs! {i}")
        print(f"Hello from the cluster stdout! {i}")
        arr += [i]
        yield f"Hello from the cluster! {arr}"


@pytest.mark.functiontest
class TestFunction:

    # ---------- Minimal / Local Level Tests (aka not unittests) ----------

    @pytest.mark.level("local")
    def test_create_function_from_name_local(self, cluster):
        local_name = "~/local_function"
        local_sum = rh.function(summer).to(cluster).save(local_name)
        del local_sum

        remote_sum = rh.function(name=local_name)
        res = remote_sum(1, 5)
        assert res == 6

        remote_sum.delete_configs()
        assert rh.exists(local_name) is False

    @pytest.mark.level("local")
    def test_create_function_from_rns(self, cluster, test_rns_folder):
        remote_func_name = get_remote_func_name(test_rns_folder)
        if cluster.on_this_cluster():
            pytest.mark.skip("Function on local cluster cannot be loaded from RNS.")

        remote_sum = rh.function(summer).to(cluster).save(remote_func_name)
        del remote_sum

        # reload the function
        remote_sum = rh.function(name=remote_func_name)
        res = remote_sum(1, 5)
        assert res == 6

        remote_sum.delete_configs()
        assert not rh.exists(remote_func_name)

    @pytest.mark.level("local")
    @pytest.mark.asyncio
    async def test_async_function(self, cluster):
        remote_sum = rh.function(async_summer).to(cluster)
        res = await remote_sum(1, 5)
        assert res == 6

    @pytest.mark.level("local")
    def test_get_function_history(self, cluster, test_rns_folder):
        remote_func_name = get_remote_func_name(test_rns_folder)

        # reload the function from RNS
        remote_sum = rh.function(summer).to(cluster).save(remote_func_name)

        history = remote_sum.history()
        assert history

    @pytest.mark.level("local")
    def test_function_in_new_env_with_multiprocessing(self, cluster):
        multiproc_remote_sum = rh.function(multiproc_np_sum, name="test_function").to(
            cluster, env=rh.env(reqs=["numpy"], name="numpy_env")
        )

        summands = [[1, 3], [2, 4], [3, 5]]
        res = multiproc_remote_sum(summands)

        assert res == [4, 6, 8]

    @pytest.mark.level("local")
    def test_generator(self, cluster):
        remote_slow_generator = rh.function(slow_generator).to(cluster)
        results = []
        for val in remote_slow_generator(5):
            assert val
            print(val)
            results += [val]
        assert len(results) == 5

    @pytest.mark.level("local")
    @pytest.mark.asyncio
    async def test_async_generator(self, cluster):
        remote_slow_generator = rh.function(async_slow_generator).to(cluster)
        results = []
        async for val in remote_slow_generator(3):
            assert val
            print(val)
            results += [val]
        assert len(results) == 3

    @pytest.mark.skip("TODO fix following local daemon refactor.")
    @pytest.mark.level("local")
    def test_remotes(self, cluster):
        pid_fn = rh.function(getpid).to(cluster)

        pid_key = pid_fn.run()
        time.sleep(1)
        pid_res = cluster.get(pid_key).data
        assert pid_res > 0

        # Test passing a remote into a normal call
        pid_blob = pid_fn.call.remote()
        pid_res = cluster.get(pid_blob.name).data
        assert pid_res > 0
        pid_res = pid_blob.fetch()
        assert pid_res > 0

    @pytest.mark.skip("Install is way too heavy, choose a lighter example")
    @pytest.mark.level("local")
    def test_function_git_fn(self, cluster):
        remote_parse = rh.function(
            fn="https://github.com/huggingface/diffusers/blob/"
            "main/examples/dreambooth/train_dreambooth.py:parse_args",
            system=cluster,
            env=[
                "torch==1.12.1 --verbose",
                "torchvision==0.13.1",
                "transformers",
                "datasets",
                "evaluate",
                "accelerate",
                "pip:./diffusers --verbose",
            ],
        )
        args = remote_parse(
            input_args=[
                "--pretrained_model_name_or_path",
                "stabilityai/stable-diffusion-2-base",
                "--instance_data_dir",
                "remote_image_dir",
                "--instance_prompt",
                "a photo of sks person",
            ]
        )
        assert (
            args.pretrained_model_name_or_path == "stabilityai/stable-diffusion-2-base"
        )

    @pytest.mark.skip("Fix .run following local daemon refactor.")
    @pytest.mark.level("local")
    def test_list_keys(self, cluster):
        pid_fn = rh.function(getpid).to(cluster)

        pid_obj1 = pid_fn.run()
        time.sleep(1)
        pid_obj2 = pid_fn.run()
        time.sleep(1)
        current_jobs = cluster.keys()
        assert set([pid_obj1, pid_obj2]).issubset(current_jobs)

        pid_obj3 = pid_fn.call.remote()
        time.sleep(1)
        pid_obj4 = pid_fn.call.remote()
        time.sleep(1)
        current_jobs = cluster.keys()
        assert set(
            [pid_obj3.name.replace("~/", ""), pid_obj4.name.replace("~/", "")]
        ).issubset(current_jobs)

    @pytest.mark.skip("Does not work properly following Module refactor.")
    @pytest.mark.level("local")
    def test_function_queueing(self, cluster):
        pid_fn = rh.function(getpid).to(cluster)

        pids = [pid_fn.enqueue(resources={"num_cpus": 1}) for _ in range(10)]
        assert len(pids) == 10

    @pytest.mark.skip("Not working properly.")
    @pytest.mark.level("minimal")
    def test_function_external_fn(self, cluster):
        """Test functioning a module from reqs, not from working_dir"""
        import numpy as np

        re_fn = rh.function(np.sum).to(cluster, env=["numpy"])
        res = re_fn(np.arange(5))
        assert int(res) == 10

    @pytest.mark.skip("Runs indefinitely.")
    # originally used ondemand_aws_docker_cluster, therefore marked as minimal
    @pytest.mark.level("minimal")
    def test_notebook(self, cluster):
        nb_sum = lambda x: multiproc_np_sum(x)
        re_fn = rh.function(nb_sum).to(cluster, env=["numpy"])

        re_fn.notebook()
        summands = list(zip(range(5), range(4, 9)))
        res = re_fn(summands)

        assert res == [4, 6, 8, 10, 12]
        re_fn.delete_configs()

    @pytest.mark.skip("Runs indefinitely.")
    def test_ssh(self):
        # TODO do this properly
        my_function = rh.function(name="local_function")
        my_function.ssh()
        assert True

    @pytest.mark.level("local")
    def test_share_and_revoke_function(self, cluster, test_rns_folder):
        remote_func_name = get_remote_func_name(test_rns_folder)

        # TODO: refactor in order to test the function.share() method.
        my_function = rh.function(fn=summer).to(cluster)
        my_function.set_endpoint(f"{cluster.endpoint()}/{my_function.name}")
        my_function.save(remote_func_name)

        my_function.share(
            users=["info@run.house"],
            access_level="read",
            notify_users=False,
        )
        with friend_account():
            my_function = rh.function(name=my_function.rns_address)
            res = my_function(1, 2)
            assert res == 3

        my_function.revoke(users=["info@run.house"])

        with pytest.raises(Exception):
            with friend_account():
                my_function = rh.function(name=my_function.rns_address)
                my_function(1, 2)

    @pytest.mark.level("release")
    def test_load_function_in_new_cluster(
        self, ondemand_aws_docker_cluster, static_cpu_pwd_cluster, test_rns_folder
    ):
        remote_func_name = get_remote_func_name(test_rns_folder)

        ondemand_aws_docker_cluster.save(
            f"@/{ondemand_aws_docker_cluster.name}"
        )  # Needs to be saved to rns, right now has a local name by default
        remote_sum = (
            rh.function(summer).to(ondemand_aws_docker_cluster).save(remote_func_name)
        )

        static_cpu_pwd_cluster.sync_secrets(["sky"])
        remote_python = (
            "import runhouse as rh; "
            f"remote_sum = rh.function(name='{remote_func_name}'); "
            "res = remote_sum(1, 5); "
            "assert res == 6"
        )
        res = static_cpu_pwd_cluster.run_python([remote_python], stream_logs=True)
        assert res[0][0] == 0

        remote_sum.delete_configs()

    @pytest.mark.level("release")
    def test_nested_diff_clusters(
        self, ondemand_aws_docker_cluster, static_cpu_pwd_cluster
    ):
        summer_cpu = rh.function(summer).to(ondemand_aws_docker_cluster)
        call_function_diff_cpu = rh.function(call_function).to(static_cpu_pwd_cluster)

        kwargs = {"a": 1, "b": 5}
        res = call_function_diff_cpu(summer_cpu, **kwargs)
        assert res == 6

    @pytest.mark.level("local")
    def test_nested_same_cluster(self, cluster):
        # When the system of a function is set to the cluster that the function is being called on, we run the function
        # locally and not via an RPC call
        summer_cpu = rh.function(fn=summer).to(system=cluster)
        call_function_cpu = rh.function(fn=call_function).to(system=cluster)

        kwargs = {"a": 1, "b": 5}
        res = call_function_cpu(summer_cpu, **kwargs)
        assert res == 6

    @pytest.mark.level("local")
    def test_http_url(self, cluster):
        remote_sum = rh.function(summer).to(cluster).save("@/remote_function")
        ssh_creds = cluster.creds_values
        addr = remote_sum.endpoint()
        auth = (
            (ssh_creds.get("ssh_user"), ssh_creds.get("password"))
            if ssh_creds.get("password")
            else None
        )
        verify = cluster.client.verify
        sum1 = requests.post(
            url=f"{addr}/call",
            json={
                "data": {
                    "args": [1, 2],
                    "kwargs": {},
                },
                "serialization": None,
                "run_name": "test_http_url",
            },
            headers=rns_client.request_headers(cluster.rns_address)
            if cluster.den_auth
            else None,
            auth=auth,
            verify=verify,
        ).json()
        assert sum1 == 3
        sum2 = requests.post(
            url=f"{addr}/call",
            json={
                "data": {
                    "args": [],
                    "kwargs": {"a": 1, "b": 2},
                },
                "serialization": None,
                "run_name": "test_http_url",
            },
            headers=rns_client.request_headers(cluster.rns_address)
            if cluster.den_auth
            else None,
            auth=auth,
            verify=verify,
        ).json()
        assert sum2 == 3

    @pytest.mark.skip(
        "Clean up following local daemon refactor. Function probably doesn't need .get anymore."
    )
    @pytest.mark.level("local")
    def test_get(self, cluster):
        my_summer = rh.function(summer).to(system=cluster)
        my_summer_run = my_summer.run(5, 6)
        time.sleep(2)
        my_summer_res = my_summer.get(my_summer_run)
        time.sleep(2)
        res = my_summer_res.resolved_state()
        # TODO: should not use .resolved_state(), need to be fixed.
        assert res == 11

    @pytest.mark.skip("Fix and clean up following local daemon refactor.")
    @pytest.mark.level("local")
    def test_get_or_call(self, cluster):
        remote_summer = rh.function(summer).to(system=cluster)
        # first run - need to implement later test for return value verification.
        run_name = f"my_run_{time.time_ns()}"
        res = remote_summer.get_or_call(run_name, False, True, 12, 5)
        assert res == 17

        # not first run - implement later (raises an exception).
        # remote_run = remote_summer.run(2, 3)
        get_or_call_res = remote_summer.get_or_call(run_name=run_name)
        # TODO: should not use .resolved_state(), need to be fixed.
        assert get_or_call_res == 17

    # ---------- Unittests ----------
    @pytest.fixture(autouse=True)
    def init_fixtures(self):
        self.function = rh.function(summer)

    @pytest.mark.level("unit")
    def test_get_unittest(self, mocker):
        mock_function = mocker.patch("runhouse.Function.get")
        mock_function.return_value = 5
        response = self.function.get(run_key="my_mocked_run")
        assert response == 5
        mock_function.assert_called_once_with(run_key="my_mocked_run")

    @pytest.mark.level("unit")
    def test_call_unittest(self, mocker):
        mock_function = mocker.patch("runhouse.Function.call")
        mock_function.return_value = 5
        response = self.function.call(3, 2)
        assert response == 5
        mock_function.assert_called_once_with(3, 2)

    @pytest.mark.level("unit")
    def test_to_unittest(self, mocker):
        mock_function = mocker.patch("runhouse.Function.to")
        # Create a Mock instance for cluster
        mock_cluster = mocker.patch("runhouse.cluster")
        local_cluster = mock_cluster(name="local_cluster")
        self.function.system = local_cluster
        mock_function.return_value = self.function

        # Call the method under test
        response_to = self.function.to(local_cluster)

        # Assertions

        assert response_to.system == local_cluster
        mock_function.assert_called_once_with(local_cluster)
        # Reset the system attribute
        self.function.system = None

    @pytest.mark.level("unit")
    def test_map_unittest(self, mocker):
        # TODO: change the test once the implementation change is ready, if necessary
        mock_function = mocker.patch("runhouse.Function.map")
        mock_function.return_value = [3, 7, 11]
        response = self.function.map([[1, 3, 5], [2, 4, 6]])
        mock_function.assert_called_once_with([[1, 3, 5], [2, 4, 6]])
        assert response == [3, 7, 11]

    @pytest.mark.level("unit")
    def test_starmap_unittest(self, mocker):
        # TODO: change the test once the implementation change is ready, if necessary
        mock_function = mocker.patch("runhouse.Function.starmap")
        mock_function.return_value = [3, 7, 11]
        response = self.function.starmap([(1, 2), (3, 4), (5, 6)])
        mock_function.assert_called_once_with([(1, 2), (3, 4), (5, 6)])
        assert response == [3, 7, 11]

    @pytest.mark.level("unit")
    def test_get_or_call_unittest(self, mocker):
        mock_function = mocker.patch("runhouse.Function.get_or_call")
        mock_function.return_value = 17
        response_first = self.function.get_or_call(
            "my_run_first_time", False, True, 12, 5
        )
        mock_function.assert_called_with("my_run_first_time", False, True, 12, 5)
        assert response_first == 17
        second_response = self.function.get_or_call("my_run_first_time")
        assert second_response == 17

    @pytest.mark.level("local")
    @pytest.mark.asyncio
    async def test_run_async_permutations(self, cluster):
        async_summer_remote = rh.function(async_summer).to(cluster)
        summer_remote = rh.function(summer).to(cluster)

        # Test things work natively
        assert await async_summer_remote(2, 3) == 5
        assert summer_remote(2, 3) == 5

        # Test that can call with run_async set
        assert await async_summer_remote(2, 3, run_async=True) == 5
        assert async_summer_remote(2, 3, run_async=False) == 5
        assert await summer_remote(2, 3, run_async=True) == 5
        assert summer_remote(2, 3, run_async=False) == 5

    @pytest.mark.level("local")
    @pytest.mark.asyncio
    async def test_returns_coroutine(self, cluster):
        returns_coroutine_remote = rh.function(returns_coroutine).to(cluster)
        async_returns_coroutine_remote = rh.function(async_returns_coroutine).to(
            cluster
        )

        # Test that can call with run_async set
        future_module = returns_coroutine_remote(2, 3)
        assert future_module.__class__.__name__ == "FutureModule"
        assert await future_module == 5

        # Test that can call with run_async set to True
        future_module = await returns_coroutine_remote(2, 3, run_async=True)
        assert future_module.__class__.__name__ == "FutureModule"
        assert await future_module == 5

        future_module = await async_returns_coroutine_remote(2, 3)
        assert future_module.__class__.__name__ == "FutureModule"
        assert await future_module == 5

        # Test that can call with run_async set to False
        future_module = async_returns_coroutine_remote(2, 3, run_async=False)
        assert future_module.__class__.__name__ == "FutureModule"
        assert await future_module == 5

    @pytest.mark.level("local")
    def test_send_function_to_fresh_env(self, cluster):
        env = rh.env(name="fresh_env", reqs=["numpy"])
        summer_remote = rh.function(summer).to(cluster, env=env)
        summer_remote(2, 3)
