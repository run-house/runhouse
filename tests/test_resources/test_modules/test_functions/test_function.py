import logging
import multiprocessing
import os
import time
import unittest
from unittest.mock import patch

import pytest
import ray.exceptions
import requests

import runhouse as rh

from tests.utils import test_account

REMOTE_FUNC_NAME = "@/remote_function"

logger = logging.getLogger(__name__)


def call_function(fn, **kwargs):
    return fn(**kwargs)


def summer(a, b):
    return a + b


async def async_summer(a, b):
    return a + b


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


def slow_getpid(a=0):
    time.sleep(10)
    return os.getpid() + a


class TestFunction:

    # ---------- Minimal / Local Level Tests (aka not unittests) ----------

    @pytest.mark.rnstest
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

    @pytest.mark.rnstest
    @pytest.mark.level("local")
    def test_create_function_from_rns(self, cluster):
        remote_sum = rh.function(summer).to(cluster).save(REMOTE_FUNC_NAME)
        del remote_sum

        # reload the function
        remote_sum = rh.function(name=REMOTE_FUNC_NAME)
        res = remote_sum(1, 5)
        assert res == 6

        remote_sum.delete_configs()
        assert not rh.exists(REMOTE_FUNC_NAME)

    @pytest.mark.asyncio
    @pytest.mark.level("local")
    async def test_async_function(self, cluster):
        remote_sum = rh.function(async_summer).to(cluster)
        res = await remote_sum(1, 5)
        assert res == 6

    @pytest.mark.rnstest
    @pytest.mark.level("local")
    def test_get_function_history(self, cluster):
        # reload the function from RNS
        remote_sum = rh.function(summer).to(cluster).save(REMOTE_FUNC_NAME)

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

    @pytest.mark.skip("Does not work properly following Module refactor.")
    @pytest.mark.clustertest
    @pytest.mark.level("local")
    def test_maps(self, cluster):
        pid_fn = rh.function(getpid, system=cluster)
        num_pids = [1] * 10
        pids = pid_fn.map(num_pids)
        assert len(set(pids)) > 1
        assert all(pid > 0 for pid in pids)

        pids = pid_fn.repeat(num_repeats=10)
        assert len(set(pids)) > 1
        assert all(pid > 0 for pid in pids)

        pids = [pid_fn.enqueue() for _ in range(10)]
        assert len(pids) == 10
        assert all(pid > 0 for pid in pids)

        re_fn = rh.function(summer, system=cluster)
        summands = list(zip(range(5), range(4, 9)))
        res = re_fn.starmap(summands)
        assert res == [4, 6, 8, 10, 12]

        alist, blist = range(5), range(4, 9)
        res = re_fn.map(alist, blist)
        assert res == [4, 6, 8, 10, 12]

    @pytest.mark.level("local")
    def test_generator(self, cluster):
        remote_slow_generator = rh.function(slow_generator).to(cluster)
        results = []
        for val in remote_slow_generator(5):
            assert val
            print(val)
            results += [val]
        assert len(results) == 5

    @pytest.mark.asyncio
    @pytest.mark.level("local")
    async def test_async_generator(self, cluster):
        remote_slow_generator = rh.function(async_slow_generator).to(cluster)
        results = []
        async for val in remote_slow_generator(5):
            assert val
            print(val)
            results += [val]
        assert len(results) == 5

    @pytest.mark.level("local")
    def test_remotes(self, cluster):
        pid_fn = rh.function(getpid, system=cluster)

        pid_key = pid_fn.run()
        time.sleep(1)
        pid_res = cluster.get(pid_key).data
        assert pid_res > 0

        # Test passing a remote into a normal call
        pid_blob = pid_fn.remote()
        time.sleep(1)
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

    @pytest.mark.level("local")
    def test_list_keys(self, cluster):
        pid_fn = rh.function(getpid).to(system=cluster)

        pid_obj1 = pid_fn.run()
        time.sleep(1)
        pid_obj2 = pid_fn.run()
        time.sleep(1)
        current_jobs = cluster.keys()
        assert set([pid_obj1, pid_obj2]).issubset(current_jobs)

        pid_obj3 = pid_fn.remote()
        time.sleep(1)
        pid_obj4 = pid_fn.remote()
        time.sleep(1)
        current_jobs = cluster.keys()
        assert set(
            [pid_obj3.name.replace("~/", ""), pid_obj4.name.replace("~/", "")]
        ).issubset(current_jobs)

    @pytest.mark.level("local")
    def test_cancel_jobs(self, cluster):
        pid_fn = rh.function(slow_getpid).to(cluster)

        pid_run1 = pid_fn.run(2)

        time.sleep(1)  # So the runkeys are more than 1 second apart
        pid_ref2 = pid_fn.run(5)

        print("Cancelling jobs")
        cluster.cancel_all()

        with pytest.raises(Exception) as e:
            print(pid_fn.get(pid_run1.name))
            assert isinstance(
                e, (ray.exceptions.TaskCancelledError, ray.exceptions.RayTaskError)
            )

        with pytest.raises(Exception) as e:
            print(pid_fn.get(pid_ref2))
            assert isinstance(
                e, (ray.exceptions.TaskCancelledError, ray.exceptions.RayTaskError)
            )

    @pytest.mark.skip("Does not work properly following Module refactor.")
    @pytest.mark.level("local")
    def test_function_queueing(self, cluster):
        pid_fn = rh.function(getpid).to(cluster)

        pids = [pid_fn.enqueue(resources={"num_cpus": 1}) for _ in range(10)]
        assert len(pids) == 10

    # @pytest.mark.skip("Not working properly.")
    # originally used ondemand_cpu_cluster, therefore marked as minimal
    @pytest.mark.level("minimal")
    def test_function_external_fn(self, cluster):
        """Test functioning a module from reqs, not from working_dir"""
        import numpy as np

        re_fn = rh.function(np.sum).to(cluster, env=["numpy"])
        res = re_fn(np.arange(5))
        assert int(res) == 10

    @pytest.mark.skip("Runs indefinitely.")
    # originally used ondemand_cpu_cluster, therefore marked as minimal
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

    @pytest.mark.rnstest
    @pytest.mark.level("local")
    def test_share_and_revoke_function(self, cluster):
        # TODO: refactor in order to test the function.share() method.
        my_function = rh.function(fn=summer).to(cluster).save(REMOTE_FUNC_NAME)

        my_function.share(
            users=["info@run.house"],
            access_level="read",
            notify_users=False,
        )
        with test_account():
            my_function = rh.function(name=my_function.rns_address)
            res = my_function(1, 2)
            assert res == 3

        my_function.revoke(users=["info@run.house"])

        with pytest.raises(Exception):
            with test_account():
                my_function = rh.function(name=my_function.rns_address)
                my_function(1, 2)

    @pytest.mark.clustertest
    @pytest.mark.rnstest
    @pytest.mark.level("thorough")
    def test_load_function_in_new_cluster(
        self, ondemand_cpu_cluster, static_cpu_cluster
    ):
        ondemand_cpu_cluster.save(
            f"@/{ondemand_cpu_cluster.name}"
        )  # Needs to be saved to rns, right now has a local name by default
        remote_sum = rh.function(summer).to(ondemand_cpu_cluster).save(REMOTE_FUNC_NAME)

        static_cpu_cluster.sync_secrets(["sky"])
        remote_python = (
            "import runhouse as rh; "
            f"remote_sum = rh.function(name='{REMOTE_FUNC_NAME}'); "
            "res = remote_sum(1, 5); "
            "assert res == 6"
        )
        res = static_cpu_cluster.run_python([remote_python], stream_logs=True)
        assert res[0][0] == 0

        remote_sum.delete_configs()

    @pytest.mark.clustertest
    @pytest.mark.level("thorough")
    def test_nested_diff_clusters(self, ondemand_cpu_cluster, static_cpu_cluster):
        summer_cpu = rh.function(summer).to(ondemand_cpu_cluster)
        call_function_diff_cpu = rh.function(call_function).to(static_cpu_cluster)

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

    @pytest.mark.clustertest
    @pytest.mark.level("local")
    def test_http_url(self, cluster):
        # TODO convert into something like function.request_args() and/or function.curl_command()
        remote_sum = rh.function(summer).to(cluster).save("@/remote_function")
        ssh_creds = cluster.ssh_creds()
        addr = (
            "http://" + cluster.LOCALHOST
            if cluster.server_connection_type in ["ssh", "aws_ssm"]
            else "https://" + cluster.address
            if cluster.server_connection_type == "tls"
            else "http://" + cluster.address
        )
        auth = (
            (ssh_creds.get("ssh_user"), ssh_creds.get("password"))
            if ssh_creds.get("password")
            else None
        )
        sum1 = requests.post(
            url=f"{addr}:{cluster.client_port}/call/{remote_sum.name}/call",
            json={"args": [1, 2]},
            headers=rh.configs.request_headers if cluster.den_auth else None,
            auth=auth,
            verify=False,
        ).json()
        assert int(sum1) == 3
        sum2 = requests.post(
            url=f"{addr}:{cluster.client_port}/call/{remote_sum.name}/call",
            json={"kwargs": {"a": 1, "b": 2}},
            headers=rh.configs.request_headers if cluster.den_auth else None,
            auth=auth,
            verify=False,
        ).json()
        assert int(sum2) == 3

    @pytest.mark.skip("Not yet implemented.")
    @pytest.mark.level("local")
    def test_http_url_with_curl(self):
        # TODO: refactor needed, once the Function.http_url() is implemented.
        # NOTE: Assumes the Function has already been created and deployed to running cluster
        s = rh.function(name="test_function")
        curl_cmd = s.http_url(a=1, b=2, curl_command=True)
        print(curl_cmd)

        # delete_configs the function data from the RNS
        self.delete_function_from_rns(s)

        assert True

    # test that deprecated arguments are still backwards compatible for now
    @pytest.mark.level("local")
    def test_reqs_backwards_compatible(self, cluster):
        # Check that warning is thrown
        with pytest.warns(UserWarning):
            remote_summer = rh.function(fn=np_summer).to(system=cluster, reqs=["numpy"])
        res = remote_summer(1, 5)
        assert res == 6

    @pytest.mark.level("local")
    def test_from_config(self, cluster):
        summer_pointers = rh.Function._extract_pointers(summer, reqs=[])
        summer_function_config = {
            "fn_pointers": summer_pointers,
            "system": None,
            "env": None,
        }
        my_summer_func = rh.Function.from_config(config=summer_function_config)
        my_summer_func = my_summer_func.to(system=cluster)
        assert my_summer_func(4, 6) == 10

        np_summer_pointers = rh.Function._extract_pointers(
            np_summer, reqs=["numpy", "./"]
        )
        np_summer_config = {
            "fn_pointers": np_summer_pointers,
            "system": None,
            "env": {"reqs": ["numpy"], "working_dir": "./"},
        }
        my_np_func = rh.Function.from_config(np_summer_config).to(system=cluster)
        assert my_np_func(1, 3) == 4

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

    @pytest.mark.level("local")
    def test_get_or_call(self, cluster):
        remote_summer = rh.function(summer).to(system=cluster)
        # first run - need to implement later test for return value verification.
        res = remote_summer.get_or_call(f"my_run_{time.time_ns()}", False, True, 12, 5)
        assert res == 17

        # not first run - implement later (raises an exception).
        remote_run = remote_summer.run(2, 3)
        get_or_call_res = remote_summer.get_or_call(
            run_name=remote_run
        ).resolved_state()
        # TODO: should not use .resolved_state(), need to be fixed.
        assert get_or_call_res == 5

    # TODO: should consider testing Function.send_secrets after secrets refactor.
    # TODO: should consider testing Function.keep_warm.

    # ---------- Unittests ----------
    @pytest.fixture(autouse=True)
    def init_fixtures(self):
        self.function = rh.function(summer)

    @pytest.mark.level("unit")
    @patch("runhouse.Function.get")
    def test_get_unittest(self, mock_get):
        mock_get.return_value = 5
        response = self.function.get(run_key="my_mocked_run")
        mock_get.assert_called_once_with(run_key="my_mocked_run")
        assert response == 5

    @pytest.mark.level("unit")
    @patch("runhouse.Function.http_url")
    # TODO: change the test once the implementation is ready
    def test_http_url_unittest(self, mock_get):
        mock_get.content("http_url not yet implemented for Function")
        mock_get.side_effect = NotImplementedError
        with pytest.raises(NotImplementedError):
            self.function.http_url()

    @pytest.mark.level("unit")
    @patch("runhouse.Function.keep_warm")
    @patch("runhouse.cluster")
    def test_keep_warm_byo_unittest(self, mock_cluster, mock_get):
        mock_get.return_value = self.function

        # BYO cluster
        regular_cluster = mock_cluster(name="regular_cluster")
        regular_cluster.autostop_mins = 1
        self.function.system = regular_cluster
        response_regular = self.function.keep_warm(autostop_mins=1)
        mock_get.assert_called_once_with(autostop_mins=1)
        assert (
            response_regular.system.autostop_mins == self.function.system.autostop_mins
        )
        assert response_regular.system.autostop_mins == 1

        self.function.system = None

    @pytest.mark.level("unit")
    @patch("runhouse.Function.keep_warm")
    @patch("runhouse.cluster")
    def test_keep_warm_on_demand_unittest(self, mock_cluster, mock_get):
        mock_get.return_value = self.function

        # on demand cluster
        on_demand_cluster = mock_cluster(name="on_demand_cluster", instance_type="aws")
        on_demand_cluster.autostop_mins = 2
        self.function.system = on_demand_cluster
        response_on_demand = self.function.keep_warm(autostop_mins=2)
        mock_get.assert_called_with(autostop_mins=2)
        assert (
            response_on_demand.system.autostop_mins
            == self.function.system.autostop_mins
        )
        assert response_on_demand.system.autostop_mins == 2

        self.function.system = None

    @pytest.mark.level("unit")
    @patch("runhouse.Function.keep_warm")
    @patch("runhouse.sagemaker_cluster")
    def test_keep_warm_unittest(self, mock_cluster, mock_get):
        mock_get.return_value = self.function

        # Sagemaker cluster
        sagemaker_cluster = mock_cluster(name="Sagemaker_cluster")
        sagemaker_cluster.autostop_mins = 3
        self.function.system = sagemaker_cluster
        response_sagemaker = self.function.keep_warm(autostop_mins=3)
        mock_get.assert_called_with(autostop_mins=3)
        assert (
            response_sagemaker.system.autostop_mins
            == self.function.system.autostop_mins
        )
        assert response_sagemaker.system.autostop_mins == 3

        self.function.system = None

    @pytest.mark.level("unit")
    @patch("runhouse.Function.notebook")
    def test_notebook_unittest(self, mock_get):
        mock_get.return_value = None
        response = self.function.notebook()
        mock_get.assert_called_once_with()
        assert response is None

    @pytest.mark.level("unit")
    @patch("runhouse.Function.remote")
    def test_remote_unittest(self, mock_get):
        mock_get.return_value = 5
        response = self.function.remote(3, 2)
        mock_get.assert_called_once_with(3, 2)
        assert response == 5

    @pytest.mark.level("unit")
    @patch("runhouse.Function.to")
    @patch("runhouse.cluster")
    def test_to_unittest(self, mock_cluster, mock_get):
        local_cluster = mock_cluster(name="my_local_cluster")
        func = rh.function(summer).to(local_cluster)
        mock_get.return_value = func
        response = self.function.to(local_cluster)
        mock_get.assert_called_with(local_cluster)
        assert response is func

    @pytest.mark.level("unit")
    @patch("runhouse.Function.map")
    def test_map_unittest(self, mock_get):
        # TODO: change the test once the implementation change is ready, if necessary
        mock_get.return_value = [3, 7, 11]
        response = self.function.map([[1, 3, 5], [2, 4, 6]])
        mock_get.assert_called_once_with([[1, 3, 5], [2, 4, 6]])
        assert response == [3, 7, 11]

    @pytest.mark.level("unit")
    @patch("runhouse.Function.starmap")
    def test_starmap_unittest(self, mock_get):
        # TODO: change the test once the implementation change is ready, if necessary
        mock_get.return_value = [3, 7, 11]
        response = self.function.starmap([(1, 2), (3, 4), (5, 6)])
        mock_get.assert_called_once_with([(1, 2), (3, 4), (5, 6)])
        assert response == [3, 7, 11]

    @pytest.mark.level("unit")
    @patch("runhouse.Function.get_or_call")
    def test_get_or_call_unittest(self, mock_get):
        mock_get.return_value = 17
        response_first = self.function.get_or_call(
            "my_run_first_time", False, True, 12, 5
        )
        mock_get.assert_called_with("my_run_first_time", False, True, 12, 5)
        assert response_first == 17
        second_response = self.function.get_or_call("my_run_first_time")
        assert second_response == 17

    @pytest.mark.level("unit")
    @pytest.mark.skip("Maybe send secrets is not relevant")
    @patch("runhouse.Function.send_secrets")
    def test_send_secrets_unittest(self, mock_get):
        # TODO: need to think if send_secrets is a relevant Function method
        pass


if __name__ == "__main__":
    unittest.main()
