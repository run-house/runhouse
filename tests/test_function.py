import logging
import os
import time
import unittest
from multiprocessing import Pool

import pytest
import ray.exceptions
import requests
import runhouse as rh
from runhouse.rns.utils.api import load_resp_content

from .conftest import cpu_clusters

REMOTE_FUNC_NAME = "@/remote_function"

logger = logging.getLogger(__name__)


def call_function(fn, **kwargs):
    return fn(**kwargs)


def torch_summer(a, b):
    # import inside so tests that don't use torch don't fail because torch isn't in their reqs
    import torch

    return int(torch.Tensor([a, b]).sum())


def summer(a, b):
    return a + b


def np_array(list):
    import numpy as np

    return np.array(list)


@pytest.mark.clustertest
@pytest.mark.rnstest
@cpu_clusters
def test_create_function_from_name_local(cluster):
    local_name = "~/local_function"
    local_sum = rh.function(summer).to(cluster).save(local_name)
    del local_sum

    remote_sum = rh.function(name=local_name)
    res = remote_sum(1, 5)
    assert res == 6

    remote_sum.delete_configs()
    assert rh.exists(local_name) is False


@pytest.mark.clustertest
@pytest.mark.rnstest
@cpu_clusters
def test_create_function_from_rns(cluster):
    remote_sum = rh.function(summer).to(cluster).save(REMOTE_FUNC_NAME)
    del remote_sum

    # reload the function
    remote_sum = rh.function(name=REMOTE_FUNC_NAME)
    res = remote_sum(1, 5)
    assert res == 6

    remote_sum.delete_configs()
    assert not rh.exists(REMOTE_FUNC_NAME)


async def async_summer(a, b):
    return a + b


@pytest.mark.clustertest
@pytest.mark.asyncio
@cpu_clusters
async def test_async_function(cluster):
    remote_sum = rh.function(async_summer).to(cluster)
    res = await remote_sum(1, 5)
    assert res == 6


@pytest.mark.clustertest
@pytest.mark.rnstest
@cpu_clusters
def test_get_function_history(cluster):
    # reload the function from RNS
    remote_sum = rh.function(summer).to(cluster).save(REMOTE_FUNC_NAME)

    history = remote_sum.history()
    assert history


def multiproc_torch_sum(inputs):
    print(f"CPUs: {os.cpu_count()}")
    with Pool() as P:
        return P.starmap(torch_summer, inputs)


@pytest.mark.clustertest
@pytest.mark.rnstest
@cpu_clusters
def test_remote_function_with_multiprocessing(cluster):
    re_fn = rh.function(multiproc_torch_sum, name="test_function").to(
        cluster, env=["torch==1.12.1"]
    )

    summands = list(zip(range(5), range(4, 9)))
    res = re_fn(summands)

    assert res == [4, 6, 8, 10, 12]


def getpid(a=0):
    return os.getpid() + a


@unittest.skip("Does not work properly following Module refactor.")
@pytest.mark.clustertest
@cpu_clusters
def test_maps(cluster):
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


@pytest.mark.clustertest
@cpu_clusters
def test_generator(cluster):
    remote_slow_generator = rh.function(slow_generator).to(cluster)
    results = []
    for val in remote_slow_generator(5):
        assert val
        print(val)
        results += [val]
    assert len(results) == 5


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


@pytest.mark.clustertest
@pytest.mark.asyncio
@cpu_clusters
async def test_async_generator(cluster):
    remote_slow_generator = rh.function(async_slow_generator).to(cluster)
    results = []
    async for val in remote_slow_generator(5):
        assert val
        print(val)
        results += [val]
    assert len(results) == 5


@pytest.mark.clustertest
@cpu_clusters
def test_remotes(cluster):
    pid_fn = rh.function(getpid, system=cluster)

    pid_key = pid_fn.run()
    pid_res = cluster.get(pid_key).fetch()
    assert pid_res > 0

    # Test passing a remote into a normal call
    pid_blob = pid_fn.remote()
    pid_res = cluster.get(pid_blob.name).fetch()
    assert pid_res > 0
    pid_res = pid_blob.fetch()
    assert pid_res > 0


@pytest.mark.clustertest
@cpu_clusters
def test_function_git_fn(cluster):
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
    assert args.pretrained_model_name_or_path == "stabilityai/stable-diffusion-2-base"


@pytest.mark.clustertest
@cpu_clusters
def test_list_keys(cluster):
    pid_fn = rh.function(getpid).to(system=cluster)

    pid_obj1 = pid_fn.run()
    pid_obj2 = pid_fn.run()

    current_jobs = cluster.keys()
    assert set([pid_obj1, pid_obj2]).issubset(current_jobs)

    pid_obj3 = pid_fn.remote()
    pid_obj4 = pid_fn.remote()

    current_jobs = cluster.keys()
    assert set([pid_obj3.name, pid_obj4.name]).issubset(current_jobs)


def slow_getpid(a=0):
    time.sleep(10)
    return os.getpid() + a


@pytest.mark.clustertest
@cpu_clusters
def test_cancel_jobs(cluster):
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


@unittest.skip("Does not work properly following Module refactor.")
@pytest.mark.clustertest
@cpu_clusters
def test_function_queueing(cluster):
    pid_fn = rh.function(getpid).to(cluster)

    pids = [pid_fn.enqueue(resources={"num_cpus": 1}) for _ in range(10)]
    assert len(pids) == 10


@pytest.mark.clustertest
@cpu_clusters
def test_function_to_env(cluster):
    cluster.run(["pip uninstall numpy -y"])

    np_func = rh.function(np_array).to(cluster, env=["numpy"])

    list = [1, 2, 3]
    res = np_func(list)
    assert res.tolist() == list


@unittest.skip("Not working properly.")
@pytest.mark.clustertest
def test_function_external_fn(ondemand_cpu_cluster):
    """Test functioning a module from reqs, not from working_dir"""
    import torch

    re_fn = rh.function(torch.sum).to(ondemand_cpu_cluster, env=["torch"])
    res = re_fn(torch.arange(5))
    assert int(res) == 10


@unittest.skip("Runs indefinitely.")
@pytest.mark.clustertest
def test_notebook(ondemand_cpu_cluster):
    nb_sum = lambda x: multiproc_torch_sum(x)
    re_fn = rh.function(nb_sum).to(ondemand_cpu_cluster, env=["torch==1.12.1"])

    re_fn.notebook()
    summands = list(zip(range(5), range(4, 9)))
    res = re_fn(summands)

    assert res == [4, 6, 8, 10, 12]
    re_fn.delete_configs()


@unittest.skip("Runs indefinitely.")
def test_ssh():
    # TODO do this properly
    my_function = rh.function(name="local_function")
    my_function.ssh()
    assert True


@pytest.mark.clustertest
@pytest.mark.rnstest
def test_share_function(ondemand_cpu_cluster):
    my_function = rh.function(fn=summer).to(ondemand_cpu_cluster).save(REMOTE_FUNC_NAME)

    my_function.share(
        users=["donny@run.house", "josh@run.house"],
        access_type="read",
        notify_users=False,
    )
    assert True


@pytest.mark.rnstest
def test_load_shared_function():
    my_function = rh.function(name=REMOTE_FUNC_NAME)
    res = my_function(1, 2)
    assert res == 3


@pytest.mark.rnstest
def delete_function_from_rns(s):
    server_url = s.rns_client.api_server_url
    resource_request_uri = s.rns_client.resource_uri(s.name)
    resp = requests.delete(
        f"{server_url}/resource/{resource_request_uri}",
        headers=s.rns_client.request_headers,
    )
    if resp.status_code != 200:
        raise Exception(
            f"Failed to delete_configs function data from path: {load_resp_content(resp)}"
        )

    try:
        # Terminate the cluster
        s.cluster.teardown_and_delete()
    except Exception as e:
        raise Exception(f"Failed to teardown the cluster: {e}")


@pytest.mark.clustertest
@pytest.mark.rnstest
def test_load_function_in_new_env(ondemand_cpu_cluster, byo_cpu):
    ondemand_cpu_cluster.save(
        f"@/{ondemand_cpu_cluster.name}"
    )  # Needs to be saved to rns, right now has a local name by default
    remote_sum = rh.function(summer).to(ondemand_cpu_cluster).save(REMOTE_FUNC_NAME)

    remote_python = (
        "import runhouse as rh; "
        f"remote_sum = rh.function(name='{REMOTE_FUNC_NAME}'); "
        "res = remote_sum(1, 5); "
        "assert res == 6"
    )
    res = byo_cpu.run_python([remote_python], stream_logs=True)
    assert res[0][0] == 0

    remote_sum.delete_configs()


@pytest.mark.clustertest
def test_nested_diff_clusters(ondemand_cpu_cluster, byo_cpu):
    summer_cpu = rh.function(summer).to(ondemand_cpu_cluster)
    call_function_diff_cpu = rh.function(call_function).to(byo_cpu)

    kwargs = {"a": 1, "b": 5}
    res = call_function_diff_cpu(summer_cpu, **kwargs)
    assert res == 6


@pytest.mark.clustertest
@cpu_clusters
def test_nested_same_cluster(cluster):
    # When the system of a function is set to the cluster that the function is being called on, we run the function
    # locally and not via an RPC call
    summer_cpu = rh.function(fn=summer).to(system=cluster)
    call_function_cpu = rh.function(fn=call_function).to(system=cluster)

    kwargs = {"a": 1, "b": 5}
    res = call_function_cpu(summer_cpu, **kwargs)
    assert res == 6


@pytest.mark.clustertest
@cpu_clusters
def test_http_url(cluster):
    rh.function(summer).to(cluster).save("@/remote_function")
    tun, port = cluster.ssh_tunnel(80, 32300)
    ssh_creds = cluster.ssh_creds()
    auth = (
        (ssh_creds.get("ssh_user"), ssh_creds.get("password"))
        if ssh_creds.get("password")
        else None
    )
    sum1 = requests.post(
        "http://127.0.0.1:80/call/remote_function/call",
        json={"args": [1, 2]},
        auth=auth,
    ).json()
    assert int(sum1) == 3
    sum2 = requests.post(
        "http://127.0.0.1:80/call/remote_function/call",
        json={"kwargs": {"a": 1, "b": 2}},
        auth=auth,
    ).json()
    assert int(sum2) == 3

    tun.close()


@unittest.skip("Not yet implemented.")
def test_http_url_with_curl():
    # NOTE: Assumes the Function has already been created and deployed to running cluster
    s = rh.function(name="test_function")
    curl_cmd = s.http_url(a=1, b=2, curl_command=True)
    print(curl_cmd)

    # delete_configs the function data from the RNS
    delete_function_from_rns(s)

    assert True


# test that deprecated arguments are still backwards compatible for now
@pytest.mark.clustertest
@cpu_clusters
def test_reqs_backwards_compatible(cluster):
    summer_cpu = rh.function(fn=summer).to(system=cluster)
    res = summer_cpu(1, 5)
    assert res == 6

    torch_summer_cpu = rh.function(fn=summer).to(system=cluster, env=["torch"])
    torch_res = torch_summer_cpu(1, 5)
    assert torch_res == 6


@pytest.mark.clustertest
@cpu_clusters
def test_setup_cmds_backwards_compatible(cluster):
    torch_summer_cpu = rh.function(fn=summer).to(system=cluster, env=["torch"])
    torch_res = torch_summer_cpu(1, 5)
    assert torch_res == 6


if __name__ == "__main__":
    unittest.main()
