import os
import unittest
from multiprocessing import Pool

import pytest
import requests
import runhouse as rh
from runhouse.rns.api_utils.resource_access import ResourceAccess
from runhouse.rns.api_utils.utils import load_resp_content


def setup():
    rh.set_folder("~/tests", create=True)


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
def test_create_function_from_name_local(cpu_cluster):
    local_sum = rh.function(fn=summer, name="local_function", system=cpu_cluster).save()
    del local_sum

    remote_sum = rh.function(name="local_function")
    res = remote_sum(1, 5)
    assert res == 6

    remote_sum.delete_configs()
    assert rh.exists("local_function") is False


@pytest.mark.clustertest
@pytest.mark.rnstest
def test_create_function_from_rns(cpu_cluster):
    remote_sum = rh.function(
        fn=summer,
        name="@/remote_function",
        system=cpu_cluster,
        dryrun=True,
    ).save()
    del remote_sum

    remote_sum = rh.function(name="@/remote_function")
    res = remote_sum(1, 5)
    assert res == 6

    remote_sum.delete_configs()
    assert rh.exists("@/remote_function") is False


@unittest.skip("Not yet implemented.")
@pytest.mark.rnstest
@pytest.mark.clustertest
def test_running_function_as_proxy(cpu_cluster):
    remote_sum = rh.function(
        fn=summer, name="@/remote_function", system=cpu_cluster, env=[]
    ).save()
    del remote_sum

    remote_sum = rh.function(name="@/remote_function")
    remote_sum.access = ResourceAccess.PROXY
    res = remote_sum(1, 5)
    assert res == 6

    remote_sum.delete_configs()
    assert rh.exists("@remote_function") is False


@pytest.mark.clustertest
@pytest.mark.rnstest
def test_get_function_history(cpu_cluster):
    remote_sum = rh.function(
        fn=summer, name="@/remote_function", system=cpu_cluster, env=[], dryrun=True
    ).save()
    remote_sum = rh.function(
        fn=summer,
        name="@/remote_function",
        system=cpu_cluster,
        env=["torch"],
        dryrun=True,
    ).save()
    remote_sum = rh.function(
        fn=summer, name="@/remote_function", system=cpu_cluster, env=[], dryrun=True
    ).save()
    name = "@/remote_function"
    remote_sum = rh.function(name=name)
    history = remote_sum.history(name=name)
    assert len(history) >= 3
    remote_sum.delete_configs()
    # TODO assert raises
    # history = remote_sum.history(name=name)
    # assert len(history) == 0


def multiproc_torch_sum(inputs):
    print(f"CPUs: {os.cpu_count()}")
    with Pool() as P:
        return P.starmap(torch_summer, inputs)


@pytest.mark.clustertest
@pytest.mark.rnstest
def test_remote_function_with_multiprocessing(cpu_cluster):
    re_fn = rh.function(
        multiproc_torch_sum,
        name="test_function",
        system=cpu_cluster,
        env=["./", "torch==1.12.1"],
    )
    summands = list(zip(range(5), range(4, 9)))
    res = re_fn(summands)
    assert res == [4, 6, 8, 10, 12]


def getpid(a=0):
    return os.getpid() + a


@pytest.mark.clustertest
def test_maps(cpu_cluster):
    pid_fn = rh.function(getpid, system=cpu_cluster)
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

    re_fn = rh.function(summer, system=cpu_cluster)
    summands = list(zip(range(5), range(4, 9)))
    res = re_fn.starmap(summands)
    assert res == [4, 6, 8, 10, 12]


@pytest.mark.clustertest
def test_remotes(cpu_cluster):
    pid_fn = rh.function(getpid, system=cpu_cluster)

    pid_ref = pid_fn.remote()
    pid_res = pid_fn.get(pid_ref)
    assert pid_res > 0

    # Test passing an objectref into a normal call
    pid_res_from_ref = pid_fn(pid_ref)
    assert pid_res_from_ref > pid_res


@pytest.mark.clustertest
def test_function_git_fn(cpu_cluster):
    remote_parse = rh.function(
        fn="https://github.com/huggingface/diffusers/blob/"
        "main/examples/dreambooth/train_dreambooth.py:parse_args",
        system=cpu_cluster,
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
def test_list_keys(cpu_cluster):
    pid_fn = rh.function(getpid, system=cpu_cluster)

    pid_ref1 = pid_fn.remote()
    pid_ref2 = pid_fn.remote()

    current_jobs = cpu_cluster.list_keys()
    assert set([pid_ref1, pid_ref2]).issubset(current_jobs)


@pytest.mark.clustertest
def test_cancel_jobs(cpu_cluster):
    pid_fn = rh.function(getpid, system=cpu_cluster)

    pid_ref1 = pid_fn.remote()
    pid_ref2 = pid_fn.remote()

    cpu_cluster.cancel(all=True)

    current_jobs = cpu_cluster.list_keys()
    assert not set([pid_ref1, pid_ref2]).issubset(current_jobs)


@pytest.mark.clustertest
def test_function_queueing(cpu_cluster):
    pid_fn = rh.function(getpid, system=cpu_cluster)

    pids = [pid_fn.enqueue(resources={"num_cpus": 1}) for _ in range(10)]
    assert len(pids) == 10


@pytest.mark.clustertest
def test_function_to_env(cpu_cluster):
    cpu_cluster.run(["pip uninstall numpy -y"])

    np_func = rh.function(np_array, system=cpu_cluster)
    np_func = np_func.to(env=["numpy"])

    list = [1, 2, 3]
    res = np_func(list)
    assert res.tolist() == list


@unittest.skip("Not working properly.")
@pytest.mark.clustertest
def test_function_external_fn(cpu_cluster):
    """Test functioning a module from reqs, not from working_dir"""
    import torch

    re_fn = rh.function(torch.sum, system=cpu_cluster, env=["torch"])
    res = re_fn(torch.arange(5))
    assert int(res) == 10


@unittest.skip("Runs indefinitely.")
@pytest.mark.clustertest
def test_notebook(cpu_cluster):
    nb_sum = lambda x: multiproc_torch_sum(x)
    re_fn = rh.function(
        nb_sum, system=cpu_cluster, env=["./", "torch==1.12.1"], dryrun=True
    )
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
def test_share_function(cpu_cluster):
    my_function = rh.function(
        fn=summer, name="@/remote_function", system=cpu_cluster
    ).save()

    my_function.share(
        users=["donny@run.house", "josh@run.house"],
        access_type="read",
        notify_users=False,
    )
    assert True


@pytest.mark.rnstest
def test_load_shared_function():
    my_function = rh.function(name="@/remote_function")
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
def test_http_url(cpu_cluster):
    remote_sum = rh.function(summer).to(cpu_cluster).save("@/remote_function")
    cpu_cluster.ssh_tunnel(80, 50052)
    sum1 = requests.post(f"http://127.0.0.1:80/call/remote_function/",
                         json={"args": [1, 2]}).json()
    assert sum1 == 3
    sum2 = requests.post(f"http://127.0.0.1:80/call/remote_function/",
                         json={"kwargs": {"a": 1, "b": 2}}).json()
    assert sum2 == 3


@unittest.skip("Not yet implemented.")
def test_http_url_with_curl():
    # NOTE: Assumes the Function has already been created and deployed to running cluster
    s = rh.function(name="test_function")
    curl_cmd = s.http_url(a=1, b=2, curl_command=True)
    print(curl_cmd)

    # delete_configs the function data from the RNS
    delete_function_from_rns(s)

    assert True


@pytest.mark.clustertest
@pytest.mark.rnstest
def test_byo_cluster_function():
    # Spin up a new basic m5.xlarge EC2 instance
    c = rh.cluster(
        instance_type="m5.xlarge",
        provider="aws",
        region="us-east-1",
        image_id="ami-0a313d6098716f372",
        name="test-byo-cluster",
    ).up_if_not()
    ip = c.address
    creds = c.ssh_creds()
    del c
    byo_cluster = rh.cluster(name="different-cluster", ips=[ip], ssh_creds=creds).save()
    re_fn = rh.function(
        multiproc_torch_sum, system=byo_cluster, env=["./", "pytest", "torch==1.12.1"]
    )
    summands = list(zip(range(5), range(4, 9)))
    res = re_fn(summands)
    assert res == [4, 6, 8, 10, 12]


@pytest.mark.clustertest
def test_byo_cluster_maps():
    pid_fn = rh.function(getpid, system="different-cluster")
    num_pids = [1] * 20
    pids = pid_fn.map(num_pids)
    assert len(set(pids)) > 1
    assert all(pid > 0 for pid in pids)

    pid_ref = pid_fn.remote()

    pids = pid_fn.repeat(num_repeats=20)
    assert len(set(pids)) > 1
    assert all(pid > 0 for pid in pids)

    pids = [pid_fn.enqueue() for _ in range(10)]
    assert len(pids) == 10

    pid_res = pid_fn.get(pid_ref)
    assert pid_res > 0

    # Test passing an objectref into a normal call
    pid_res_from_ref = pid_fn(pid_ref)
    assert pid_res_from_ref > pid_res

    re_fn = rh.function(summer, system="different-cluster")
    summands = list(zip(range(5), range(4, 9)))
    res = re_fn.starmap(summands)
    assert res == [4, 6, 8, 10, 12]


@pytest.mark.clustertest
@pytest.mark.rnstest
def test_load_function_in_new_env(cpu_cluster):
    cpu_cluster.save(name="@/rh-cpu")
    remote_sum = rh.function(
        fn=summer, name="@/remote_function", system=cpu_cluster, env=[], dryrun=True
    ).save()

    byo_cluster = rh.cluster(name="different-cluster")
    byo_cluster.send_secrets(["ssh"])
    remote_python = (
        "import runhouse as rh; "
        "remote_sum = rh.function(name='remote_function'); "
        "res = remote_sum(1, 5); "
        "assert res == 6"
    )
    res = byo_cluster.run_python([remote_python], stream_logs=True)
    assert res[0][0] == 0

    remote_sum.delete_configs()


@pytest.mark.clustertest
def test_nested_diff_clusters(cpu_cluster):
    summer_cpu = rh.function(fn=summer, system=cpu_cluster)
    call_function_diff_cpu = rh.function(fn=call_function, system="different-cluster")

    kwargs = {"a": 1, "b": 5}
    res = call_function_diff_cpu(summer_cpu, **kwargs)
    assert res == 6


@pytest.mark.clustertest
def test_nested_same_cluster(cpu_cluster):
    summer_cpu = rh.function(fn=summer, system=cpu_cluster)
    call_function_cpu = rh.function(fn=call_function, system=cpu_cluster)

    kwargs = {"a": 1, "b": 5}
    res = call_function_cpu(summer_cpu, **kwargs)
    assert res == 6


# test that deprecated arguments are still backwards compatible for now
@pytest.mark.clustertest
def test_reqs_backwards_compatible(cpu_cluster):
    summer_cpu = rh.function(fn=summer, system=cpu_cluster, reqs=[])
    res = summer_cpu(1, 5)
    assert res == 6

    torch_summer_cpu = rh.function(fn=summer, system=cpu_cluster, reqs=["torch"])
    torch_res = torch_summer_cpu(1, 5)
    assert torch_res == 6


@pytest.mark.clustertest
def test_setup_cmds_backwards_compatible(cpu_cluster):
    torch_summer_cpu = rh.function(fn=summer, system=cpu_cluster, reqs=["torch"])
    torch_res = torch_summer_cpu(1, 5)
    assert torch_res == 6


if __name__ == "__main__":
    setup()
    unittest.main()
