import json
import os
import unittest
from multiprocessing import Pool

import requests

import runhouse as rh
from runhouse.rns.api_utils.resource_access import ResourceAccess


def setup():
    rh.set_folder("~/tests", create=True)


def torch_summer(a, b):
    # import inside so tests that don't use torch don't fail because torch isn't in their reqs
    import torch

    return int(torch.Tensor([a, b]).sum())


def summer(a, b):
    return a + b


def test_create_send_from_name_local():
    local_sum = rh.send(
        fn=summer, name="local_send", hardware="^rh-cpu", reqs=["local:./"]
    ).save()
    del local_sum

    remote_sum = rh.send(name="local_send")
    res = remote_sum(1, 5)
    assert res == 6

    remote_sum.delete_configs()
    assert rh.exists("local_send") is False


def test_create_send_from_rns():
    remote_sum = rh.send(
        fn=summer, name="@/remote_send", hardware="^rh-cpu", reqs=[], dryrun=True
    ).save()
    del remote_sum

    remote_sum = rh.send(name="@/remote_send")
    res = remote_sum(1, 5)
    assert res == 6

    remote_sum.delete_configs()
    assert rh.exists("@/remote_send") is False


@unittest.skip("Not yet implemented.")
def test_running_send_as_proxy():
    remote_sum = rh.send(
        fn=summer, name="@/remote_send", hardware="^rh-cpu", reqs=[], dryrun=False
    ).save()
    del remote_sum

    remote_sum = rh.send(name="@/remote_send")
    remote_sum.access = ResourceAccess.proxy
    res = remote_sum(1, 5)
    assert res == 6

    remote_sum.delete_configs()
    assert rh.exists("@remote_send") is False


def test_get_send_history():
    remote_sum = rh.send(
        fn=summer, name="@/remote_send", hardware="^rh-cpu", reqs=[], dryrun=True
    ).save()
    remote_sum = rh.send(
        fn=summer, name="@/remote_send", hardware="^rh-cpu", reqs=["torch"], dryrun=True
    ).save()
    remote_sum = rh.send(
        fn=summer, name="@/remote_send", hardware="^rh-cpu", reqs=[], dryrun=True
    ).save()
    name = "@/remote_send"
    remote_sum = rh.send(name=name)
    history = remote_sum.history(name=name)
    assert len(history) >= 3
    assert "torch" in history[1]["data"]["reqs"]
    remote_sum.delete_configs()
    # TODO assert raises
    # history = remote_sum.history(name=name)
    # assert len(history) == 0


def multiproc_torch_sum(inputs):
    print(f"CPUs: {os.cpu_count()}")
    with Pool() as P:
        return P.starmap(torch_summer, inputs)


def test_remote_send_with_multiprocessing():
    re_fn = rh.send(
        multiproc_torch_sum,
        name="test_send",
        hardware="^rh-cpu",
        reqs=["./", "torch==1.12.1"],
    )
    summands = list(zip(range(5), range(4, 9)))
    res = re_fn(summands)
    assert res == [4, 6, 8, 10, 12]


def getpid(a=0):
    return os.getpid() + a


def test_maps():
    pid_fn = rh.send(getpid, hardware="^rh-cpu")
    num_pids = [1] * 50
    pids = pid_fn.map(num_pids)
    assert len(set(pids)) > 1

    pid_ref = pid_fn.remote()

    pids = pid_fn.repeat(num_repeats=50)
    assert len(set(pids)) > 1

    pids = [pid_fn.enqueue() for _ in range(10)]
    assert len(pids) == 10

    pid_res = pid_fn.get(pid_ref)
    assert pid_res > 0

    # Test passing an objectref into a normal call
    pid_res_from_ref = pid_fn(pid_ref)
    assert pid_res_from_ref > pid_res

    re_fn = rh.send(summer, hardware="^rh-cpu")
    summands = list(zip(range(5), range(4, 9)))
    res = re_fn.starmap(summands)
    assert res == [4, 6, 8, 10, 12]


def test_send_git_fn():
    remote_parse = rh.send(
        fn="https://github.com/huggingface/diffusers/blob/"
        "main/examples/dreambooth/train_dreambooth.py:parse_args",
        hardware="^rh-cpu",
        reqs=[
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


@unittest.skip("Not working properly.")
def test_send_external_fn():
    """Test sending a module from reqs, not from working_dir"""
    import torch

    re_fn = rh.send(torch.sum, hardware="^rh-cpu", reqs=["torch"])
    res = re_fn(torch.arange(5))
    assert int(res) == 10


@unittest.skip("Runs indefinitely.")
def test_notebook():
    nb_sum = lambda x: multiproc_torch_sum(x)
    re_fn = rh.send(
        nb_sum, hardware="^rh-cpu", reqs=["./", "torch==1.12.1"], dryrun=True
    )
    re_fn.notebook()
    summands = list(zip(range(5), range(4, 9)))
    res = re_fn(summands)
    assert res == [4, 6, 8, 10, 12]
    re_fn.delete_configs()


@unittest.skip("Runs indefinitely.")
def test_ssh():
    # TODO do this properly
    my_send = rh.send(name="local_send")
    my_send.ssh()
    assert True


def test_providing_access_to_send():
    my_send = rh.send(
        fn=summer, name="@/remote_send", hardware="^rh-cpu", dryrun=True
    ).save()
    added_users, new_users = my_send.share(
        users=["dongreenberg", "jlewit1"], access_type=ResourceAccess.read
    )
    assert added_users or new_users
    my_send.delete_configs()


def delete_send_from_rns(s):
    server_url = s.rns_client.api_server_url
    resource_request_uri = s.rns_client.resource_uri(s.name)
    resp = requests.delete(
        f"{server_url}/resource/{resource_request_uri}",
        headers=s.rns_client.request_headers,
    )
    if resp.status_code != 200:
        raise Exception(
            f"Failed to delete_configs send data from url: {json.loads(resp.content)}"
        )

    try:
        # Terminate the cluster
        s.cluster.teardown_and_delete()
    except Exception as e:
        raise Exception(f"Failed to teardown the cluster: {e}")


@unittest.skip("Not yet implemented.")
def test_http_url():
    # TODO [DG] shouldn't have to specify fn here as a callable / at all?
    s = rh.send(fn=summer, name="test_send", hardware="^rh-cpu")

    # Generate and call the URL
    http_url = s.http_url()
    assert http_url

    res = s(a=1, b=2)

    # delete_configs the Send data from the RNS
    # delete_send_from_rns(s)

    assert res == 3


@unittest.skip("Not yet implemented.")
def test_http_url_with_curl():
    # NOTE: Assumes the Send has already been created and deployed to running cluster
    s = rh.send(name="test_send")
    curl_cmd = s.http_url(a=1, b=2, curl_command=True)
    print(curl_cmd)

    # delete_configs the send data from the RNS
    delete_send_from_rns(s)

    assert True


def test_byo_cluster_send():
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
    re_fn = rh.send(
        multiproc_torch_sum, hardware=byo_cluster, reqs=["./", "torch==1.12.1"]
    )
    summands = list(zip(range(5), range(4, 9)))
    res = re_fn(summands)
    assert res == [4, 6, 8, 10, 12]


def test_byo_cluster_maps():
    pid_fn = rh.send(getpid, hardware="different-cluster")
    num_pids = [1] * 50
    pids = pid_fn.map(num_pids)
    assert len(set(pids)) > 1

    pid_ref = pid_fn.remote()

    pids = pid_fn.repeat(num_repeats=50)
    assert len(set(pids)) > 1

    pids = [pid_fn.enqueue() for _ in range(10)]
    assert len(pids) == 10

    pid_res = pid_fn.get(pid_ref)
    assert pid_res > 0

    # Test passing an objectref into a normal call
    pid_res_from_ref = pid_fn(pid_ref)
    assert pid_res_from_ref > pid_res

    re_fn = rh.send(summer, hardware="different-cluster")
    summands = list(zip(range(5), range(4, 9)))
    res = re_fn.starmap(summands)
    assert res == [4, 6, 8, 10, 12]


if __name__ == "__main__":
    setup()
    unittest.main()
