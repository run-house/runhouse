import logging
import time
import unittest

import pytest

import runhouse as rh

from tests.test_function import multiproc_torch_sum

TEMP_FILE = "my_file.txt"
TEMP_FOLDER = "~/runhouse-tests"

logger = logging.getLogger(__name__)


def do_printing_and_logging():
    for i in range(6):
        # Wait to make sure we're actually streaming
        time.sleep(1)
        print(f"Hello from the cluster! {i}")
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
def test_get_from_cluster(cpu_cluster):
    print_fn = rh.function(fn=do_printing_and_logging, system=cpu_cluster)
    run_obj = print_fn.run()
    assert isinstance(run_obj, rh.Run)

    res = cpu_cluster.get(run_obj.name, stream_logs=True)
    assert res == list(range(50))


@pytest.mark.clustertest
def test_put_and_get_on_cluster(cpu_cluster):
    test_list = list(range(5, 50, 2)) + ["a string"]
    cpu_cluster.put("my_list", test_list)
    ret = cpu_cluster.get("my_list")
    assert all(a == b for (a, b) in zip(ret, test_list))


@pytest.mark.clustertest
def test_stream_logs(cpu_cluster):
    print_fn = rh.function(fn=do_printing_and_logging, system=cpu_cluster)
    res = print_fn(stream_logs=True)
    # TODO [DG] assert that the logs are streamed
    assert res == list(range(50))


@pytest.mark.clustertest
def test_multiprocessing_streaming(cpu_cluster):
    re_fn = rh.function(
        multiproc_torch_sum, system=cpu_cluster, env=["./", "torch==1.12.1"]
    )
    summands = list(zip(range(5), range(4, 9)))
    res = re_fn(summands, stream_logs=True)
    assert res == [4, 6, 8, 10, 12]


@pytest.mark.clustertest
def test_tqdm_streaming(cpu_cluster):
    # Note, this doesn't work properly in PyCharm due to incomplete
    # support for carriage returns in the PyCharm console.
    print_fn = rh.function(fn=do_tqdm_printing_and_logging, system=cpu_cluster)
    res = print_fn(steps=40, stream_logs=True)
    assert res == list(range(50))


@pytest.mark.clustertest
def test_cancel_run(cpu_cluster):
    print_fn = rh.function(fn=do_printing_and_logging, system=cpu_cluster)
    run_obj = print_fn.run()
    assert isinstance(run_obj, rh.Run)

    key = run_obj.name
    cpu_cluster.cancel(key)
    with pytest.raises(Exception) as e:
        cpu_cluster.get(key, stream_logs=True)
    # NOTE [DG]: For some reason the exception randomly returns in different formats
    assert "ray.exceptions.TaskCancelledError" in str(
        e.value
    ) or "This task or its dependency was cancelled by" in str(e.value)


if __name__ == "__main__":
    unittest.main()
