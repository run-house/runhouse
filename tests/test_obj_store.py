import logging
import time
import unittest

import pytest

import runhouse as rh
from tqdm.auto import tqdm  # progress bar

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
    progress_bar = tqdm(range(steps))
    for i in range(steps):
        # Wait to make sure we're actually streaming
        time.sleep(0.1)
        progress_bar.update(1)
    return list(range(50))


@pytest.mark.clustertest
def test_get_from_cluster(cpu):
    print_fn = rh.function(fn=do_printing_and_logging, system=cpu)
    key = print_fn.remote()
    assert isinstance(key, str)
    res = cpu.get(key, stream_logs=True)
    assert res == list(range(50))


@pytest.mark.clustertest
def test_put_and_get_on_cluster(cpu):
    test_list = list(range(5, 50, 2)) + ["a string"]
    cpu.put("my_list", test_list)
    ret = cpu.get("my_list")
    assert all(a == b for (a, b) in zip(ret, test_list))


@pytest.mark.clustertest
def test_stream_logs(cpu):
    print_fn = rh.function(fn=do_printing_and_logging, system=cpu)
    res = print_fn(stream_logs=True)
    # TODO [DG] assert that the logs are streamed
    assert res == list(range(50))


@pytest.mark.clustertest
def test_multiprocessing_streaming(cpu):
    re_fn = rh.function(multiproc_torch_sum, system=cpu, reqs=["./", "torch==1.12.1"])
    summands = list(zip(range(5), range(4, 9)))
    res = re_fn(summands, stream_logs=True)
    assert res == [4, 6, 8, 10, 12]


@pytest.mark.clustertest
def test_tqdm_streaming(cpu):
    # Note, this doesn't work properly in PyCharm due to incomplete
    # support for carriage returns in the PyCharm console.
    print_fn = rh.function(fn=do_tqdm_printing_and_logging, system=cpu)
    res = print_fn(steps=40, stream_logs=True)
    assert res == list(range(50))


@pytest.mark.clustertest
def test_cancel_run(cpu):
    print_fn = rh.function(fn=do_printing_and_logging, system=cpu)
    key = print_fn.remote()
    assert isinstance(key, str)
    res = cpu.cancel(key)
    assert res == "Cancelled"
    try:
        cpu.get(key, stream_logs=True)
    except Exception as e:
        assert "This task or its dependency was cancelled by" in str(e)


if __name__ == "__main__":
    unittest.main()
