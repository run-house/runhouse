import logging
import time
import unittest

import runhouse as rh

from tests.test_send import multiproc_torch_sum

TEMP_FILE = "my_file.txt"
TEMP_FOLDER = "~/runhouse-tests"

logger = logging.getLogger(__name__)


def get_test_cluster():
    return rh.cluster(name="test-cpu").up_if_not()


def do_printing_and_logging():
    for i in range(6):
        # Wait to make sure we're actually streaming
        time.sleep(1)
        print(f"Hello from the cluster! {i}")
        logger.info(f"Hello from the cluster logs! {i}")
    return list(range(50))


def test_get_from_cluster():
    cluster = get_test_cluster()
    print_fn = rh.send(fn=do_printing_and_logging, hardware=cluster)
    key = print_fn.remote()
    assert isinstance(key, str)
    res = cluster.get(key, stream_logs=True)
    assert res == list(range(50))


def test_stream_logs():
    cluster = get_test_cluster()
    print_fn = rh.send(fn=do_printing_and_logging, hardware=cluster)
    res = print_fn(stream_logs=True)
    # TODO [DG] assert that the logs are streamed
    assert res == list(range(50))


def test_multiprocessing_streaming():
    cluster = get_test_cluster()
    re_fn = rh.send(multiproc_torch_sum, hardware=cluster, reqs=["./", "torch==1.12.1"])
    summands = list(zip(range(5), range(4, 9)))
    res = re_fn(summands, stream_logs=True)
    assert res == [4, 6, 8, 10, 12]


def test_cancel_run():
    cluster = get_test_cluster()
    print_fn = rh.send(fn=do_printing_and_logging, hardware=cluster)
    key = print_fn.remote()
    assert isinstance(key, str)
    res = cluster.cancel(key)
    assert res == "Cancelled"
    try:
        cluster.get(key, stream_logs=True)
    except Exception as e:
        assert "This task or its dependency was cancelled by" in str(e)


if __name__ == "__main__":
    unittest.main()
