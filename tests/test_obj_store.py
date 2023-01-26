import unittest

import runhouse as rh
import logging
import time

from tests.test_send import multiproc_torch_sum

TEMP_FILE = 'my_file.txt'
TEMP_FOLDER = '~/runhouse-tests'

logger = logging.getLogger(__name__)

def do_printing_and_logging():
    for i in range(6):
        # Wait to make sure we're actually streaming
        time.sleep(1)
        print(f'Hello from the cluster! {i}')
        logger.info(f'Hello from the cluster logs! {i}')
    return list(range(50))

def test_get_from_cluster():
    cluster = rh.cluster(name='^rh-cpu').up_if_not()
    print_fn = rh.send(fn=do_printing_and_logging, hardware=cluster)
    key = print_fn.remote()
    assert isinstance(key, str)
    res = cluster.get(key, stream_logs=True)
    assert res == list(range(50))

def test_stream_logs():
    cluster = rh.cluster(name='^rh-cpu').up_if_not()
    print_fn = rh.send(fn=do_printing_and_logging, hardware=cluster)
    res = print_fn(stream_logs=True)
    # TODO [DG] assert that the logs are streamed
    assert res == list(range(50))

def test_multiprocessing():
    re_fn = rh.send(multiproc_torch_sum, hardware='^rh-cpu', reqs=['./', 'torch==1.12.1'])
    summands = list(zip(range(5), range(4, 9)))
    res = re_fn(summands, stream_logs=True)
    assert res == [4, 6, 8, 10, 12]


if __name__ == '__main__':
    unittest.main()
