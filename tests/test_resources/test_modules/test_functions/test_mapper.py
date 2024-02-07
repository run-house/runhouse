import logging
import multiprocessing
import os

import pytest

import runhouse as rh

REMOTE_FUNC_NAME = "@/remote_function"

logger = logging.getLogger(__name__)


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


def get_pid_and_ray_node(a=0):
    import ray

    return (
        os.getpid(),
        ray.runtime_context.RuntimeContext(ray.worker.global_worker).get_node_id(),
    )


class TestMapper:

    """Testing strategy:

    1. Explicit mapper - for each of the below, test map, starmap, call (round robin), and test that
        threaded calls are non-blocking:
        1. local mapper with local functions (rh.mapper(my_fn, processes=8))
        2. local mapper with remote functions (rh.mapper(rh.fn(my_fn).to(cluster), processes=8))
        3. remote mapper with local functions (rh.mapper(my_fn, processes=8).to(cluster))
        4. local mapper with additional replicas added manually (rh.mapper(fn, processes=8).add_replicas(my_replicas)
    2. Implicit mapper
        1. fn.map(), fn.starmap(), fn.call()

    """

    @pytest.mark.level("local")
    def test_local_mapper_remote_function(self, cluster):
        # Test .map()
        num_replicas = 3
        pid_fn = rh.function(getpid).to(cluster)
        mapper = rh.mapper(pid_fn, num_replicas=num_replicas)
        assert len(mapper.replicas) == num_replicas
        for i in range(num_replicas):
            assert mapper.replicas[i].system == cluster
        assert mapper.replicas[0].env.name == pid_fn.env.name
        assert mapper.replicas[1].env.name == pid_fn.env.name + "_replica_0"
        assert mapper.replicas[2].env.name == pid_fn.env.name + "_replica_1"
        pids = mapper.map([0] * 10)
        assert len(pids) == 10
        assert len(set(pids)) == num_replicas

        # Test .starmap() and reusing the envs
        summer_fn = rh.function(summer).to(cluster)
        sum_mapper = rh.mapper(summer_fn, num_replicas=num_replicas)
        assert len(sum_mapper.replicas) == num_replicas
        for i in range(num_replicas):
            assert sum_mapper.replicas[i].system == cluster
        assert sum_mapper.replicas[0].env.name == summer_fn.env.name
        assert sum_mapper.replicas[1].env.name == summer_fn.env.name + "_replica_0"
        assert sum_mapper.replicas[2].env.name == summer_fn.env.name + "_replica_1"
        res = sum_mapper.starmap([[1, 2]] * 10)
        assert res == [3] * 10
        res = sum_mapper.map([1] * 10, [2] * 10)
        assert res == [3] * 10

        # Doing this down here to confirm that first mapper using the envs isn't corrupted
        pids = mapper.starmap([[0]] * 10)
        assert len(pids) == 10
        assert len(set(pids)) == num_replicas

        # Test call
        assert len(set(mapper.call() for _ in range(4))) == 3

    @pytest.mark.level("thorough")
    def test_multinode_map(self, multinode_cpu_cluster):
        num_replicas = 6
        env = rh.env(compute={"CPU": 0.5}, reqs=["pytest"])
        pid_fn = rh.function(get_pid_and_ray_node).to(multinode_cpu_cluster, env=env)
        mapper = rh.mapper(pid_fn, num_replicas=num_replicas)
        assert len(mapper.replicas) == num_replicas
        for i in range(num_replicas):
            assert mapper.replicas[i].system == multinode_cpu_cluster
        ids = mapper.map([0] * 100)
        pids, nodes = zip(*ids)
        assert len(pids) == 100
        assert len(set(pids)) == num_replicas
        assert len(set(nodes)) == 2
        assert len(set(node for (pid, node) in [mapper.call() for _ in range(10)])) == 2

    @pytest.mark.skip
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
