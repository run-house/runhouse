import concurrent.futures
import queue

import runhouse as rh


def do_something(i: int):
    import os

    import ray

    return (
        os.getpid(),
        ray.runtime_context.RuntimeContext(ray.worker.global_worker).get_node_id(),
        i,
    )


q = queue.Queue()
results = []


def worker(clusterenv):
    cluster, env = clusterenv
    tasks_processed = 0
    do_something_remote = rh.function(do_something).to(system=cluster, env=env)

    while True:
        try:
            task = q.get(timeout=1)
            if task == -1:
                q.put(-1)
                break
            else:
                results.append(do_something_remote(task))
                tasks_processed += 1
        except queue.Empty:
            pass

    return tasks_processed


if __name__ == "__main__":
    num_nodes = 5
    cluster = (
        rh.cluster(name="rh-multinode", num_instances=num_nodes, instance_type="CPU:2")
        .save()
        .up_if_not()
    )
    cluster.restart_server()

    envs = []
    for i in range(num_nodes):
        env = rh.env(name=f"env_{i}", compute={"CPU": 1}).to(cluster).save()
        envs.append(env)

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_nodes) as executor:
        futs = [executor.submit(worker, (cluster, env)) for env in envs]
        for i in range(100):
            q.put(i)
        q.put(-1)
        for fut in concurrent.futures.as_completed(futs):
            print(fut.result())

    print(results)
