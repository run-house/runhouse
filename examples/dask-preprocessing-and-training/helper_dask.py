# ## Helper Function: Launch a Dask cluster on the Runhouse cluster
# We start a scheduler on the head node and workers on the other nodes
def launch_dask_cluster(cluster):
    import threading

    def start_scheduler():
        cluster.run("dask scheduler --port 8786")

    def start_worker(head_node_ip, node):
        cluster.run(f"dask worker tcp://{head_node_ip}:8786", node=node)

    stable_ips = [ip[0] for ip in cluster.stable_internal_external_ips]
    head_node_ip = stable_ips[0]

    # Start the scheduler on the head node
    scheduler_thread = threading.Thread(target=start_scheduler)
    scheduler_thread.daemon = True
    scheduler_thread.start()

    # Start workers and connect to the scheduler
    import time
    for ip in cluster.ips:
        time.sleep(3)
        cluster.run('pip install "dask[distributed]"', node=ip)
        worker_thread = threading.Thread(target=start_worker, args=(head_node_ip, ip))
        worker_thread.daemon = True
        worker_thread.start()
