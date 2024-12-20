Processes
=========

.. raw:: html

    <p><a href="https://colab.research.google.com/github/run-house/notebooks/blob/stable/docs/api-process.ipynb">
    <img height="20px" width="117px" src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a></p>

On your Runhouse cluster, whether you have one node or multiple nodes,
you may want to run things in different processes on the cluster.

There are a few key use cases for separating your logic into different
processes:

1. Creating processes that require certain amounts of resources.
2. Creating processes on specific nodes.
3. Creating processes with specific environment variables.
4. General OS process isolation – allowing you to kill things on the
   cluster without touching other running logic.

You can put your Runhouse Functions/Modules into specific processes, or
even run bash commands in specific processes.

Let’s set up a basic cluster and some easy logic to send to it.

.. code:: ipython3

    def see_process_attributes():
        import os
        import time
        import socket

        log_level = os.environ.get("LOG_LEVEL")

        if log_level == "DEBUG":
            print("Debugging...")
        else:
            print("No log level set.")

        # Return the IP that this is scheduled on
        return socket.gethostbyname(socket.gethostname())


.. code:: ipython3

    import runhouse as rh

    cluster = rh.cluster(name="multi-gpu-cluster", accelerators="A10G:1", num_nodes=2, provider="aws").up_if_not()


.. parsed-literal::
    :class: code-output

    I 12-17 13:12:17 provisioner.py:560] [32mSuccessfully provisioned cluster: multi-gpu-cluster[0m
    I 12-17 13:12:18 cloud_vm_ray_backend.py:3402] Run commands not specified or empty.
    Clusters
    [2mAWS: Fetching availability zones mapping...[0mNAME               LAUNCHED        RESOURCES                                                                  STATUS  AUTOSTOP  COMMAND
    multi-gpu-cluster  a few secs ago  2x AWS(g5.xlarge, {'A10G': 1})                                             UP      (down)    /Users/rohinbhasin/minico...
    ml_ready_cluster   1 hr ago        1x AWS(m6i.large, image_id={'us-east-1': 'docker:python:3.12.8-bookwor...  UP      (down)    /Users/rohinbhasin/minico...

    [?25h

We can now create processes based on whatever requirements we want.
Covering all the examples above:

.. code:: ipython3

    # Create some processes with GPU requirements. These will be on different nodes since each node only has one GPU, and we'll check that
    p1 = cluster.ensure_process_created("p1", compute={"GPU": 1})
    # This second process will also have an env var set.
    p2 = cluster.ensure_process_created("p2", compute={"GPU": 1}, env_vars={"LOG_LEVEL": "DEBUG"})

    # We can also send processes to specific nodes if we want
    p3 = cluster.ensure_process_created("p3", compute={"node_idx": 1})

    cluster.list_processes()




.. parsed-literal::
    :class: code-output

    {'default_process': {'name': 'default_process',
      'compute': {},
      'runtime_env': None,
      'env_vars': {}},
     'p1': {'name': 'p1',
      'compute': {'GPU': 1},
      'runtime_env': {},
      'env_vars': None},
     'p2': {'name': 'p2',
      'compute': {'GPU': 1},
      'runtime_env': {},
      'env_vars': {'LOG_LEVEL': 'DEBUG'}},
     'p3': {'name': 'p3',
      'compute': {'node_idx': 1},
      'runtime_env': {},
      'env_vars': None}}



Note that we always create a ``default_process``, which is where all
Runhouse Functions/Modules end up if you don’t specify processes when
sending them to the cluster. This ``default_process`` always lives on
the head node of your cluster.

Now, let’s see where these processes ended up using our utility method
set up above.

.. code:: ipython3

    remote_f1 = rh.function(see_process_attributes).to(cluster, process=p1)
    print(remote_f1())


.. parsed-literal::
    :class: code-output

    INFO | 2024-12-17 13:23:01 | runhouse.resources.functions.function:236 | Because this function is defined in a notebook, writing it out to /Users/rohinbhasin/work/notebooks/see_process_attributes_fn.py to make it importable. Please make sure the function does not rely on any local variables, including imports (which should be moved inside the function body). This restriction does not apply to functions defined in normal Python files.
    INFO | 2024-12-17 13:23:04 | runhouse.resources.module:507 | Sending module see_process_attributes of type <class 'runhouse.resources.functions.function.Function'> to multi-gpu-cluster
    INFO | 2024-12-17 13:23:04 | runhouse.servers.http.http_client:439 | Calling see_process_attributes.call
    [36mNo log level set.
    [0mINFO | 2024-12-17 13:23:04 | runhouse.servers.http.http_client:504 | Time to call see_process_attributes.call: 0.71 seconds
    172.31.89.87


.. code:: ipython3

    remote_f2 = rh.function(see_process_attributes).to(cluster, process=p2)
    print(remote_f2())


.. parsed-literal::
    :class: code-output

    INFO | 2024-12-17 13:23:32 | runhouse.resources.functions.function:236 | Because this function is defined in a notebook, writing it out to /Users/rohinbhasin/work/notebooks/see_process_attributes_fn.py to make it importable. Please make sure the function does not rely on any local variables, including imports (which should be moved inside the function body). This restriction does not apply to functions defined in normal Python files.
    INFO | 2024-12-17 13:23:34 | runhouse.resources.module:507 | Sending module see_process_attributes of type <class 'runhouse.resources.functions.function.Function'> to multi-gpu-cluster
    INFO | 2024-12-17 13:23:34 | runhouse.servers.http.http_client:439 | Calling see_process_attributes.call
    [36mDebugging...
    [0mINFO | 2024-12-17 13:23:35 | runhouse.servers.http.http_client:504 | Time to call see_process_attributes.call: 0.53 seconds
    172.31.94.40


We can see that, since each process required one GPU, they were
scheduled on different machines. You can also see that the environment
variable we set in the second process was propagated, as our logging
output is different. Let’s check now that the 3rd process we explicitly
sent to the second node is on the second node.”

.. code:: ipython3

    remote_f3 = rh.function(see_process_attributes).to(cluster, process=p3)
    print(remote_f3())


.. parsed-literal::
    :class: code-output

    INFO | 2024-12-17 13:27:05 | runhouse.resources.functions.function:236 | Because this function is defined in a notebook, writing it out to /Users/rohinbhasin/work/notebooks/see_process_attributes_fn.py to make it importable. Please make sure the function does not rely on any local variables, including imports (which should be moved inside the function body). This restriction does not apply to functions defined in normal Python files.
    INFO | 2024-12-17 13:27:08 | runhouse.resources.module:507 | Sending module see_process_attributes of type <class 'runhouse.resources.functions.function.Function'> to multi-gpu-cluster
    INFO | 2024-12-17 13:27:08 | runhouse.servers.http.http_client:439 | Calling see_process_attributes.call
    [36mNo log level set.
    [0mINFO | 2024-12-17 13:27:08 | runhouse.servers.http.http_client:504 | Time to call see_process_attributes.call: 0.54 seconds
    172.31.94.40


Success! We can also ``run_bash`` within a specific process, if we want
to make sure our bash command runs on the same node as a function we’re
running.

.. code:: ipython3

    cluster.run_bash("ip addr", process=p2)




.. parsed-literal::
    :class: code-output

    [[0,
      '1: lo: <LOOPBACK,UP,LOWER_UP> mtu 65536 qdisc noqueue state UNKNOWN group default qlen 1000\n    link/loopback 00:00:00:00:00:00 brd 00:00:00:00:00:00\n    inet 127.0.0.1/8 scope host lo\n       valid_lft forever preferred_lft forever\n    inet6 ::1/128 scope host \n       valid_lft forever preferred_lft forever\n2: ens5: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 9001 qdisc mq state UP group default qlen 1000\n    link/ether 12:4c:76:66:e8:bb brd ff:ff:ff:ff:ff:ff\n    altname enp0s5\n    inet 172.31.94.40/20 brd 172.31.95.255 scope global dynamic ens5\n       valid_lft 3500sec preferred_lft 3500sec\n    inet6 fe80::104c:76ff:fe66:e8bb/64 scope link \n       valid_lft forever preferred_lft forever\n3: docker0: <NO-CARRIER,BROADCAST,MULTICAST,UP> mtu 1500 qdisc noqueue state DOWN group default \n    link/ether 02:42:ac:9e:2b:8f brd ff:ff:ff:ff:ff:ff\n    inet 172.17.0.1/16 brd 172.17.255.255 scope global docker0\n       valid_lft forever preferred_lft forever\n',
      ''],
     [...]]



You can see that this ran on the second node. Finally, you can also kill
processes, which you may want to do if you use asyncio to run long
running functions in a process.

.. code:: ipython3

    cluster.kill_process(p3)
    cluster.list_processes()




.. parsed-literal::
    :class: code-output

    {'default_process': {'name': 'default_process',
      'compute': {},
      'runtime_env': None,
      'env_vars': {}},
     'p1': {'name': 'p1',
      'compute': {'GPU': 1},
      'runtime_env': {},
      'env_vars': None},
     'p2': {'name': 'p2',
      'compute': {'GPU': 1},
      'runtime_env': {},
      'env_vars': {'LOG_LEVEL': 'DEBUG'}}}
