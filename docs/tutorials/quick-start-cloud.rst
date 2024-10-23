Quick Start
===========

.. raw:: html

    <p><a href="https://colab.research.google.com/github/run-house/notebooks/blob/stable/docs/quick-start-cloud.ipynb">
    <img height="20px" width="117px" src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a></p>

Runhouse lets you serverlessly dispatch and execute regular Python on
your existing cloud infrastructure. You can quickly develop, test, and
iterate on your ML programs, from your local IDE while executing on
powerful remote compute. Then, identically execute your code in
production simply by scheduling the dispatch while keeping the
underlying program code exactly the same.

This tutorial demonstrates how to

- Connect to an existing remote IP, fresh cloud VM, or fresh Kubernetes
  pod in Python as a Runhouse cluster
- Send a locally defined function onto the remote compute and call it as
  a service

Installing Runhouse
-------------------

The Runhouse base package can be installed with:

.. code:: ipython3

    !pip install runhouse

To use Runhouse to launch on-demand clusters, please instead run the
following command.

.. code:: ipython3

    !pip install "runhouse[sky]"

.. code:: ipython3

    import runhouse as rh

Local Python Function
---------------------

First, let’s define the function that we want to be run on our remote
compute. This is just a regular Python function; no decorators,
wrappers, or configs are necessary.

.. code:: ipython3

    def get_platform(a = 0):
        import platform
        return platform.platform()

Runhouse Cluster
----------------

In Runhouse, a “cluster” is a unit of compute, somewhere you can send
code, data, or requests to execute. We define a Runhouse cluster using
the ``rh.cluster`` factory function.

This requires having access to an existing VM (via SSH), a cloud
provider account to launch elastic compute, or a Kubernetes cluster
(~/.kube/config). If you do not have access to a cluster, you can try
the `local
version <https://www.run.house/docs/tutorials/quick-start-local>`__ of
this tutorial, which sets up and deploys the Python function to a local
server.

To use a cluster that’s already running:

.. code:: ipython3

    cluster = rh.cluster(
        name="rh-cluster",
        host="example-cluster",  # hostname or ip address,
        ssh_creds={"ssh_user": "ubuntu", "ssh_private_key": "~/.ssh/id_rsa"},  # credentials for ssh-ing into the cluster
    )

If you do not have a cluster up, but have cloud credentials (e.g. AWS,
Google Cloud, Azure) for launching clusters or a kubeconfig for an
existing Kubernetes cluster, you can set up and launch an on-demand
cluster with ``rh.ondemand_cluster``. This uses SkyPilot under the hood,
so run ``sky check`` in a CLI first to make sure credentials are set up
properly.

.. code:: ipython3

    cluster = rh.ondemand_cluster(
        name="rh-cluster",
        instance_type="CPU:2+",
        provider="aws"
    )
    cluster.up_if_not()

There are a number of options to specify the resources more finely, such
as GPUs (``instance_type="A10G:4"``), cloud provider names
(``instance_type="m5.xlarge"``), ``num_instances=n`` for multiple
instances, ``memory``, ``disk_size``, ``region``, ``image_id``,
``open_ports``, ``spot``, and more. See the `on_demand_cluster
docs <https://www.run.house/docs/api/python/cluster#runhouse.ondemand_cluster>`__.
You can also omit the provider argument to allocate from the cheapest
available source for which you have credentials.

Deploy Code to the Cluster
--------------------------

Simply wrap the function in ``rh.function`` and send it to the cluster
with ``.to``. This deploys the function to the cluster as a service by
syncing over the code, setting up any specified Runhouse environment
(see `Envs <https://www.run.house/docs/tutorials/api-envs>`__ or
dependencies, environment variables, secrets, conda environments),
importing the synced code, and serving it in the Runhouse API server.

Classes, or ``Modules`` are also supported. Remote instances of a remote
class have persisted state, enabling powerful usage patterns.

.. code:: ipython3

    remote_get_platform = rh.function(get_platform).to(cluster)


.. parsed-literal::
    :class: code-output

    INFO | 2024-05-16 03:20:53.066103 | Because this function is defined in a notebook, writing it out to /Users/donny/code/notebooks/docs/get_platform_fn.py to make it importable. Please make sure the function does not rely on any local variables, including imports (which should be moved inside the function body). This restriction does not apply to functions defined in normal Python files.
    INFO | 2024-05-16 03:20:53.079931 | Port 32300 is already in use. Trying next port.
    INFO | 2024-05-16 03:20:53.081995 | Forwarding port 32301 to port 32300 on localhost.
    INFO | 2024-05-16 03:20:54.215570 | Server rh-cluster is up.
    INFO | 2024-05-16 03:20:54.224806 | Copying package from file:///Users/donny/code/notebooks to: rh-cluster
    INFO | 2024-05-16 03:20:55.395007 | Calling _cluster_default_env.install
    INFO | 2024-05-16 03:20:55.948421 | Time to call _cluster_default_env.install: 0.55 seconds
    INFO | 2024-05-16 03:20:55.960756 | Sending module get_platform of type <class 'runhouse.resources.functions.function.Function'> to rh-cluster


Deploying the function to the cluster took ~2 seconds, and the function
we defined above, ``get_platform``, now exists remotely on the cluster,
and can be called remotely using ``remote_fn``. You can call this remote
function normally from local, with ``remote_fn()``, and it runs on the
cluster and returns the result to our local environment.

When we run the local and remote versions of this function, you see
different results based on where it executes.

.. code:: ipython3

    print(f"Local Platform: {get_platform()}")
    print(f"Remote Platform: {remote_get_platform()}")


.. parsed-literal::
    :class: code-output

    INFO | 2024-05-16 03:21:03.941205 | Calling get_platform.call


.. parsed-literal::
    :class: code-output

    Local Platform: macOS-14.4.1-arm64-arm-64bit


.. parsed-literal::
    :class: code-output

    INFO | 2024-05-16 03:21:04.513689 | Time to call get_platform.call: 0.57 seconds


.. parsed-literal::
    :class: code-output

    Remote Platform: Linux-5.15.0-1049-aws-x86_64-with-glibc2.31


If you launched an on-demand cluster, you can terminate it by calling
``cluster.teardown()``.

.. code:: ipython3

    cluster.teardown()

Dive Deeper
-----------

What we just did, running a locally defined function on remote compute,
is just the tip of the iceberg of what’s possible with Runhouse. With a
large suite of even more abstractions and features, Runhouse lets you
quickly and seamlessly integrate between local and remote environments.

We recommend you now review the `extended guide on getting
started <https://www.run.house/docs/tutorials/quick-start-cloud>`__ with
Runhouse. You can also take a look at our
`examples <https://www.run.house/examples>`__ or at the `API
reference <https://www.run.house/docs/api/python>`__
