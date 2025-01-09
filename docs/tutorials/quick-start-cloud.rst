Minimal Example
===============

.. raw:: html

    <p><a href="https://colab.research.google.com/github/run-house/notebooks/blob/stable/docs/quick-start-cloud.ipynb">
    <img height="20px" width="117px" src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a></p>

This tutorial demonstrates how to use Runhouse to:

- Launch a Runhouse cluster from elastic compute
- Send a locally defined function onto the remote compute and call it on
  the remote cluster.

Runhouse lets you dispatch and execute regular Python on remote compute
“serverlessly” whether the program is a simple function or a distributed
multi-node training. You have rapid, local-like development and
iteration from your local IDE while having the flexibility to execute on
powerful remote compute. Then, identically execute your code in
production simply by scheduling the code to launch compute and dispatch
to it, but without any need to change the underlying program code.

We assume you have already installed and set up Runhouse according to
the `setup
guide <https://www.run.house/docs/tutorials/quick-start-den>`__.

.. code:: ipython3

    import runhouse as rh
    rh.login('your token') # From https://www.run.house/account if not logged in via CLI command `runhouse login` already

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
provider account to launch elastic compute, or a Kubernetes cluster.
Here, we will launch an on-demand cluster using elastic compute. As
noted above, if you have not already enabled launching with Runhouse,
you should review `Setting Up
Runhouse <https://www.run.house/docs/tutorials/quick-start-den>`__.

.. code:: ipython3

    cluster = rh.ondemand_cluster(
        name="rh-cluster",
        num_cpus="4",
        provider="aws", # gcp, kubernetes, etc.
        launcher="den" # Switch to `local` if you are using Runhouse to launch from your local machine via Skypilot
    )
    cluster.up_if_not()

There are a number of options to specify the resources more finely, such
as GPUs (``accelerators="A10G:4"``), cloud provider names
(``instance_type="m5.xlarge"``), ``num_nodes=n`` for multiple instances,
``memory``, ``disk_size``, ``region``, ``use_spot``, and more. See the
`on_demand_cluster
docs <https://www.run.house/docs/api/python/cluster#runhouse.ondemand_cluster>`__.

To use a cluster that’s already running:

.. code:: ipython3

    cluster = rh.cluster(
        name="rh-cluster",
        host="example-cluster",  # hostname or ip address,
        ssh_creds={"ssh_user": "ubuntu", "ssh_private_key": "~/.ssh/id_rsa"},  # credentials for ssh-ing into the cluster
    )

Deploy Code to the Cluster
--------------------------

Simply wrap the function in ``rh.function`` and send it to the cluster
with ``.to``. This deploys the function to the cluster as a service by
syncing over the code, importing the synced code, and serving it in the
Runhouse API server.

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


Saving and Reloading
--------------------

You can save the resources we created above to your Runhouse account
with the ``.save()`` method.

.. code:: ipython3

    remote_get_platform.save()
    cluster.save() # Clusters are automatically be saved by Runhouse

Once saved, resources can be reloaded from any environment in which you
are logged into. For instance, if you are running this in a Colab
notebook, you can jump into your terminal, call ``runhouse login``, and
then reconstruct and run the function on the cluster with the following
Python script:

.. code:: ipython3

   import runhouse as rh

   if __name__ == "__main__":
       reloaded_fn = rh.function(name="get_platform")
       print(reloaded_fn())

The ``name`` used to reload the function is the method name by default.
You can customize a function name using the following syntax:

.. code:: ipython3

   remote_get_platform = rh.function(fn=get_platform, name="my_function").to(cluster)

Sharing
-------

You can also share your resource with collaborators, and choose which
level of access to give. Once shared, they will be able to see the
resource in their dashboard as well, and be able to load and use the
shared resource. They’ll need to load the resource using its full name,
which includes your username (``/your_username/get_platform``).

.. code:: ipython3

    remote_get_platform.share(
        users=["teammate1@email.com"],
        access_level="write",
    )

Web UI
------

After saving your resources, you can log in and see them on your `Den
dashboard <https://www.run.house/dashboard>`__, labeled as
``/<username>/rh-cluster`` and ``/<username>/get_platform``.

Clicking into the resource provides information about your resource. You
can view the resource metadata, previous versions, and activity, or add
a description to the resource.

Teardown
--------

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
