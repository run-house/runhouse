Getting Started Example
=======================

.. raw:: html

    <p><a href="https://colab.research.google.com/github/run-house/runhouse/blob/stable/docs/notebooks/getting_started.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a></p>

In this basic getting started example, we demonstrate how you can use Runhouse to bridge the gap
between local and remote compute, and create Resources that can be saved, reused, and shared.

For instructions on installing Runhouse and setting up compute prior to running this example,
please first refer to :ref:`Installation and Setup Guide`.

To import runhouse:

.. code:: python

    import runhouse as rh

Optionally, to create and login to a (free) Runhouse account, for better resource saving and sharing
later on in the example:

.. code:: python

    rh.login()

Running local functions on remote hardware
------------------------------------------

First let's define a simple local function which returns the number of CPUs available.

.. code:: python

    def num_cpus():
        import multiprocessing
        return multiprocessing.cpu_count()

    num_cpus()

Next, instantiate the cluster that we want to run this function on. This can be either an existing
cluster where you pass in an IP address and SSH credentials, or a cluster associated with supported
Cloud account (AWS, GCP, Azure, LambdaLabs), where it is automatically launched (and optionally
terminated) for you.

.. code:: python

    # Using an existing, bring-your-own cluster
    cluster = rh.cluster(
                name="cpu-cluster",
                ips=['<ip of the cluster>'],
                ssh_creds={'ssh_user': '<user>', 'ssh_private_key':'<path_to_key>'},
            )

    # Using a Cloud provider
    cluster = rh.cluster(
                name="cpu-cluster",
                instance_type="CPU:8",
                provider="cheapest",      # options: "AWS", "GCP", "Azure", "Lambda", or "cheapest"
            )

If using a cloud cluster, we can launch the cluster with ``.up()`` or ``.up_if_not()``.

Note that it may take a few minutes for the cluster to be launched through the Cloud provider and
set up dependencies.

.. code:: python

    cluster.up_if_not()

Now that we have our function and remote cluster set up, we're ready to see how to run this function on our cluster!

We wrap our local function in ``rh.function``, and associate this new function with the cluster. Now, whenever we call
this new function, just as we would call any other Python function, it runs on the cluster instead of local.

.. code:: python

    num_cpus_cluster = rh.function(name="num_cpus_cluster", fn=num_cpus).to(system=cluster)

    num_cpus_cluster()

Saving, Reusing, and Sharing
----------------------------

Runhouse supports saving down the metadata and configs for resources like clusters and functions, so that you can load
them from a different environment, or share it with your collaborators.

.. code:: python

    num_cpus_cluster.save()

    num_cpus_cluster.share(
        users=["<email_to_runhouse_account>"],
        access_type="write",
    )

Now, you, or whoever you shared it with, can reload this function from anther dev environment (like a different Colab,
local, or on a cluster), as long as you are logged in to your Runhouse account.

.. code:: python

    reloaded_function = rh.function(name="num_cpus_cluster")
    reloaded_function()

Terminate the Cluster
---------------------

To terminate the cluster, you can run:

.. code:: python

    cluster.teardown()

Summary
-------

Here, we demonstrated how to use runhouse to create references to remote clusters, run local functions on the cluster, and save/share and reuse functions with a Runhouse account.

Runhouse also lets you:

- Send and save data (folders, blobs, tables) between local, remote, and file storage
- Send, save, and share dev environments
- Reload and reuse saved resources (both compute and data) from different environments (with a Runhouse account)
- ... and much more!
