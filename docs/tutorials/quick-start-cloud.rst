Cloud Quick Start
=================

.. raw:: html

    <p><a href="https://colab.research.google.com/github/run-house/notebooks/blob/stable/docs/quick-start-cloud.ipynb">
    <img height="20px" width="117px" src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a></p>

Runhouse lets you quickly and easily deploy your Python code as
production-grade applications on your own infra.

This tutorial demonstrates how to

-  Start a cloud VM with the Runhouse API server running on it
-  Send a locally defined function onto the VM to serve it as a service.

Installing Runhouse
-------------------

The Runhouse base package can be installed with:

.. code:: ipython3

    !pip install runhouse

To use Runhouse to launch on-demand clusters, please instead run the
following command. This additionally installs
`SkyPilot <https://github.com/skypilot-org/skypilot>`__, which is used
for launching fresh VMs through your cloud provider.

.. code:: ipython3

    !pip install "runhouse[sky]"

Local Python Function
---------------------

First, let’s define the function that we want to be run on our remote
compute. This is just a regular Python function; no decorators,
wrappers, or configs are necessary.

.. code:: ipython3

    def get_pid(a = 0):
        import os
        return os.getpid()

Runhouse Cluster
----------------

In Runhouse, a “cluster” is a unit of compute, somewhere you can send
code, data, or requests to execute. We define a Runhouse cluster using
the ``rh.cluster`` factory function.

This requires having access to a cluster or a cloud provider account. If
you do not have access to a cluster, you can try the `local
version <https://www.run.house/docs/tutorials/quick-start-local>`__ of this
tutorial, which sets up and deploys the Python function to a local
server, rather than a remote cluster.

To use a cluster that’s already running:

.. code:: ipython3

    cluster = rh.cluster(
        name="rh-cluster",
        host="example-cluster",  # hostname or ip address,
        ssh_creds={"ssh_user": "ubuntu", "ssh_private_key": "~/.ssh/id_rsa"},  # credentials for ssh-ing into the cluster
    )

If you do not have a cluster up, but have cloud credentials (e.g. aws,
gcp, azure) for launching clusters, you can set up and launch an
on-demand cluster with ``rh.ondemand_cluster``. This uses SkyPilot under
the hood, so run ``sky check`` on CLI first to set up the cloud
credentials locally.

.. code:: ipython3

    cluster = rh.ondemand_cluster(
        name="rh-cluster",
        instance_type="CPU:2+",
        provider="aws"
    )
    cluster.up_if_not()

    # terminate this cluster with `cluster.teardown()` in Python, or `sky down rh-cluster` in CLI

Deployment
----------

For the function, simply wrap it in ``rh.function``, then send it to the
cluster with ``.to``. This sets up the function on the cluster as a
proper service, by syncing over the code and setting up and specified
dependencies. Furthermore, it runs through SSH, and no additional auth,
port, or manual setup is necessary.

Modules, or classes, are also supported. For finer control of where the
function/module runs, you will also be able to specify the environment
(a list of package requirements, a Conda env, or Runhouse env) where it
runs. These are covered in more detail in the API tutorials.

.. code:: ipython3

    remote_fn = rh.function(get_pid).to(cluster)


.. parsed-literal::
    :class: code-output

    INFO | 2024-02-26 21:01:50.579156 | Writing out function to /Users/caroline/Documents/runhouse/notebooks/docs/get_pid_fn.py. Please make sure the function does not rely on any local variables, including imports (which should be moved inside the function body).
    INFO | 2024-02-26 21:01:50.584346 | Copying package from file:///Users/caroline/Documents/runhouse/notebooks to: rh-cluster
    INFO | 2024-02-26 21:01:54.745264 | Calling base_env.install


.. parsed-literal::
    :class: code-output

    [36mInstalling Package: notebooks with method reqs.
    [0m[36mreqs path: notebooks/requirements.txt
    [0m[36mnotebooks/requirements.txt not found, skipping
    [0m

.. parsed-literal::
    :class: code-output

    INFO | 2024-02-26 21:01:56.116714 | Time to call base_env.install: 1.37 seconds
    INFO | 2024-02-26 21:02:04.892297 | Sending module get_pid to rh-cluster



.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">
    </pre>



The function we defined above, ``get_pid``, now exists remotely on the
cluster, and can be called remotely using ``remote_fn``. You can call
this remote function just as you would any other Python function, with
``remote_fn()``, and it runs on the cluster and returns the result to
our local environment.

Below, we run both the local and remote versions of this function, which
give different results and confirms that the functions are indeed being
run on different processes.

.. code:: ipython3

    print(f"Local PID {get_pid()}")
    print(f"Remote PID {remote_fn()}")


.. parsed-literal::
    :class: code-output

    INFO | 2024-02-26 21:02:43.117612 | Calling get_pid.call
    INFO | 2024-02-26 21:02:44.228964 | Time to call get_pid.call: 1.11 seconds




.. parsed-literal::
    :class: code-output

    Local PID 27818
    Remote PID 33366




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

To learn more, please take a look at our other tutorials, or at the `API
reference <https://www.run.house/docs/api/python>`__
