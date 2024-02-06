Getting Started
===============

.. raw:: html

    <p><a href="https://colab.research.google.com/github/run-house/runhouse/blob/stable/docs/notebooks/api/quick_start.ipynb">
    <img height="20px" width="117px" src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a></p>

Runhouse is a Python framework for composing and sharing
production-quality backend apps and services ridiculously quickly and on
your own infra.

This getting started guide gives a quick walkthrough of Runhouse basics:

-  Create your own Runhouse resource abstractions for a cluster and
   function
-  Send local code to remote infra to be run instantly
-  Save/reload/share resources through Runhouse Den resource management

Also, sneak peaks of more advanced Runhouse features and where to look
to learn more about them.

Installing Runhouse
-------------------

.. code:: ipython3

    !pip install runhouse

To use runhouse to interact with remote clusters, please instead run the
following command. This additionally installs
`SkyPilot <https://github.com/skypilot-org/skypilot>`__, which is used
for launching on-demand clusters and interacting with runhouse clusters.

.. code:: ipython3

    !pip install runhouse[sky]

Runhouse Basics: Remote Cluster and Function
--------------------------------------------

Let’s start by seeing how simple it is to send arbitrary code, in this
case a function, and run it on your remote compute.

.. code:: ipython3

    import runhouse as rh

.. code:: ipython3

    def run_home(name: str):
        return f"Run home {name}!"

To run this function on remote compute:

1. Construct a RH cluster, which wraps your remote compute
2. Create a RH function for ``run_home``, and send it to exist and run
   on the cluster
3. Call the RH function as you would any other function. This function
   runs on your remote cluster and returns the results to local

Runhouse Cluster
~~~~~~~~~~~~~~~~

Construct your Runhouse cluster by wrapping an existing cluster you have
up. In Runhouse, a “cluster” is a unit of compute, somewhere you can
send code, data, or requests to execute.

More advanced cluster types like on-demand (automatically spun up/down
for you with your cloud credentials) or Sagemaker clusters are also
supported. These require some setup and are discussed in `Compute
Tutorial <https://www.run.house/docs/tutorials/api/compute>`__.

.. code:: ipython3

    cluster = rh.cluster(
        name="rh-cluster",
        host="example-cluster",  # hostname or ip address,
        ssh_creds={"ssh_user": "ubuntu", "ssh_private_key": "~/.ssh/sky-key"}
    )

Runhouse Function
~~~~~~~~~~~~~~~~~

For the function, simply wrap it in ``rh.function``, then send it to the
cluster with ``.to``.

Modules, or classes, are also supported. For finer control of where the
function/module runs, you will also be able to specify the environment
(a list of package requirements, a Conda env, or Runhouse env) where it
runs. These are covered in more detail in the `Compute
Tutorial <https://www.run.house/docs/tutorials/api/compute>`__.

.. code:: ipython3

    remote_fn = rh.function(run_home).to(cluster)


.. parsed-literal::
    :class: code-output

    INFO | 2024-01-04 19:16:41.757114 | Writing out function to /Users/caroline/Documents/runhouse/runhouse/docs/notebooks/api/run_home_fn.py. Please make sure the function does not rely on any local variables, including imports (which should be moved inside the function body).
    INFO | 2024-01-04 19:16:41.760892 | Setting up Function on cluster.
    INFO | 2024-01-04 19:16:41.763236 | Copying package from file:///Users/caroline/Documents/runhouse/runhouse to: example-cluster
    INFO | 2024-01-04 19:16:41.764370 | Running command: ssh -T -i ~/.ssh/sky-key -o Port=22 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o IdentitiesOnly=yes -o ExitOnForwardFailure=yes -o ServerAliveInterval=5 -o ServerAliveCountMax=3 -o ConnectTimeout=30s -o ForwardAgent=yes -o ControlMaster=auto -o ControlPath=/tmp/skypilot_ssh_caroline/41014bb4d3/%C -o ControlPersist=300s ubuntu@example-cluster 'bash --login -c -i '"'"'true && source ~/.bashrc && export OMP_NUM_THREADS=1 PYTHONWARNINGS=ignore && (mkdir -p ~/runhouse/)'"'"' 2>&1'
    INFO | 2024-01-04 19:16:42.395589 | Calling base_env.install


.. parsed-literal::
    :class: code-output

    base_env servlet: Calling method install on module base_env
    Env already installed, skipping


.. parsed-literal::
    :class: code-output

    INFO | 2024-01-04 19:16:42.822556 | Time to call base_env.install: 0.43 seconds
    INFO | 2024-01-04 19:16:43.044922 | Function setup complete.


.. code:: ipython3

    remote_fn("Jack")


.. parsed-literal::
    :class: code-output

    INFO | 2024-01-04 19:16:46.179161 | Calling run_home.call


.. parsed-literal::
    :class: code-output

    base_env servlet: Calling method call on module run_home


.. parsed-literal::
    :class: code-output

    INFO | 2024-01-04 19:16:46.426212 | Time to call run_home.call: 0.25 seconds



.. parsed-literal::
    :class: code-output

    'Run home Jack!'


Extensions
~~~~~~~~~~

That was just a very basic example of taking local code/data, converting
it to a Runhouse object, sending it to remote compute, and running it
there. Each of these Runhouse abstractions, like a cluster or a
function, is referred to as a Resource. Runhouse supports a whole lot of
extra functionality on this front, including

-  Automated clusters: On-demand clusters (through SkyPilot), AWS
   Sagemaker clusters, and (soon) Kubernetes clusters
-  Env and package management: run functions/modules on dedicated envs
   on the cluster
-  Modules: setup and run Python classes in addition to functions
-  Additional function flexibility/features: remote or async functions,
   with logging, streaming, etc
-  Data resources: send folders, tables, blobs to remote clusters or
   file storage

Runhouse Den: Saving, Reloading, and Sharing
--------------------------------------------

By creating a `Runhouse Den <https://www.run.house/dashboard>`__ account
and logging in, you can save down your resources (cluster,
function/module, data, etc), reload them from anywhere, or even share
with other users, like your teammates. Once loaded, these resources are
ready to be used without additional setup required.

Then, on the Web dashboard UI, access, visualize, and manage any of your
resources, along with version history.

Login
~~~~~

To login, simply call ``rh.login()`` in Python or ``runhouse login`` in
CLI. As part of logging in, Runhouse also optionally offers secrets
management, where it can automatically detect locally set up provider
secrets, and gives you the option to upload them securely into your
account. For more information on Secrets management, refer to the
`Secrets
Tutorial <https://www.run.house/docs/main/en/tutorials/api/secrets>`__.

.. code:: ipython3

    rh.login()

Saving and Reloading
~~~~~~~~~~~~~~~~~~~~

You can save the resources we created above with:

.. code:: ipython3

    cluster.save()
    remote_fn.save()



.. parsed-literal::
    :class: code-output

    <runhouse.resources.functions.function.Function at 0x105bbdd60>


If you check on your `dashboard <https://www.run.house/dashboard>`__,
you’ll see that the cluster “rh-cluster” and function “run_home” have
been saved. Clicking into the resource will show you the resource
metadata.

Now, you can also jump to another environment (for example, your
terminal if running this on a notebook), call ``runhouse login``, and
then run the function on the cluster with the following Python script:

.. code:: ipython3

    """
    import runhouse as rh

    if __name__ == "__main__":
        reloaded_fn = rh.function(name="run_home")
        print(reloaded_fn("Jane"))
    """

Sharing
~~~~~~~

To share your resources with another user, such as a teammate, simply
call the following:

.. code:: ipython3

    remote_fn.share(
        users=["teammate1@email.com"],
        access_level="write",
    )

Conclusion
----------

To recap, in this guide we covered:

-  Creating basic Runhouse resource types: cluster and function
-  Running a Runhouse function on remote compute
-  Saving, reloading, and sharing resources through Runhouse Den

Dive Deeper
~~~~~~~~~~~

This is just the start, and there’s much more to Runhouse. To learn
more, please take a look at these other tutorials, or at the `API
documentation <https://www.run.house/docs/api/python>`__

-  Compute API Usage https://www.run.house/docs/tutorials/api/compute
-  Data API Usage https://www.run.house/docs/tutorials/api/data
-  Secrets Management https://www.run.house/docs/tutorials/api/secrets
-  Resource Management
   https://www.run.house/docs/tutorials/api/resources
-  Stable Diffusion Inference Example
   https://www.run.house/docs/tutorials/examples/inference
