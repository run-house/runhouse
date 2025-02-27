Development Guide
=================

This page offers a more detailed guide on using Runhouse to develop and deploy your ML projects. With Runhouse, you
can manage all of your compute and make it available through ephemeral clusters for both research and production
workflows.

- Launch compute from any source and manage all your cloud accounts and Kubernetes clusters as one pool.
- Iterably develop and reproducibly execute ML workloads at any scale.
- Execute distributed workloads on multiple node clusters without any infrastructure setup.
- Monitor resource usage in detail and access persisted logs in the web UI.
- Save and reload services and resources (clusters, functions, classes, etc).

Getting Started
---------------

This guide assumes you have already followed the Installation and Compute Setup steps of the `Installation and Setup
Guide <https://www.run.house/docs/installation-setup>`__.

For a quicker run-through, we recommend you first start with the `Quick Start
<https://www.run.house/docs/tutorials/quick-start-cloud>`_ guide.

If you have any questions about what is described here, please reach out to `hello@run.house <mailto:hello@run.house>`_
or ping us on `Discord <https://discord.gg/RnhB6589Hs>`_, and we'd be happy to walk you through the details.

1. Define Compute
-----------------
Runhouse allows you to define compute requirements in code, and launch ephemeral clusters from the compute pool we
described in the prior section. Here, you can define the required CPU, GPU, memory, and disk requirements (or name
a specific cluster) to use.

.. image:: https://runhouse-tutorials.s3.amazonaws.com/Pull+Compute+from+Compute+Pool.jpg
  :alt: Runhouse pulls compute from a pool of resources
  :width: 750
  :align: center

For instance, to create a cluster on AWS with an A10 GPU attached, you can write:

.. code:: python

    import runhouse as rh

    cluster = rh.ondemand_cluster(
        name="rh-cluster", # This cluster can be saved and reused by name. We will prefix your username when saved, e.g. /my_username/rh-cluster
        instance_type="A10G:1", # There are a number of options available for instance_type, check out the docs to see them all
        provider="aws", # Specify a cloud provider
        autostop_mins=90, # Remember to set autostop_mins to avoid leaving clusters running indefinitely.
        launcher="den", # Launch the cluster with Runhouse; use 'local' for local credentials
    ).up_if_not()

You can also define a Runhouse Image containing a base machine or Docker image, along with other setup steps (e.g.
package, installs, bash commands, env vars), and pass it into the factory function above to specify cluster setup prior
to starting Runhouse on the cluster.

After the cluster is up, you can also run CLI commands on the cluster using ``cluster.run_bash()`` to run additional
setup commands.

.. code:: python

    cluster.run_bash(['pip install numpy'])

You can find full documentation about the Runhouse cluster API in the `Cluster docs
<https://www.run.house/docs/tutorials/api-clusters>`_.

Starting the Runhouse Server Daemon
"""""""""""""""""""""""""""""""""""
Once the compute is brought up, the client will start the Runhouse API server daemon on the compute and form a secure network
connection (either over SSH or HTTP/S). This is what enables the magic of Runhouse.

* In the next step, you will dispatch regular Python functions and modules for remote execution.
  The Runhouse daemon can be thought of as a "Python object store," holding key-value pairs of names and these dispatched Python objects in
  memory, and exposing an HTTP API to call methods on those objects by name.

* By default, objects are held in a single default worker process but can be sent to other worker processes, including
  on other nodes in the cluster, to achieve powerful parallelism out of the box.

* When the object is used, there is a ``GET http://myserver:32300/my_object/my_method``, and the daemon will look up
  the object named "my_object," issue an instruction for its worker to call the method "my_method" on it, and return
  the result.

* The HTTP server and workers can handle thousands of concurrent calls per second, and have similar latency to Flask
  under most conditions.

.. note::

    To run the daemon on your local machine, you will need a few additional requirements not included
    in the base Runhouse package. You can install them by running :code:`pip install "runhouse[server]"`.

2. Dispatch Your Code
---------------------
Once you have established a connection to compute, the development pattern is to continuously dispatch code to the
cluster and execute it there. You are doing local-like execution and debug, but with the power of the remote compute.
Runhouse is agnostic to whether you dispatch using a Notebook or run directly from a Python script.

Specifically to do the dispatch, you wrap your local function with ``rh.function()`` or class with ``rh.module()``,
and send it to the cluster with ``.to(cluster)``. For functions, you can call them directly as if they were local
functions, and they run remotely. For modules, you instantiate a remote instance of the object which is stateful;
you can access this remote object by name and make multi-threaded calls to its methods.

When you ``.to()`` a local function or class to the cluster, the corresponding repository or package, along with any
local dependencies, is rsynced to the cluster. An instruction containing the import path is then sent to the cluster to
construct the function or class in a specific worker, and it is inserted into the key-value store. We avoid and
discourage serializing code, as serialization often leads to version mismatch errors between local and remote package
versions.

After the object is deployed to the server, the Runhouse Python client returns a local callable stub. It behaves like
the original object but forwards method calls over HTTP to the remote object on the cluster.

.. code:: python

      def add_two_numbers(a,b):
            return a+b

            remote_add = rh.function(add_two_numbers).to(cluster) # Send to the cluster that has already been defined above

.. code:: python

      class TorchTrainer:
         def __init__(self):
            ..

         def train(self, X, y):
            ..

         def test(self, X, y):
            ..


      if __name__ == "__main__":
         cluster.pip_install(["torch"])  # Install packages not already in the cluster image
         RemoteTrainer = rh.module(TorchTrainer).to(cluster)  # Send to cluster
         trainer = RemoteTrainer(name='remote-instance-of-trainer')  # Instantiate remote object

.. note::

      The code that should only run locally (e.g. defining compute, dispatch, and calling remote objects for execution)
      should live within a ``if __name__ == "__main__":`` block in a script. This way, the code will not execute on
      remote compute when it is sent there.

Read more about `functions and modules <https://www.run.house/docs/tutorials/api-modules>`_.

3. Execute Your Code Remotely
-----------------------------
It's now possible to use your remote objects as if they were local. From here on, you can think of Runhouse as
facilitating regular programming but with the objects living remotely, maybe in a different cluster,
region, or cloud than the local code. Python behavior such as async, exceptions, printing, and logging are all
preserved across remote calls, but can also be disabled or controlled if desired.

.. code:: python

      result = remote_add(1,2)
      print(result)

      X, y = ...  # Load data
      trainer.train(X,y)

As noted above, you should be iteratively dispatching and executing code. If you make local updates to the
``add_two_numbers`` function or the ``TorchTrainer`` class, you can simply re-run ``.to()``, and it should take <2
seconds to redeploy. The underlying cluster is persisted and stateful until you choose to down it, so you can take
advantage of the remote file system and memory during interactive development as well.

These remote objects are accessible from anywhere you are authenticated with Runhouse, so you and your team can make
multi-threaded calls against them. Calling microservices is actually a familiar pattern in programming; however, no
team would ever manually split their ML pipeline into multiple applications due to the DevOps overhead.


.. image:: https://runhouse-tutorials.s3.amazonaws.com/Iterative+Dispatch+from+Notebook.jpg
  :alt: Iteratively develop and dispatch code to remote execution
  :width: 550
  :align: center

4. Saving and Loading
---------------------
Runhouse resources (clusters, functions, modules) can be saved, shared, and reused based on a compact JSON metadata
signature. This allows for easy sharing of clusters and services across users and environments. For instance, the team
might want to use a single shared embeddings service to save costs and improve reproducibility.

Runhouse comes with a built-in metadata store / service registry called `Den <https://www.run.house/dashboard>`_ to
facilitate convenient saving, loading, sharing, and management of these resources. Den can be accessed via an HTTP
API or from any Python interpreter with a Runhouse token (either in ``~/.rh/config.yaml`` or an ``RH_TOKEN``
environment variable):

.. code-block:: python

    # Save to Den
    remote_add.save(name="my_function")

    # Reload the function and invoke it remotely on the cluster
    my_func = rh.function(name="/my_username/my_function")

    # Share the function with another user, giving them access to call or modify the resource
    my_func.share("user_a@gmail.com", access_level="write")

You can access the metadata directly by calling ``resource.config()`` and reconstruct the resource with
``<Resource Type>.from_config(config)``.

5. Terminating Modules, Workers, or Clusters
--------------------------------------------
When a remote object is no longer needed, it can be deallocated from the remote compute by calling
``cluster.delete(obj_name)``. This will remove the object from the key-value store and free up the memory on the
worker. A worker process can similarly be terminated with ``cluster.delete(worker_name)``, terminating its activities
and freeing its memory.

To down a cluster when the task is complete and the resource is no longer needed, you can simply call
``cluster.teardown()`` or let the autostop handle the cluster termination.

Moving to Production
--------------------
A key advantage of using Runhouse is that the code developed locally has already been executing production-like on
remote compute the entire time. This means research-to-production is an abstract checkpoint in development rather than
an actual task to rewrite pipelines for production over different hardware/data.

If your code is for a non-recurring task, then great, check your code into version control and you are already done. If
you are deploying a recurring job like recurring training, then simply move the Runhouse launching code into the
orchestrator or scheduler of your choice. You should not repackage ML code into orchestrator nodes and make
orchestrators your runtime. Instead, you should use orchestrators as minimal systems to schedule and observe your jobs,
but the jobs themselves will continue to be executed serverlessly with Runhouse from each node. This saves considerable
time upfront, setting up the first orchestrator to run in less than an hour (compared to multiple weeks in traditional
ML research-to-production).

As an example, you might want to make the first task of your orchestrator pipeline to simply bring up the cluster and
dispatch code to the new cluster. You can see that we are using the same underlying code (directly importing it from
a source file), and then reusing the object and cluster by name across steps.

.. code:: python

      @task()
      def up_and_dispatch():
            image = (
                rh.Image("base_setup")
                .from_docker("nvcr.io/nvidia/pytorch:23.10-py3")
                .pip_install(["torch"])
            )
            cluster = rh.ondemand_cluster(
                  name="rh-cluster",
                  instance_type="A10G:1",
                  provider="aws",
                  image=image,
            ).up_if_not()

            from my_code import TorchTrainer
            RemoteTrainer = rh.module(TorchTrainer).to(cluster)
            trainer = RemoteTrainer(name='remote-instance-of-trainer')

      @task()
      def embed():
            cluster = rh.cluster(name="rh-cluster")
            trainer = cluster.get(name='remote-instance-of-trainer')
            X, y = ...  # Load data
            trainer.train(X,y)

Runhouse recommends creating a Docker container which fixes the environment, dependencies, and program code for
production pipelines. There are significant benefits to containerization, rather than, for instance, worrying
about new breaking changes from package installation with PyPi. This is actually still unproblematic for additional
future iteration or debug, since you still easily interactively layer on changes to the environment from local, even
when you launch with the container.

.. image:: https://runhouse-tutorials.s3.amazonaws.com/Identical+Dispatch+in+Production.jpg
  :alt: Send code from research and production to compute
  :width: 750
  :align: center

My Pipeline is in Production, What's Next?
------------------------------------------
Once in production, your ML pipelines will inevitably encounter failures that require debugging. With Runhouse,
engineers can easily and locally reproduce production runs, modify the underlying code, and push changes to the
codebase. There’s no need to debug through the orchestrator or rebuild and resubmit. Furthermore, deploying with
Runhouse tends to result in fewer errors from the start, as the code is developed in a production-like environment.

This also makes the transition from production back to research seamless. Many teams dread revisiting the
research-to-production process, so once code is deployed, there’s often little motivation to make incremental
improvements to the pipeline. With Runhouse, pipelines already run serverlessly, ensuring that incremental changes
merged into the team codebase are automatically reflected in the production pipeline after being tested through
standard development processes.

There are other benefits to using Runhouse in production as you scale up usage. A few are included here:

* **Shared services**: Deploy shared services, such as an embeddings endpoint, and allow all pipelines to either call
  it by name as a live service or import the code from the team repository to deploy it independently within each
  pipeline. Any updates or improvements to the shared service are automatically applied to all pipelines without
  requiring changes to pipeline code.

* **Compute abstraction**: As you add new resources to your pool, get credits from new clouds, or get new quota, if all
  users are using Runhouse to allocate ephemeral compute, there is no need to update any code or configuration files at
  the user level. The new resources are added by the platform team, and then automatically adopted by the full team.

* **Infrastructure Migrations**: With Runhouse, your application code is entirely undecorated Python and the dispatch
  happens to arbitrary compute. If you ever choose to abandon your existing orchestrator, cloud provider, or any other
  tool, you simply have to move a small amount of dispatch code and infrastructure code configuration.

* **Adopting Distributed Frameworks**: Runhouse is a perfect complement to distributed frameworks, with some built-in
  abstractions that let you scale to multiple clusters or start using Ray clusters easily.
