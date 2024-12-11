How to Use Runhouse
===================
This page offers a more detailed guide on using Runhouse to develop and deploy your ML projects. If you have any questions about what is described here, please reach out to
`hello@run.house <mailto:hello@run.house>`_ or ping us on Discord and we'd be happy to walk you through the details.

Quick Start
----------------
Before reviewing this detailed guide, we recommend you start with the `Quick Start <https://www.run.house/docs/tutorials/quick-start-cloud>`_ guide.

* Install Runhouse with ``pip install runhouse``

* Optionally install with a specific cloud like ``pip install "runhouse[aws]"`` or with SkyPilot for elastic compute ``pip install "runhouse[sky]``

* Optionally create an account on the `Runhouse website <https://www.run.house/dashboard>`_ or with ``runhouse login --sync-secrets`` to enable saving, reloading, and centralized authentication / secrets management.

Access to a Pool of Compute
-----------------------
In order to use Runhouse, you must be able to access compute resources, which can take any form (e.g. VMs, elastic compute, Kubernetes). You should
think about all the compute resources you have as a single pool, from which Runhouse allows you to launch ephemeral clusters to execute your code.

* **Kubernetes**: All you need is a kubeconfig

* **Elastic Compute**: We use Skypilot under the hood to launch elastic compute, and support most clouds. You can run ``sky check`` in CLI after installing Runhouse to confirm you have access to the cloud.

* **Existing Clusters**: Runhouse supports a variety of authentication methods to access existing clusters, including SSH with keys or passwords.

For initial projects and getting started quickly, launching from local credentials is possible. In this setting, you already unlock
serverless execution for your Python ML code, but you cannot take advantage of advanced usage patterns that are unlocked through compute saving and reuse.

For production settings, we recommend that users load cloud secrets, Kubeconfig, and available compute resources into Runhouse Den and authenticate from
all launch environments using only the Runhouse token. Platform teams gain centralized observability over utilization, including insights into who is launching clusters,
how often they are launched, and the resources or tasks executed on them. Access management becomes much simpler, especially in multi-cloud or multi-cluster environments.
To get started with Den enabled, simply run ``runhouse login --sync-secrets`` in the CLI.

Start Your Project
-------------------
Once you have established access to compute, you can start developing a new ML project. The following steps will provide the details of how to use Runhouse, starting
from a blank page in your IDE.

1. Define Compute
^^^^^^^^^^^^^^^^^
Runhouse allows you to define compute requirements in code, and launch ephemeral clusters from the compute pool we described in the prior section.
Here, you can define the required CPU, GPU, memory, and disk requirements (or name a specific cluster) to use.

.. image:: https://runhouse-tutorials.s3.amazonaws.com/Pull+Compute+from+Compute+Pool.jpg
  :alt: Runhouse pulls compute from a pool of resources
  :width: 750

For instance, to create a cluster on AWS with an A10 GPU attached using an arbitrary Docker image, you can write:

.. code:: python

    import runhouse as rh

    cluster = rh.ondemand_cluster(
        name="rh-cluster", # This cluster can be saved and reused by name. We will prefix your username when saved, e.g. /my_username/rh-cluster
        instance_type="A10G:1", # There are a number of options available for instance_type, check out the docs to see them all
        provider="aws", # Specify a cloud provider
        image_id="docker:nvcr.io/nvidia/pytorch:23.10-py3", # Use a Docker image
        autostop_mins=90, # Remember to set autostop_mins to avoid leaving clusters running indefinitely.
        launcher="den", # Launch the cluster with Runhouse; use 'local' for local credentials
    ).up_if_not()

You can run CLI commands on the cluster using ``cluster.run()`` to layer on setup steps beyond the underlying image; for instance, installing other packages.

.. code:: python

    cluster.run(['pip install numpy'])

You can find full documentation about the Runhouse cluster API `in the Cluster docs <https://www.run.house/docs/tutorials/api-clusters>`_.

1a. Starting the Runhouse Server Daemon
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
If not already running, the client will start the Runhouse API server daemon
on the compute and form a secure network connection (either over SSH or HTTP/S).

* The daemon can be thought of as a "Python object server," holding key-value pairs of names and Python objects in memory (objects you will dispatch to it in the next step), and exposing an HTTP API to call methods on those objects by name.
* By default, objects are held in a single default worker process but can be sent to other worker processes, including on other nodes in the cluster, to achieve powerful parallelism out of the box.
* When the object is used, and there is a ``GET http://myserver:32300/my_object/my_method``, the daemon will look up the object named "my_object," issue an instruction for its worker to call the method "my_method" on it, and return the result.
* The HTTP server and workers can handle thousands of concurrent calls per second, and have similar latency to Flask under most conditions.

2. Dispatch Your Code
^^^^^^^^^^^^^^^^^^^^^^
Once you have established a connection to compute, the development pattern is to continuously dispatch code to the cluster and execute it there.
You are doing local-like execution and debug, but with the power of the remote compute. Runhouse is agnostic to whether you dispatch
using a Notebook or run directly from a Python script.

Specifically to do the dispatch, you wrap your local function with ``rh.function()`` or class with ``rh.module()``. For functions, you can call them directly
as if they were local functions. For modules, you instantiate a remote instance of the object which is stateful; you can access this remote object by name and make
multi-threaded calls to its methods.

For the function or class defined in the local code, that repository or package is rsynced to the cluster.
An instruction containing the import path is then sent to the cluster to construct the function or class in a specific worker, and it is upserted into the key-value store.
We avoid serializing code and strongly discourage it, as code serialization often leads to versioning mismatch errors between local and remote package versions.

After the object is deployed to the server, the Runhouse Python client returns a local callable stub which behaves like the original object but forwards method calls
over HTTP to the remote object on the cluster.

.. code:: python

      def add_two_numbers(a,b):
            return a+b

      remote_add = rh.function(add_two_numbers).to(cluster)

      class TorchTrainer:
         def __init__(self):
            ..

         def train(self, X, y):
            ..

         def test(self, X, y):
            ..

      if __name__ == "__main__":
         cluster.install_packages(["torch"])
         RemoteTrainer = rh.module(TorchTrainer).to(cluster) # Send to cluster
         trainer = RemoteTrainer(name='remote-instance-of-trainer') # Instantiate remote object

.. note::

      The code that should only run locally (e.g. defining compute, dispatch, and calling remote objects for execution)
      should live within a ``if __name__ == "__main__":`` block in a script. This way, the code will not execute on remote compute
      when it is sent there.

Read more about `functions and modules <https://www.run.house/docs/tutorials/api-modules>`_.

3. Execute Your Code Remotely
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
It's now possible to use your remote objects as if they were local. From here on, you can think of Runhouse as
facilitating regular object-oriented programming but with the objects living remotely, maybe in a different cluster, region, or cloud than the local code.
Python behavior like async, exceptions, printing, and logging are all preserved across remote calls but can be disabled or controlled if desired.

.. code:: python

      result = remote_add(1,2)
      print(result)

      X, y = ...  # Load data
      trainer.train(X,y)

As noted above, you should be iteratively dispatching and executing code. If you make updates to the ``add_two_numbers`` function or the ``TorchTrainer`` class, you can simply
re-run ``.to()``, and it should take <2 seconds to redeploy. The underlying cluster is persisted and stateful until you choose to down it, so you can take advantage
of the remote file system and memory during interactive development as well.

These remote objects are accessible from anywhere you are authenticated with Runhouse, so you and your team can make multi-threaded calls against them.
Calling microservices is actually a familiar pattern in programming; however, no team would ever manually split their ML pipeline into multiple applications due to the DevOps overhead.


.. image:: https://runhouse-tutorials.s3.amazonaws.com/Iterative+Dispatch+from+Notebook.jpg
  :alt: Iteratively develop and dispatch code to remote execution
  :width: 550

4. Saving and Loading
^^^^^^^^^^^^^^^^^^^^^
Runhouse resources (clusters, functions, modules) can be saved, shared, and reused based on a compact
JSON metadata signature. This allows for easy sharing of clusters and services across users and environments. For instance,
the team might want to use a single shared embeddings service to save costs and improve reproducibility.

Runhouse comes with a built-in metadata store / service registry called
`Den <https://www.run.house/dashboard>`_ to facilitate convenient saving, loading, sharing, and management of these
resources. Den can be accessed via an HTTP API or from any Python interpreter with a Runhouse token
(either in ``~/.rh/config.yaml`` or an ``RH_TOKEN`` environment variable):

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
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
When a remote object is no longer needed, it can be deallocated from
the remote compute by calling ``cluster.delete(obj_name)``. This will remove the object from the key-value store and
free up the memory on the worker. A worker process can similarly be terminated with ``cluster.delete(worker_name)``,
terminating its activities and freeing its memory.

To down a cluster when the task is complete and the resource is no longer needed, you can simply call ``cluster.teardown()``
or let the autostop handle the down.

.. code-block:: python

    cluster.teardown()

Moving to Production
----------------
A key advantage of using Runhouse is that the code developed locally has already been executing production-like on remote compute the entire time. This means
research-to-production is an abstract checkpoint in development rather than an actual task to rewrite pipelines for production over different hardware/data.

If your code is for a non-recurring task, then great, check your code into version control and you are already done. If you are deploying a recurring
job like recurring training, then simply move the Runhouse launching code into the orchestrator or scheduler of your choice. You should not
repackage ML code into orchestrator nodes and make orchestrators your runtime. Instead, you should use orchestrators as minimal systems to schedule and observe your jobs,
but the jobs themselves will continue to be executed serverlessly with Runhouse from each node. This saves considerable time upfront as setting up
the first orchestrator run less than an hour (compared to multiple weeks in traditional ML research-to-production).

As an example, you might want to make the first task of your orchestrator pipeline simply bringing up the cluster and
dispatching code to the new cluster. You can see that we are using the same underlying code (directly importing it from a source file), and then
reusing the object and cluster by name across steps.

.. code:: python

      @task()
      def up_and_dispatch():
            image = (
                rh.Image("base_setup")
                .from_docker("nvcr.io/nvidia/pytorch:23.10-py3")
                .install_packages(["torch"])
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

Runhouse recommends creating a Docker container which fixes the environment, dependencies, and program code for production pipelines.
There are significant benefits to containerization, rather than, for instance, worrying about new breaking changes from package
installation with PyPi. This is actually still unproblematic for additional future iteration or debug, since you still easily interactively layer on changes to the environment
from local, even when you launch with the container.

.. image:: https://runhouse-tutorials.s3.amazonaws.com/Identical+Dispatch+in+Production.jpg
  :alt: Send code from research and production to compute
  :width: 750

My Pipeline is in Production, What's Next?
----------------------
Once in production, your ML pipelines will eventually experience some failures you need to debug. With Runhouse engineers can easily reproduce production runs on local,
make changes to the underlying code, and simply push a change to the codebase. There is no debugging through the orchestrator, and no need to rebuild and resubmit.
However, we find that deploying with Runhouse has fewer errors to begin with, as the code has already been developed in a production-like environment.

This also makes production-to-research a seamless process. Many teams are loathe to revisit the research-to-production process again, so when code is deployed
to production, there is little appetite to make small incremental improvements to the pipeline. With Runhouse, the pipeline is already running serverlessly, so
incremental changes that are merged to the team codebase are automatically reflected in the production pipeline once tested via normal development processes.

There are other benefits to using Runhouse in production as you scale up usage. A few are included here:

* **Shared services**: You may want to deploy shared services like an embeddings endpoint, and have all pipelines call it by name as a live service *or* import the code
from the underlying team repository and stand it up separately in each pipeline. Either way, if you every update or improve this shared service,
all pipelines will receive the downstream updates without any changes to the pipeline code.
* **Compute abstraction**: As you add new resources to your pool, get credits from new clouds, or get new quota, if all users are using Runhouse to allocate
ephemeral compute, there is no need to update any code or configuration files at the user level. The new resources are added by the platform team, and then automatically
adopted by the full team.
* **Infrastructure Migrations**: With Runhouse, your application code is entirely undecorated Python and the dispatch happens to arbitrary compute. If you ever choose
to abandon your existing orchestrator, cloud provider, or any other tool, you simply have to move a small amount of dispatch code and infrastructure code configuration.
* **Adopting Distributed Frameworks**: Runhouse is a perfect complement to distributed frameworks, with some built-in abstractions that let you scale to multiple clusters
or start using Ray clusters easily.
