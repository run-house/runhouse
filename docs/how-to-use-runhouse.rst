Using Runhouse
==========================
This page will guide you can use Runhouse to develop and deploy your ML projects.

Detailed Start Guide to Using Runhouse
---------------------------------------

Quick Start
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Before reviewing this detailed guide, we recommend you start with the `Quick Start <https://www.run.house/docs/tutorials/quick-start-cloud>`_ guide.

* Install Runhouse with `pip install runhouse`
* Optionally install with a specific cloud like `pip install "runhouse[aws]"` or with SkyPilot for elastic compute `pip install "runhouse[sky]`
* Optionally create an account on the Runhouse website or with `runhouse login --sync-secrets` to enable saving, reloading, and centralized authentication / secrets management.

Access to a Compute Pool
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
In order to use Runhouse, users must be able to access compute resources, which can take any form (e.g. VMs, elastic compute, Kubernetes). You should
consider all the compute resources you have as a single pool, from which Runhouse allows you to launch ephemeral clusters to execute your code.

* Kubernetes: All you need is a Kubeconfig
* Elastic Compute: We use Skypilot under the hood to launch elastic compute, and support most clouds. You can run `sky check` after installing Runhouse to confirm you have access to the cloud.
* Existing Clusters: Runhouse supports a variety of authentication methods to access existing clusters, including SSH with keys or passwords.

For intitial projects and getting started quickly, launching from local credentials is possible. In this setting, you already unlock
serverless execution for your Python ML code, but you cannot take advantage of advanced usage patterns that are unlocked through compute saving and reuse.

For production settings, we recommend that users load cloud secrets, Kubeconfig, and available compute into Runhouse Den, and authenticate from all
launch environments using the Runhouse token only. Once credentials are stored in Den, permissions to launch ephemeral compute can be granted to ML
team members through Runhouse authentication. Platform teams gain centralized observability over utilization, including who is launching clusters,
how often clusters are launched, and what resources or tasks are executed on them. Access management also becomes much simpler, especially in multi-cloud
or multi-cluster settings, where keys and authentication can be managed centrally. To get started with Den enabled, simply run ``runhouse login --sync-secrets`` in the CLI.

Starting a Project
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Runhouse enables execution through three easy steps:

**1. Define Compute**: Runhouse allows you to define compute requirements in code, and launch ephemeral clusters from the compute pool we described in the prior section.
Here, you can define the required CPU, GPU, memory, and disk requirements (or name a specific cluster) to use. For instance, to create a cluster on AWS with
an A10 GPU attached using an Docker image, you can write:

.. code:: python

    import runhouse as rh
    cluster = rh.ondemand_cluster(
        name="rh-cluster",
        instance_type="A10G:1",
        provider="aws",
        image_id="docker:nvcr.io/nvidia/pytorch:23.10-py3",
    ).up_if_not()


You can easily run commands against the cluster using ``cluster.run()`` to layer on setup steps beyond the underlying image.

.. code:: python

    cluster.run(['pip install numpy'])

You can find full documentation about the Runhouse cluster API `in the Cluster docs <https://www.run.house/docs/tutorials/api-clusters>`_.

**2. Dispatch Your Code**: You can dispatch functions and classes to Runhouse, by wrapping with ``rh.function()`` or ``rh.module()``. For functions, you can call them directly
as if they were local functions. For modules, you instantiate a remote instance of the object; you can access this remote object by name and make
multi-threaded calls to its methods.

.. code:: python

      def add_two_numbers(a,b):
            return a+b

      remote_add = rh.function(add_two_numbers).to(cluster)

.. code:: python

      class TorchTrainer:
         def __init__(self):
            ..

         def train(self, X, y):
            ..

         def test(self, X, y):
            ..

      my_env = rh.env(reqs=["torch"], name="my-env") # Define the need for PyTorch
      RemoteTrainer = rh.module(TorchTrainer).to(cluster, env=my_env) # Send to cluster
      trainer = RemoteTrainer(name='remote-instance-of-trainer') # Instantiate remote object

Read more about `functions and modules <https://www.run.house/docs/tutorials/api-modules>`_.

**3. Execute Your Code Remotely**: It's now possible to use your remote objects as if they were local.

.. code:: python

      result = remote_add(1,2)
      print(result)
      X, y = ...  # Load data
      trainer.train(X,y)

In development, you should be iteratively dispatching and executing code. If you make updates to the ``add_two_numbers`` function or the ``TorchTrainer`` class, you can simply
re-run ``.to()``, and it should take <2 seconds to redeploy. The underlying cluster is persisted and stateful until you choose to down it, so you can take advantage
of the remote file system and memory during interactive development as well.

These remote objects are accessible from anywhere you are authenticated with Runhouse, so you and your team can make multi-threaded calls against them. Runhouse essentially
has automatically turned this BERT embedding class into a remote service (with the latency of a FastAPI app).

.. note::

   As a practical note, make sure that any code in your Python file that’s meant to only run
   locally (such as creating a cluster, dispatching code, or calling remote code) is placed within a ``if __name__ == "__main__":`` block.
   Otherwise, that code will run when Runhouse attempts to import your
   code remotely. For example, you wouldn’t want ``function.to(cluster)`` to run again on the cluster. This is not necessary when using a notebook. Please see our `examples
   directory <https://github.com/run-house/runhouse/tree/main/examples>`__ for implementation details.

Moving to Production
^^^^^^^^^^^^^^^^^^^^^
A key advantage of using Runhouse is that the code developed locally has already been executing production-like on remote compute the entire time. This means
research-to-production is a abstract checkpoint in development rather than an actual task to rewrite pipelines for production over different hardware/data.

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
            cluster = rh.ondemand_cluster(
                  name="rh-cluster",
                  instance_type="A10G:1",
                  provider="aws",
                  image_id="docker:nvcr.io/nvidia/pytorch:23.10-py3",
            ).up_if_not()

            from my_code import TorchTrainer
            my_env = rh.env(reqs=["torch"], name="my-env")
            RemoteTrainer = rh.module(TorchTrainer).to(cluster, env=my_env)
            trainer = RemoteTrainer(name='remote-instance-of-trainer')

      @task()
      def embed():
            cluster = rh.cluster(name="rh-cluster")
            trainer = cluster.get(name='remote-instance-of-trainer')
            X, y = ...  # Load data
            trainer.train(X,y)

For production, Runhouse does recommend creating a Docker container which fixes the environment, dependencies, and program code. While
in development, the ability to interactively alter the remote environment is useful, in production, there are significant benefits to
containerization, rather than, for instance, worrying about new breaking changes from package installation with PyPi. This is actually
still unproblematic for additional future iteration or debug, since you can easily interactively layer on changes to the environment
from local, even when you launch with the container.

In Production, for the Long Term
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
In the long run, debugging failures and making updates to the pipeline is also extremely easy, as engineers can easily reproduce production runs on local,
make changes to the underlying code, and simply push to the codebase. This means that debugging pipelines

This also makes production-to-research a seamless process. Many teams are loathe to revisit the research-to-production process again, and so when code is deployed
to production, there is little appetite to make small incremental improvements to the pipeline. With Runhouse, the pipeline is already running serverlessly, and so
incremental changes that are merged to the team codebase are automatically reflected in the production pipeline once tested via normal development processes.

There are other benefits to using Runhouse in production as you scale up usage. A few are included here:

* **Shared services**: You may want to deploy shared services like an embeddings endpoint, and have all pipelines call it by name as a live service *or* import the code
from the underlying team repository and stand it up separately in each pipeline. Either way, if you every update or improve this shared service,
all pipelines will receive the downstream updates without any changes to the pipeline code.
* **Compute abstraction**: As you add new resources to your pool, get credits from new clouds, or get new quota, if all users are using Runhouse to allocate
ephemeral compute, there is no need to update any code or configuration files at the user level. The new resources are added by the platform team, and then automatically
adopted by the full team.

Under the Hood: Details about the Runhouse API
-------------------------------------------------------
Where the above describes the usage flow of Runhouse, this section is intended to provide interested users with the technical details of
how Runhouse offloads function and classes as services. Understanding this section is not necessary to use Runhouse, but it can help users
who want to better understand what is happening under the hood. If you have any questions about what is described here, please reach out to
`hello@run.house <mailto:hello@run.house>`_ and we'd be happy to walk you through the details.

You can follow along with this annotated code snippet:

.. code-block:: python

    import runhouse as rh

    # [1] and [2]
    gpu = rh.cluster(name="rh-a10x", instance_type="A10G:1", provider="aws").up_if_not()

    # [3]
    sd_worker = rh.env(reqs=["torch", "transformers", "diffusers"], name="sd_generate")
    remote_sd_generate = rh.function(sd_generate).to(gpu, env=sd_worker)

    # [4]
    imgs = remote_sd_generate("A hot dog made out of matcha.")
    imgs[0].show()

    # [5]
    remote_sd_generate.save()
    sd_upsampler = rh.function(name="/my_username/sd_upsampler")
    high_res_imgs = sd_upsampler(imgs)

    # [6]
    gpu.teardown()


1. Specify and/or Allocate Compute
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    gpu = rh.cluster(name="rh-a10x", instance_type="A10G:1", provider="aws").up_if_not()

Runhouse can allocate compute to the application on the fly, either by
utilizing an existing VM or Ray cluster, or allocating a new one using local cloud or K8s credentials. The
``rh.cluster`` constructor is generally used to specify and interact with remote compute.

You can bring up the cluster using ``cluster.up_if_not()`` or check if it is up using ``cluster.is_up()``.

2. Starting the Runhouse Server Daemon
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
If not already running, the client will start the Runhouse API server daemon
on the compute and form a secure network connection (either over SSH or HTTP/S). Dependencies can be specified to be
installed before starting the daemon.

#. The daemon can be thought of as a "Python object server", holding key-value pairs of names and Python
   objects in memory, and exposing an HTTP API to call methods on those objects by name.
#. The objects are held in a single default worker process by default but can be sent to other worker
   processes, including on other nodes in the cluster, to achieve powerful parallelism out of the box.
#. If I call GET http://myserver:32300/my_object/my_method, the daemon will look up the object named
   "my_object", issue an instruction for its worker to call the method "my_method" on it, and
   return the result.
#. The HTTP server and workers can handle thousands of concurrent calls per second, and have similar latency
   under simple conditions to Flask.
#. New workers can be constructed with ``rh.env``, which specifies the details of the Python environment
   (packages, environment variables) in which the process will be constructed. By default, workers live
   in the same Python environment as the daemon but can also be started in a conda environment or a
   separate node. To configure the environment of the daemon itself, such as setting environment variables
   or installing dependencies which will apply across all workers by default, you can pass an ``rh.env`` to the
   ``default_env`` argument of the ``rh.cluster`` constructor.

3. Deploying Functions or Classes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: python

    sd_worker = rh.env(reqs=["torch", "transformers", "diffusers"], name="sd_generate")
    remote_sd_generate = rh.function(sd_generate).to(gpu, env=sd_worker)

The user specifies a function or class to be deployed to the remote compute
using the ``rh.function`` or ``rh.module`` constructors (or by subclassing ``rh.Module``), and calling
``remote_obj = my_obj.to(my_cluster, env=my_env)``. The Runhouse client library extracts the path, module name,
and importable name from the function or class. If the function or class is defined in local code, the repo or
package is rsynced onto the cluster. An instruction with the import path is sent to the cluster to
construct the function or class in a particular worker and upserts it into the key-value store.

4. Calling the Function or Class
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: python

    imgs = remote_sd_generate("A hot dog made out of matcha.")
    imgs[0].show()

After deploying the function, class, or object into the server, the Runhouse
Python client returns a local callable stub which behaves like the original object but forwards method calls
over HTTP to the remote object on the cluster.

#. If a stateful instance of a class is desired, an ``__init__`` method can be called on the remote class to
   instantiate a new remote object from the class and assign it a name.
#. If arguments are passed to the method, they're serialized with cloudpickle and sent with the HTTP request.
   Serializing code, such as functions, classes, or dataclasses, is strongly discouraged, as it can lead to
   versioning mismatch errors between local and remote package versions.
#. From here on, you can think of Runhouse as facilitating
   regular object-oriented programming but with the objects living remotely, maybe in a different cluster,
   region, or cloud than the local code.
#. Python behavior like async, exceptions, printing, and logging are all preserved across remote calls but
   can be disabled or controlled if desired.

This is a common pattern - calling a function or class as a remote service just a microservice.
However, doing it manually divides the code into multiple applications, multiplying the DevOps overhead, as each requires its own configuration,
automation, scaling, etc. Runhouse combines the best of both approaches: providing limitless compute dynamism and
flexibility in Python without disrupting the runtime or fragmenting the application, by offloading functions and classes to remote compute as services on the fly.

5. Saving and Loading
^^^^^^^^^^^^^^^^^^^^^
Runhouse resources (clusters, functions, modules, environments) can be saved, shared, and reused based on a compact
JSON metadata signature. This allows for easy sharing of clusters and services across users and environments. For instance,
the team might want to use a single shared embeddings service to save costs and improve reproducibility.

Runhouse comes with a built-in metadata store / service registry called
`Den <https://www.run.house/dashboard>`__ to facilitate convenient saving, loading, sharing, and management of these
resources. Den can be accessed via an HTTP API or from any Python interpreter with a Runhouse token
(either in ``~/.rh/config.yaml`` or an ``RH_TOKEN`` environment variable) like so:

.. code-block:: python

    import runhouse as rh

    remote_func = rh.function(fn=my_func).to(my_cluster, env=my_env, name="my_function")

    # Save to Den
    remote_func.save()

    # Reload the function and invoke it remotely on the cluster
    remote_func = rh.function(name="/my_username/my_function")
    res = remote_func(*args, **kwargs)

    # Share the function with another user, giving them access to call or modify the resource
    remote_func.share("user_a@gmail.com", access_level="write")

You can access the metadata directly by calling ``resource.config()`` and reconstruct the resource with
``<Resource Type>.from_config(config)``.


6. Terminating Modules, Workers, or Clusters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: python

    gpu.teardown()

When a remote object is no longer needed, it can be deallocated from
the remote compute by calling ``cluster.delete(obj_name)``. This will remove the object from the key-value store and
free up the memory on the worker. A worker process can similarly be terminated with ``cluster.delete(worker_name)``,
terminating its activities and freeing its memory. An on-demand cluster can be terminated with ``cluster.teardown()``,
or by setting its ``autostop_mins``, which will auto-terminate it after a period of inactivity.
