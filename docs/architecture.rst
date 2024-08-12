Architecture Overview
=====================

Runhouse is a Python library that allows any application to flexibly and powerfully utilize remote compute
infrastructure by deploying and calling remote services on the fly. It is principally designed for Machine
Learning-style workloads (online, offline, training, and inference), where the need for heterogeneous
remote compute is frequent and flexibility is paramount to minimize costs.

Incorporating heterogeneous compute into the runtime of an application, like workflow
orchestrators (Airflow, Prefect) or distributed libraries (Ray, Spark) do, is far more disruptive and less flexible at
every level (development workflow, debugging, DevOps, infra) than calling the heterogeneous portions as remote services.
Compare converting your Python application into an Airflow DAG to run the training portion on a GPU, vs.
making an HTTP call within the application to the training function running as a service on a GPU.
Calling a function or class as a remote service is a common pattern (e.g. microservices, Temporal)
but divides the code into multiple applications. This multiplies the DevOps overhead, each having their own
configuration, automation, scaling, etc. Runhouse achieves the best of both approaches: limitless compute dynamism and
flexibility in Python without disrupting the runtime or cleaving the application, by offloading
functions and classes to remote compute as services on the fly.

Why?
----

This solves a few major problems for AI teams:

#. **Cost**: Runhouse introduces the flexibility to allocate compute only while needed, right-size instances based on
   the size of the workload, work across multiple regions or clouds for lower costs, and share compute and services
   across tasks. Users typically see cost savings on the order of 50-75%, depending on the workload.
#. **Development at scale**: Powerful hardware such as GPUs or distributed clusters (Spark, Ray) can be hugely
   disruptive, requiring all development, debugging, automation, and deployment to occur on their runtime. Ray, Spark,
   or PyTorch distributed users for example must be tunneled into the head node at all times for development, leading
   to a proliferation of hosted notebook services as a stop-gap. Runhouse allows Python to orchestrate to these
   systems remotely, returning the development workflow and operations to standard Python. Teams using Runhouse
   can abandon hosted development notebooks and sandboxes entirely, again saving considerable cost and
   research-to-production time.
#. **Infrastructure overhead**: Runhouse thoughtfully captures infrastructure concerns in code, providing a clear
   contract between the application and infrastructure, and saving ML teams from learning all the infra, networking,
   security, and DevOps underneath.

High-level Flow
---------------

The basic flow of how Runhouse offloads function and classes as services is as follows. You can follow along with this
annotated code snippet:

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
``rh.cluster`` constructor is generally used to specify and interact with remote compute, including bringing it up
if necessary (``cluster.up_if_not()``).

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

5. Saving and Loading
^^^^^^^^^^^^^^^^^^^^^
.. code-block:: python

    remote_sd_generate.save()
    sd_upsampler = rh.function(name="/my_username/sd_upsampler")
    high_res_imgs = sd_upsampler(imgs)

The Runhouse client can save and load objects to and from the local filesystem, or to a
remote metadata store. This allows for easy sharing of clusters and services across users and environments,
and for versioning and rollback of objects. The metadata store can be accessed from any Python interpreter,
and is backed by UIs and APIs to view, monitor, and manage all resources.

6. Terminating Modules, Workers, or Clusters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: python

    gpu.teardown()

When a remote object is no longer needed, it can be deallocated from
the remote compute by calling ``cluster.delete(obj_name)``. This will remove the object from the key-value store and
free up the memory on the worker. A worker process can similarly be terminated with ``cluster.delete(worker_name)``,
terminating its activities and freeing its memory. An on-demand cluster can be terminated with ``cluster.teardown()``,
or by setting its ``autostop_mins``, which will auto-terminate it after a period of inactivity.

Comparing to other systems
--------------------------

Runhouse's APIs bear similarity to other systems, so it's helpful to compare and contrast. In many cases,
Runhouse is not a replacement for these systems but rather a complement or extension. In others, you may be able
to replace your usage of the other system entirely with Runhouse.

Distributed frameworks (e.g. Ray, Spark, Elixr)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Distributed frameworks make it possible to offload execution onto separate
compute, like a different process or node within a their cluster runtime. Runhouse
can be seen as similar but with the crucial distinction of dispatching execution to compute *outside* of its own
runtime (which is just Python) or orchestrating *between* clusters (even of different types).
For this reason, it has no other runtime to setup than Python itself, can be used to orchestrate your distributed code so you
can use your Ray or Spark clusters less disruptively within your stack (e.g. sending a function which uses
Ray over to the head node of the Ray cluster, where the Ray will execute as usual).

This also fixes certain sharp edges with these systems to significantly reduce costs, such as the inability to use
more than one cluster in an application or sharing a cluster between multiple callers. Is also means the local and
remote compute are largely decoupled, with no shared runtime which will break if one disconnects or goes down.

Workflow orchestrators (e.g. Airflow, Prefect, Dagster, Flyte, Metaflow, Argo)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Workflow orchestrators can allocate heterogeneous compute
on the the fly but act as the runtime itself for the program and only support certain pre-defined and highly
constrained DAGs. By allocating services Runhouse allows for arbitrary control flow and utilization of remote
hardware, making Python itself the orchestrator.
For example, with Runhouse it's easy to allocate small compute to start a training but if the training fails due to OOM
restart it with a slightly larger box. Other compute flexibility like multi-region or multi-cloud which other
orchestrators struggle with are trivial for Runhouse.

Generally, workflow orchestrators are built to be good at monitoring, telemetry, fault-tolerance, and scheduling, so
we recommend using one strictly for those features and using Runhouse within your pipeline nodes for the heterogeneous
compute and remote execution. You can also save a lot of money by reusing compute across multiple nodes or reusing
services across multiple pipelines with Runhouse, which is generally not possible with workflow orchestrators.

Serverless frameworks (e.g. Modal, AWS Lambda)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Serverless frameworks allow for the allocation of services on the fly but within a well-defined sandbox, and not
strictly from within regular Python - they require specific pre-packaging or CLI launch
commands outside Python. Runhouse runs fully in a Python interpreter so it can extend the compute power of practically
any existing Python application, and allocates services inside your own compute, wherever that may be. We may even
support serverless systems as compute backends in the future.

Infrastructure in code (e.g. SkyPilot, Pulumi)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Infrastructure in code tools allocate compute on the fly but can't utilize it instantly
to offload execution within the application (though you could call a predefined script entrypoint or API
endpoint). Runhouse uses SkyPilot to allocate compute but is vertically integrated to be able
to perform allocation, (re)deployment, and management of a new service all in Python so the new compute can be used
instantly within the existing application. It also doesn't need to perform allocation to create new services -
it can use existing compute or static VMs.

GPU/Accelerator dispatch (e.g. PyTorch, Jax, Mojo)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
GPU/Accelerator dispatch systems give the ability to offload computation to a local GPU or
TPU. Runhouse does not have this capability but can offload a function or class to a remote instance with an
accelerator, which can then itself use libraries like PyTorch or Jax (and maybe one day Mojo) to use the accelerator.

Saving, Loading, and Sharing
----------------------------

Runhouse resources (clusters, functions, modules, environments) can be saved, shared, and reused based on a compact
JSON metadata signature. This allows for easy sharing of clusters and services across users and environments, which
can often lead to massive cost savings. Runhouse comes with a built-in metadata store / service registry called
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
