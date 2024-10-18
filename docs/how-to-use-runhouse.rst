Using Runhouse
==========================
This page will guide you through how Runhouse fit

Setup
------------------

Starting a Project
------------------

Moving to Production
--------------------

Maintenance and Debug
---------------------



Detailed Flow of a Hello World Example
---------------------------------------
The technical details of how Runhouse offloads function and classes as services is as follows. You can follow along with this
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
