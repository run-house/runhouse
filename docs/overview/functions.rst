Functions
====================================

Runhouse allows you to function code a cluster, but still interact with it as a native runnable :ref:`Function` object
(see `tutorial 01 <https://github.com/run-house/tutorials/tree/main/t01_Stable_Diffusion/>`_).
When you do this, the following steps occur:

1. We check if the cluster is up, and bring up the cluster if not (only possible for OnDemandClusters)
2. We check that the cluster's gRPC server has started to handle requests to do things like install packages, run modules, get previously executed results, etc. If it hasn't, we install Runhouse on the cluster and start the gRPC server. The gRPC server initializes Ray.
3. We collect the dependencies from the :code:`reqs` parameter and install them on the cluster via :code:`cluster.install_packages()`. By default, we'll sync over the working git repo and install its :code:`requirements.txt` if it has one.


When you run your function module, we send a gRPC request to the cluster with the module name and function entrypoint to run.
The gRPC server adds the module to its python path, imports the module, grabs the function entrypoint, runs it,
and returns your results.

You can stream in logs from the cluster as your module runs by passing :code:`stream_logs=True` into your call line:


.. code-block:: python

    images = generate_gpu('A dog.', num_images=1, steps=50, stream_logs=True)


We plan to support additional form factors for modules beyond "remote Python function" shortly, including HTTP endpoints, custom ASGIs, and more.


Advanced Function Usage
~~~~~~~~~~~~~~~~~~~
There are a number of ways to call a Function beyond just :code:`__call__`.

:code:`.remote` will call the function async (using Ray) and return a reference (`Ray ObjectRef <https://docs.ray.io/en/latest/ray-core/objects.html>`_)
to the object on the cluster. You can pass the ref into another function and it will be automatically
dereferenced once on the cluster. This is a convenient way to avoid passing large objects back and forth to your
laptop, or to run longer execution in notebooks without locking up the kernel.

.. code-block:: python

    images_ref = generate_gpu.remote('A dog.', num_images=1, steps=50)
    images = rh.get(images_ref)
    # or
    my_other_function(images_ref)


:code:`.enqueue` will queue up your function call on the cluster to make sure it doesn't run simultaneously with other
calls, but will wait until the execution completes.

:code:`.map` and :code:`.starmap` are easy way to parallelize your function (again using Ray on the cluster).

.. code-block:: python

    generate_gpu.map(['A dog.', 'A cat.', 'A biscuit.'], num_images=[1]*3, steps=[50]*3)


will run the function on each of the three prompts, and return a list of the results.
Note that the num_images and steps arguments are broadcasted to each prompt, so the first prompt will get 1 image.


.. code-block:: python

    generate_gpu.starmap([('A dog.', 1), ('A cat.', 2), ('A biscuit.', 3)], steps=50)

is the same as map as above, but we can pass the arguments as a list of tuples, and the steps argument as a single value, since it's the same for all three prompts.
