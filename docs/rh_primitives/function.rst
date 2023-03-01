Function
====================================

A Function is a portable code block that can be sent to remote hardware to run as a subroutine or service.
It is comprised of the entrypoint, system (:ref:`Cluster`), and requirements necessary to run it.


Function Factory Method
~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: runhouse.function

Function Class
~~~~~~~~~~~~~~

.. autoclass:: runhouse.Function
   :members:
   :exclude-members:

    .. automethod:: __init__

Basic API Usage
~~~~~~~~~~~~~~~
To initialize a function, from a local function:

.. code-block:: python

   def local_function(arg1, arg2):  # standard local Python method
      ...

   # Runhouse Function, to be run on my_cluster Cluster
   my_function = rh.function(fn=local_function, system=my_cluster, reqs=['requirements.txt'])
   # or, equivalently
   my_function = rh.function(fn=local_function).to(my_cluster, reqs=['requirements.txt'])

To use the function, simply call it as you would the local function.

.. code-block:: python

   result = my_function(arg1, arg2)  # runs the function on my_cluster, and returns the result locally

To intialize a function, from an existing Github function:

.. code-block:: python

   my_github_function = rh.function(
                           fn='https://github.com/huggingface/diffusers/blob/v0.11.1/examples/dreambooth/train_dreambooth.py:main',
                           system=my_cluster,
                           reqs=['requirements.txt'],
                        )

To name the function, to be accessed by name later on, pass in the `name` argument to the factory method.

.. code-block:: python

   my_function = rh.function(fn=local_function, system=my_cluster, reqs=['requirements.txt'], name="my_function_name")

Advanced API Usage
~~~~~~~~~~~~~~~~~~
To stream logs, pass in ``stream_logs=True`` to the function call.

.. code-block:: python

   my_function(arg1, arg2, stream_logs=True)

``.remote()``: Call the function async (using Ray) and return a reference (Ray ObjectRef) to the object
on the cluster. This ref can be passed into another function, and will be automatically dereferenced
once on the cluster. This is a convenient way to avoid passing large objects back and forth to your
laptop, or to run longer execution in notebooks without locking up the kernel.

.. code-block:: python

   result_ref = my_function.remote(arg1, arg2)

   result = rh.get(result_ref)  # load the result to local
   my_other_function(result_ref)  # pass in ref to another function

``.enqueue()``: Queue up the function call on the cluster. This ensures it doesn’t run simultaneously with other calls, but will wait until the execution completes.

.. code-block:: python

   my_function.enqueue()

``.map()`` and ``.starmap()`` are easy way to parallelize a function (again using Ray on the cluster).
``.map`` maps a function over a list of arguments, while ``.starmap`` unpacks the elements of the iterable
while mapping.

.. code-block:: python

   # .map
   arg1_map = [1, 2]
   arg2_map = [2, 5]
   map_results = my_function.map(arg1=arg1_map, arg2=arg2_map)

   # .starmap
   starmap_args = [[1, 2], [1, 3], [1, 4]]
   starmap_results = my_function.starmap(starmap_args)
