Run
====================================

A Run is a Runhouse primitive used for capturing logs, inputs, results within a particular function call,
CLI or Python command, or context manager.
Runs also serve as a convenient way to trace the usage and upstream / downstream dependencies between different
Runhouse artifacts.


Run Factory Method
~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: runhouse.run

Run Class
~~~~~~~~~~~~~~

.. autoclass:: runhouse.Run
   :members:
   :exclude-members:

    .. automethod:: __init__


Use Cases
~~~~~~~~~
Runs can be useful in a number of different ways:

- **Caching**: Using Runs, we can easily trace the usage and dependencies between various
  resources. For example, we may have a pipeline that produces a pre-preprocessed dataset, trains a model, and exports
  the model for inference. It would be useful to know which functions (or microservices) were used to produce each
  stage of the pipeline, and which data artifacts were created along the way. Caching also works among a team, so
  when one person on a team creates a Run, others can benefit from the cached result (without needing to check
  whether the result is present yet).


- **Auditability**: With Runs we can easily inspect inputs, logs and outputs for a particular function call or
  command execution. This makes it much easier to debug or reproduce a particular workflow.


- **Sharing**: Runs (like all other Runhouse objects) can easily be shared among team members. This is useful when we have
  different services that are dependent on a single output (e.g. a model).


- **Reusability**: Runs make it much easier to reproduce or re-run previous workflows. For example, if we need to run
  some script on a recurring basis, we don't have to worry about re-running each step in its entirety if we have already
  have a cached run for that step.


Run Components
~~~~~~~~~~~~~~
A Run may contain some (or all) of these core components:

- **Name**: A unique identifier for the Run.

- **Folder**: Where the Run's data lives on its associated system.

- **Function**: A function to be executed on a cluster.

- **Commands**: A record of Python or CLI commands executed on a cluster.

- **Upstream dependencies**: Runhouse artifacts loaded by the Run.

- **Downstream dependencies**: Runhouse artifacts saved by the Run.

.. note::
    Artifacts represent any Runhouse primitive (e.g. :ref:`Blob`, :ref:`Function`, :ref:`Table`, etc.) that is
    loaded or saved by the Run.


Run Data
~~~~~~~~
When a Run is completed it stores the following information in its dedicated folder on the system:

- **Config**: A JSON of the Run's core data. This includes fields such as: :code:`name`, :code:`status`,
  :code:`start_time`, :code:`end_time`, :code:`run_type`, :code:`fn_name`, :code:`cmds`, :code:`upstream_artifacts`,
  and :code:`downstream_artifacts`.
  Note: When saving the Run to Runhouse Den (RNS) (via the :code:`save()` method), the Run's metadata that will be
  stored.

- **Function inputs**: The pickled inputs to the function that triggered the Run (where relevant).

- **Function result**: The pickled result of the function that triggered the Run (where relevant).

- **Stdout**: Stdout from the Run's execution.

- **Stderr**: Stderr from the Run's execution.

.. note::
    By default, the Run's folder sits in the :code:`~/.rh/logs/<run_name>` directory if the system is a cluster,
    or in the :code:`rh` folder of the working directory if the Run is local. See :ref:`Run Logs` for more details.


Creating a Run
~~~~~~~~~~~~~~~
There are three ways to create a Run:

Calling a Function
------------------
We can create a Run when executing a function by providing the :code:`run_name` argument, a string
representing the custom name to assign the Run.
By default :code:`run_name` is set to :code:`None`.

.. code-block:: python

    import runhouse as rh

    def summer(a, b):
        return a + b

    # Initialize the cluster object (and provision the cluster if it does not already exist)
    cpu = rh.cluster(name="^rh-cpu")

    # Create a function object and send it to the cpu cluster
    my_func = rh.function(summer, name="my_test_func").to(cpu)

    # Call the function with its input args, and provide it with a `run_name` argument
    fn_res = my_func(1, 2, run_name="my_fn_run")


When this function :code:`my_func` is called, Runhouse triggers the function execution on the cluster
and returns the Run's result. The Run's config metadata, inputs, and stdout / stderr are also stored in the .rh folder
of the cluster's file system.

.. note::

    In addition to calling the function directly, you can also
    use :code:`.run()` :ref:`async interface <Asynchronous Run>`.

Running Commands
--------------------------------
Another way to create a Run is by executing Python or CLI command(s).
When we run these commands, the stdout and stderr from the command will be saved in the Run's dedicated
folder on the cluster.

To create a Run by executing a CLI command:

.. code-block:: python

    # Run a CLI command on the cluster and provide the `run_name` param to trigger a run
    return_codes = cpu.run(["python --version"], run_name="my_cli_run")

To create a Run by executing Python commands:

.. code-block:: python

     return_codes = cpu.run_python(
        [
            "import pickle",
            "import logging",
            "local_blob = rh.blob(name='local_blob', data=pickle.dumps(list(range(50))), mkdir=True).write()",
            "logging.info(f'Blob path: {local_blob.path}')",
            "local_blob.rm()",
        ],
        run_name="my_cli_run",
    )


Advanced API Usage
------------------

Loading Runs
~~~~~~~~~~~~
There are a few ways to load a Run:

- **From a cluster**: Load a Run that lives on a cluster by using :code:`cluster.get_run()`.
  This method takes a :code:`run_name` argument with the name of the Run to load.

- **From a system**: Load a Run from any system by using the the :code:`rh.run()` method. This method takes either
  a :code:`path` argument specifying the path to the Run's folder, or a :code:`name` argument specifying the
  name of the Run to load.

.. note::
    Each of these methods will return a :ref:`Run` object.

**Loading Results**

To load the results of the Run, we can call the :code:`result()` method:

.. code-block:: python

    # Load the run from its cluster
    fn_run = cpu.get_run("my_fn_run")

    # If the function for this run has finished executing, we can load the result:
    result = pickle.loads(fn_run.result())

.. tip::
    We can also call :code:`fn_run.stderr()` to view the Run's stderr output, and :code:`fn_run.stdout()`
    to view the Run's stdout output.


Accessing and Using Runs
~~~~~~~~~~~~~~~~~~~~~~~~
We can trace activity and capture logs within a block of code using a context manager. By default the Run's config
will be stored on the local filesystem in the :code:`rh/<run_name>` folder of the current working directory,
or the :code:`.rh/logs/<run_name>` folder if running on a cluster.

.. code-block:: python

    import runhouse as rh

    with rh.run(name="ctx_run") as r:
        # Add all Runhouse objects loaded or saved in the context manager to
        # the Run's artifact registry (upstream + downstream artifacts)

        my_func = rh.function(name="my_existing_run")
        my_func.save("my_new_func")

        my_func(1, 2, run_name="my_new_run")

        current_run = my_func.system.get_run("my_new_run")
        run_res = pickle.loads(current_run.result())
        print(f"Run result: {run_res}")

    r.save()

    print(f"Saved Run to path: {r.path}")


.. note::
    We can specify the path to the Run's folder by providing the :code:`path` argument to the :code:`Run` object.
    If we do not specify a path the folder will be created in the :code:`rh` folder of the current working directory.


We can then re-load this Run from its system (in this case the local file system):

.. code-block:: python

    import runhouse as rh

    ctx_run = rh.run(path="~/rh/runs/my_ctx_run")

Caching
~~~~~~~
Runhouse provides varying levels of control for running and caching the results of a Run.

We can invoke a run both synchronously and asynchronously, and with or without caching:

Synchronous Run
---------------

To create a Run which executes a function synchronously without any caching, we call the function and
provide the :code:`run_name` argument. The function will be executed on the cluster, and will
return its result once completed.

For a fully synchronous run which also checks for a cached result, we can call the :code:`get_or_call()` method
on the function. If a result already exists with this Run name, the result will be returned.
Otherwise, the function will be executed synchronously on the cluster and the result will be returned once
the function execution is complete:

.. code-block:: python

    import runhouse as rh

    my_func = rh.function(name="my_func")
    res = my_func.get_or_call(1, 2, run_name="my_fn_run")


Asynchronous Run
----------------
To run a function asynchronously without any caching, we can call the :code:`run()` method. The function will
begin executing on the cluster in the background, and in the meantime a :code:`Run` object will be returned:

.. code-block:: python

    import runhouse as rh

    my_func = rh.function(name="my_func")
    run_obj = my_func.run(1, 2, run_name="my_async_run")


For a fully asynchronous run which also checks for a cached result, we can call the :code:`get_or_run()` method
on the function. A :code:`Run` object will be returned whether the result is cached or not:

.. code-block:: python

    import runhouse as rh

    my_func = rh.function(name="my_func")
    run_obj = my_func.get_or_run(1, 2, run_name="my_async_run")



Moving Runs Between Systems
~~~~~~~~~~~~~~~~~~~~~~~~~~~

To copy the Run's folder contents from the cluster to the local environment:

.. code-block:: python

    my_run = cpu.get_run("my_run")
    local_run = my_run.to("here")

By default, this will be copied in to the :code:`rh` directory in your current projects working directory, but
you can overwrite this by passing in a :code:`path` parameter.


To copy the Run's folder contents to a remote storage bucket:

.. code-block:: python

    my_run = cpu.get_run("my_run")
    my_run_on_s3 = my_run.to("s3", path="/s3-bucket/s3-folder")

Run Logs
~~~~~~~~

You can view a Run's logs in python:

.. code-block:: python

    cpu.get("my_run", stream_logs=True)

Or via the command line:

.. code-block:: cli

    runhouse logs cpu my_run

Cancelling a Run
~~~~~~~~~~~~~~~~

For cancelling a Run:

.. code-block:: python

    cpu.cancel("my_run")

Or via the command line:

.. code-block:: cli

    runhouse cancel cpu my_run
