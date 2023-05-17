Run
====================================

A Run is a Runhouse primitive used for capturing logs, inputs, results, or other artifacts that are
loaded or saved within a particular function call, CLI or Python command, or context manager.
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

Run Components
~~~~~~~~~~~~~~
A Run may contain some (or all) of these core components:

- **Name**: A unique identifier for the Run.

- **System**: Where the Run lives (can be a cluster or local filesystem).

- **Folder**: Where the Run's data lives on the system.

- **Function**: A function to be executed on a cluster.

- **Commands**: Python or CLI commands to be executed on a cluster.

- **Upstream dependencies**: Runhouse artifacts loaded by the Run.

- **Downstream dependencies**: Runhouse artifacts saved by the Run.

.. note::
    Artifacts represent any Runhouse primitive (e.g. :ref:`Blob`, :ref:`Function`, :ref:`Table`, etc.) that is
    loaded or saved by the Run. This is useful for tracing the dependencies between Runhouse objects.


Run Data
~~~~~~~~
When a Run is completed it stores the following information in its dedicated folder on the system:

- **Config**: A JSON of the Run's metadata. This is also the data that will be stored on Runhouse servers if the Run is saved.

- **Function inputs**: The pickled inputs to the function that triggered the Run (where relevant).

- **Function result**: The pickled result of the function that triggered the Run (where relevant).

- **Stdout**: Stdout from the Run's execution.

- **Stderr**: Stderr from the Run's execution.

.. note::
    By default, the Run's folder sits in the :code:`~/.rh/logs/<run_name>` directory if the system is a cluster,
    or in the :code:`rh` folder of the working directory if the Run is local.


Creating a Run
~~~~~~~~~~~~~~~
There are three ways to create a Run:

Calling a Function
------------------
We can create a Run when executing a function by providing the :code:`name_run` argument. This can be either a string
representing the custom name to assign the Run, or a boolean :code:`True` to indicate this Run should be named.

.. code-block:: python

    import runhouse as rh

    def summer(a, b):
        return a + b

    # Initialize the cluster object (and provision the cluster if it does not already exist)
    cpu = rh.cluster("^rh-cpu").up_if_not()

    # Create a function object and send it to the cpu cluster
    my_func = rh.function(summer, name="my_test_func").to(cpu)

    # Call the function with its input args, and provide it with a `name_run` argument to trigger a run
    my_func(1, 2, name_run="my_fn_run")


When this function :code:`my_func` is called, Runhouse asynchronously triggers the function execution on the cluster,
and returns the Run's name which can be used to retrieve the results when finished.

In order to get the results of the Run, we can call the :code:`result()` method:

.. code-block:: python

    fn_run = cpu.get_run("my_fn_run")

    # If the function for this run has finished executing, we can load the result:
    result = fn_run.result()

.. tip::
    See :ref:`Viewing RPC Logs` for more info on how Runhouse stores logs on a cluster.

Running Commands
--------------------------------
Another way to create a Run is by executing Python or CLI command(s).
When we run these commands, the stdout and stderr from the command will be saved in the Run's dedicated
folder on the cluster.

To create a Run by executing a CLI command:

.. code-block:: python

    # Run a CLI command on the cluster and provide the `name_run` param to trigger a run
    return_codes = cpu.run(["python --version"], name_run="my_cli_run")

To create a Run by executing Python commands:

.. code-block:: python

     return_codes = cpu.run_python(
        [
            "import runhouse as rh",
            "cpu = rh.cluster('^rh-cpu')",
            "rh.cluster('^rh-cpu').save()",
        ],
        name_run="my_cli_run",
    )


Context Manager
---------------
We can trace activity and capture logs within a block of code using a context manager. By default the Run's config
will be stored on the local filesystem in the :code:`rh/<run_name>` folder of the current working directory.

.. code-block:: python

    import runhouse as rh

    with rh.run(name="my_ctx_mgr_run") as r:
        # Add all Runhouse objects loaded or saved in the context manager to
        # the Run's artifact registry (upstream + downstream artifacts)

        my_func = rh.Function.from_name(FUNC_NAME)
        my_func.save("my_new_func")

        my_func(1, 2, name_run="my_new_run")

        current_run = my_func.system.get_run("my_new_run")
        run_res = current_run.result()
        print(f"Run result: {run_res}")

    print(f"Saved Run with name: {r.name} to path: {r.folder.path}")


We can then load this Run from the local file system:

.. code-block:: python

    import runhouse as rh

    ctx_run = rh.Run.from_file(name="my_ctx_run")
    print(f"Loaded run from path: {ctx_run.folder.path}"})


Advanced API Usage
~~~~~~~~~~~~~~~~~~

To copy the Run's folder contents from the cluster to your local env:

.. code-block:: python

    import runhouse as rh

    cpu = rh.cluster("^rh-cpu")
    my_run = cpu.get_run("my_run")

    local_run = my_run.to("here")

By default, this will be copied in to the :code:`rh` directory in your current projects working directory, but
you can overwrite this by passing in a :code:`path` parameter.


To copy the Run's folder contents to a remote storage bucket:

.. code-block:: python

    import runhouse as rh

    cpu = rh.cluster("^rh-cpu")
    fn_run = cpu.get_run("my_run")

    my_run_on_s3 = my_run.to("s3", path="/s3-bucket/s3-folder")


To check if a run was already completed, and load results if so:

.. code-block:: python

    import runhouse as rh

    my_func = rh.Function.from_name("my_func")
    run_result = my_func.get(run_str="my_run")

If the run has completed, the result will be returned. If the Run is still running or has not yet been triggered,
the Run's name will be returned. If the Run failed, the stderr logs will be returned.

.. tip::
    To load the latest run associated with this function, you can also call: :code:`my_func.get("latest")`

To check if a run was already completed, and trigger a new run if not:

.. code-block:: python

    import runhouse as rh

    cpu = rh.cluster("^rh-cpu")
    my_func = cpu.get_run("my_func")

    my_run = my_func.get_or_run(run_name="my_new_run", a=1, b=2)


If the Run has completed, the result will be returned. If the Run is still running or has not yet been triggered,
the Run's name will be returned. If the Run failed, the stderr logs will be returned.

If a Run does not exist, a new one will be created. By default, the function will be run synchronously before returning
the result. We can also choose to execute the function asynchronously on the cluster by
setting :code:`run_async=True`, in which case the Run's name will be returned while execution continues on the cluster.
