Compute: Cluster, Functions, Env
================================

.. raw:: html

    <p><a href="https://colab.research.google.com/github/run-house/runhouse/blob/stable/docs/notebooks/compute.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a></p>


Runhouse has several abstractions to provide a seamless flow of code and
execution across local and remote compute. The abstractions are cloud
provider-agnostic, and provide living, reusable services.

The Cluster and Function APIs blur the line between program execution
and deployment.

The Env and Package APIs help to provide convenient dependency isolation
and management.

Install Runhouse
----------------

.. code:: python

    !pip install runhouse

.. code:: python

    import runhouse as rh

Optionally, to login to Runhouse to sync any secrets.

.. code:: python

    !runhouse login

Cluster
-------

Runhouse provides various APIs for interacting with remote clusters,
such as terminating an on-demand cloud cluster or running remote CLI or
Python commands from your local dev environment.

Initialize your Cluster
~~~~~~~~~~~~~~~~~~~~~~~

There are two types of supported cluster types:

1. Bring-your-own (BYO) Cluster: these are existing clusters that you
   already have up, and access through an IP address and SSH
   credentials.
2. On-demand Cluster associated with your cloud account (AWS, GCP,
   Azure, LambdaLabs). There are additional features for these clusters,
   such as cluster (auto) stop. Please refer to
   ``Installation and Setup Guide`` for instructions on first getting
   cloud credentials set up.

.. code:: python

    # BYO cluster
    cluster = rh.cluster(
                  name="cpu-cluster",
                  ips=['<ip of the cluster>'],
                  ssh_creds={'ssh_user': '<user>', 'ssh_private_key':'<path_to_key>'},
              )

    # Using a Cloud provider
    cluster = rh.cluster(
                  name="cpu-cluster",
                  instance_type="CPU:8",
                  provider="cheapest",       # "AWS", "GCP", "Azure", "Lambda", or "cheapest" (default)
                  autostop_mins=60,          # Optional, defaults to default_autostop_mins; -1 suspends autostop
              )
    # Launch the cluster, only supported for on-demand clusters
    cluster.up()

The ``name`` parameter provided is used for saving down or loading
previous saved clusters. It is also used for various CLI commands for
the cluster.

To ssh into the cluster:

.. code:: python

    # Python
    cluster.ssh()

    # CLI
    !ssh cpu-cluster

To tunnel a JupyterLab server into your local browser:

.. code:: python

    # Python
    cluster.notebook()

    # CLI
    !runhouse notebook cpu-cluster

To run CLI or Python commands on the cluster:

.. code:: python

    cluster.run(['pip install numpy && pip freeze | grep numpy'])


.. parsed-literal::

    INFO | 2023-05-06 20:52:13,632 | Running command on cpu-cluster: pip install numpy && pip freeze | grep numpy


.. parsed-literal::

    Requirement already satisfied: numpy in /opt/conda/lib/python3.10/site-packages (1.24.3)
    numpy==1.24.3


.. parsed-literal::

    [(0,
      'Requirement already satisfied: numpy in /opt/conda/lib/python3.10/site-packages (1.24.3)\nnumpy==1.24.3\n',
      "Warning: Permanently added '3.95.164.76' (ECDSA) to the list of known hosts.\r\n")]



.. code:: python

    cluster.run_python(['import numpy', 'print(numpy.__version__)'])


.. parsed-literal::

    INFO | 2023-05-06 20:52:27,945 | Running command on cpu-cluster: python3 -c "import numpy; print(numpy.__version__)"
    1.24.3



.. parsed-literal::

    [(0, '1.24.3\n', '')]



Function
--------

Runhouse’s Function API lets you define functions to be run on remote
hardware. Simply pass in a local (or a GitHub) function, the intended
remote hardware, and any dependencies; Runhouse will handle the rest for
you.

Basic Functions
~~~~~~~~~~~~~~~

Let’s start with a simple local function ``getpid``, which takes in an
optional parameter ``a`` and returns the process ID plus ``a``.

.. code:: python

    # Local Function
    def getpid(a=0, b=0):
        import os
        return os.getpid() + a + b

To construct a function that runs ``getpid`` on a remote cluster, we
wrap it using ``rh.function``, and specify ``system=cluster``. There are
two ways of doing so:

.. code:: python

    # Remote Function
    getpid_remote = rh.function(fn=getpid, system=cluster)
    # or, equivalently
    getpid_remote = rh.function(fn=getpid).to(system=cluster)


.. parsed-literal::

    INFO | 2023-05-06 20:52:47,822 | Writing out function function to /content/getpid_fn.py. Please make sure the function does not rely on any local variables, including imports (which should be moved inside the function body).
    INFO | 2023-05-06 20:52:47,825 | Setting up Function on cluster.
    INFO | 2023-05-06 20:52:47,829 | Copying local package content to cluster <cpu-cluster>
    INFO | 2023-05-06 20:52:49,316 | Installing packages on cluster cpu-cluster: ['./']
    INFO | 2023-05-06 20:52:49,474 | Function setup complete.


To run the function, simply call it just as you would a local function,
and the function automatically runs on your specified hardware!

.. code:: python

    print(f"local: {getpid()}")
    print(f"remote: {getpid_remote()}")


.. parsed-literal::

    local: 163
    INFO | 2023-05-06 20:53:20,020 | Running getpid via gRPC
    INFO | 2023-05-06 20:53:20,152 | Time to send message: 0.12 seconds
    remote: 24056


Git Functions
~~~~~~~~~~~~~

A neat feature of Runhouse is the ability to take a function from a
Github repo, and create a wrapper around that function to be run on
remote. This saves you the effort of needing to clone or copy a
function. To do so, simply pass in the function url into
``rh.function``.

We’ve implemented the same ``getpid`` function in our Runhouse test
suite
`here <https://github.com/run-house/runhouse/blob/v0.0.4/tests/test_function.py#L114>`__.
Below, we demonstrate how we can directly use the GitHub link and
function name to run this function on remote hardware, without needing
to clone the repo ourselves or reimplement the function locally.

.. code:: python

    pid_git_remote = rh.function(
        fn='https://github.com/run-house/runhouse/blob/v0.0.4/tests/test_function.py:getpid',
        system=cluster,
    )


.. parsed-literal::

    INFO | 2023-05-06 20:53:34,652 | Setting up Function on cluster.
    INFO | 2023-05-06 20:53:34,671 | Installing packages on cluster cpu-cluster: ['GitPackage: https://github.com/huggingface/diffusers.git@v0.11.1', 'torch==1.12.1', 'torchvision==0.13.1', 'transformers', 'datasets', 'evaluate', 'accelerate', 'pip:./diffusers']
    INFO | 2023-05-06 20:54:21,841 | Function setup complete.


.. code:: python

    pid_git_remote()

Additional Function Call Types and Utils
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In addition to the usual function call, Runhouse also supports the
following function types: ``remote``, ``get``, ``repeat``, ``enqueue``,
``map``, and ``starmap``.

We demonstrate the behavior of each of these using the same
``getpid_remote`` function above.

``.remote`` and ``.get``
^^^^^^^^^^^^^^^^^^^^^^^^

Call the function async (using Ray) and return a reference (Ray
ObjectRef) to the object on the cluster.

This is a convenient way to avoid passing large objects back and forth
to your laptop, or to run longer execution in notebooks without locking
up the kernel.

.. code:: python

    getpid_remote_ref = getpid_remote.remote()


.. parsed-literal::

    INFO | 2023-05-06 21:03:17,494 | Running getpid via gRPC
    INFO | 2023-05-06 21:03:17,622 | Time to send message: 0.12 seconds
    INFO | 2023-05-06 21:03:17,624 | Submitted remote call to cluster. Result or logs can be retrieved
     with run_key "getpid_20230506_210317", e.g.
    `rh.cluster(name="/carolineechen/cpu-cluster").get("getpid_20230506_210317", stream_logs=True)` in python
    `runhouse logs "cpu-cluster" getpid_20230506_210317` from the command line.
     or cancelled with
    `rh.cluster(name="/carolineechen/cpu-cluster").cancel("getpid_20230506_210317")` in python or
    `runhouse cancel "cpu-cluster" getpid_20230506_210317` from the command line.


You can use ``.get`` to retrive the value of a reference.

.. code:: python

    getpid_remote.get(getpid_remote_ref)


.. parsed-literal::

    INFO | 2023-05-06 21:03:23,068 | Running getpid via gRPC
    INFO | 2023-05-06 21:03:23,194 | Time to send message: 0.12 seconds




.. parsed-literal::

    26948



You can also directly pass in the ref to another function, and it will
be automatically dereferenced once on the cluster.

.. code:: python

    getpid_remote(getpid_remote_ref)


.. parsed-literal::

    INFO | 2023-05-06 21:03:20,388 | Running getpid via gRPC
    INFO | 2023-05-06 21:03:20,513 | Time to send message: 0.12 seconds




.. parsed-literal::

    51004



``.repeat``
^^^^^^^^^^^

To repeat the function call multiple times, call ``.repeat`` and pass in
the number of times to repeat the function. The function calls take
place across multiple processes, so we see that there are several
process IDs being returned.

.. code:: python

    getpid_remote.repeat(num_repeats=10)


.. parsed-literal::

    INFO | 2023-05-06 20:59:13,495 | Running getpid via gRPC
    INFO | 2023-05-06 20:59:15,381 | Time to send message: 1.88 seconds




.. parsed-literal::

    [26201, 26196, 26200, 26198, 26203, 26202, 26199, 26197, 26346, 26375]



``.enqueue``
^^^^^^^^^^^^

This queues up the function call on the cluster. It ensures a function
call doesn’t run simultaneously with other calls, but will wait until
the execution completes.

.. code:: python

    [getpid_remote.enqueue() for _ in range(3)]


.. parsed-literal::

    INFO | 2023-05-06 21:00:02,004 | Running getpid via gRPC
    INFO | 2023-05-06 21:00:02,772 | Time to send message: 0.77 seconds
    INFO | 2023-05-06 21:00:02,774 | Running getpid via gRPC
    INFO | 2023-05-06 21:00:03,583 | Time to send message: 0.81 seconds
    INFO | 2023-05-06 21:00:03,585 | Running getpid via gRPC
    INFO | 2023-05-06 21:00:04,339 | Time to send message: 0.75 seconds




.. parsed-literal::

    [26786, 26815, 26845]



``.map`` and ``.starmap``
^^^^^^^^^^^^^^^^^^^^^^^^^

These are ways to parallelize a function. ``.map`` maps a function over
a list of arguments, while ``.starmap`` unpacks the elements of the
iterable while mapping.

.. code:: python

    a_map = [1, 2]
    b_map = [2, 5]
    getpid_remote.map(a=a_map, b=b_map)


.. parsed-literal::

    INFO | 2023-05-06 21:06:05,078 | Running getpid via gRPC
    INFO | 2023-05-06 21:06:06,310 | Time to send message: 1.22 seconds




.. parsed-literal::

    [27024, 27023, 27021, 27019, 27020, 27022, 27023, 27023, 27023, 27023]



.. code:: python

    starmap_args = [[1, 2], [1, 3], [1, 4]]
    getpid_remote.starmap(starmap_args)

``stream_logs``
^^^^^^^^^^^^^^^

To stream logs, pass in ``stream_logs=True`` to the function call.

.. code:: python

    getpid_remote(stream_logs=True)


.. parsed-literal::

    INFO | 2023-05-06 21:06:29,351 | Running getpid via gRPC
    INFO | 2023-05-06 21:06:29,477 | Time to send message: 0.12 seconds
    INFO | 2023-05-06 21:06:29,483 | Submitted remote call to cluster. Result or logs can be retrieved
     with run_key "getpid_20230506_210629", e.g.
    `rh.cluster(name="/carolineechen/cpu-cluster").get("getpid_20230506_210629", stream_logs=True)` in python
    `runhouse logs "cpu-cluster" getpid_20230506_210629` from the command line.
     or cancelled with
    `rh.cluster(name="/carolineechen/cpu-cluster").cancel("getpid_20230506_210629")` in python or
    `runhouse cancel "cpu-cluster" getpid_20230506_210629` from the command line.
    :task_name:getpid
    :task_name:getpid




.. parsed-literal::

    27165



Env + Packages
--------------

Our sample ``getpid`` function used only builtin Python dependencies, so
we did not need to worry about the function environment.

For more complex functions relying on external dependencies, Runhouse
provides concepts for packages (individual dependencies/installations)
and environments (group of packages or a conda env).

Package Types
~~~~~~~~~~~~~

Runhouse supports ``pip``, ``conda``, ``reqs`` and ``git`` packages,
which can be constructed in the following ways.

Often times, if using Packages in the context of environments (Envs),
you don’t need to construct them yourself, but can just pass in the
corresponding string, and Runhouse internals will handle the conversion
and installation for you.

.. code:: python

    pip_package = rh.Package.from_string("pip:numpy")
    conda_package = rh.Package.from_string("conda:torch")
    reqs_package = rh.Package.from_string("reqs:./")
    git_package = rh.GitPackage(git_url='https://github.com/huggingface/diffusers.git',
                      install_method='pip',
                      revision='v0.11.1')

You can also send packages between local, remote, and file storage.

.. code:: python

    local_package = rh.Package.from_string("local/path/to/folder")

    package_on_s3 = local_package.to(system="s3", path="/s3/path/to/folder")
    package_on_cluster = local_package.to(system=cluster)

Envs
~~~~

Envs, or environments, keep track of your package installs and
corresponding versions. This allows for reproducible dev environments,
and convenient dependency isolation and management.

The basic environment just consists of a list of Packages, or strings
that represent the packages.

.. code:: python

    env = rh.env(reqs=["numpy", reqs_package, git_package])

When you send an environment object to a cluster, the environment is
automatically set up (packages are installed) on the cluster.

.. code:: python

    env_on_cluster = env.to(system=cluster)

Putting it all together – Cluster, Function, Env
------------------------------------------------

Now that we understand how clusters, functions, and
packages/environments work, we can go on to implement more complex
functions that require external dependencies, and seamlessly run them on
a remote cluster.

.. code:: python

    def add_lists(list_a, list_b):
      import numpy as np

      return np.add(np.array(list_a), np.array(list_b))

Note that in the function defined, we include the import statement
``import numpy as np`` within the function. The import needs to be
inside the function definition in notebook or interactive environments,
but can be outside the function if being used in a Python script.

.. code:: python

    env = rh.env(reqs=["numpy"])
    add_lists_remote = rh.function(fn=add_lists).to(system=cluster, env=env)

    list_a = [1, 2, 3]
    list_b = [2, 3, 4]
    add_lists_remote(list_a, list_b)

Cluster Termination
-------------------

To terminate the cluster, you can call ``sky down cluster-name`` in CLI
or ``cluster_obj.teardown()`` in Python.

.. code:: python

    !sky down cpu-cluster
    # or
    cluster.teardown()
