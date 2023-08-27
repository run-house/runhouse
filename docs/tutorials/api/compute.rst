Compute: Clusters, Functions, Packages, & Envs
==============================================

.. raw:: html

    <p><a href="https://colab.research.google.com/github/run-house/runhouse/blob/stable/docs/notebooks/basics/compute.ipynb">
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

There are three types of supported cluster types:

1. **Bring-your-own (BYO) Cluster**: Existing clusters that you
   already have up, and access through an IP address and SSH
   credentials. Please refer to the :ref:`Bring-Your-Own Cluster` section for further instructions.

2. **On-demand Cluster**: Associated with your cloud account (AWS, GCP,
   Azure, LambdaLabs). There are additional features for these clusters,
   such as cluster (auto) stop. Please refer to
   :ref:`On-Demand Cluster` for instructions on first getting
   cloud credentials set up.

3. **SageMaker Cluster**: Clusters that are created and managed
   through SageMaker, which can be used as a compute backend (just like BYO or On-Demand clusters)
   or for running dedicated training jobs. Please refer to the :ref:`SageMaker Clusters` section for instructions on
   getting setup with SageMaker.

.. code:: python

    # BYO cluster
    cluster = rh.cluster(  # using private key
                name="cpu-cluster",
                ips=['<ip of the cluster>'],
                ssh_creds={'ssh_user': '<user>', 'ssh_private_key':'<path_to_key>'},
            )

    cluster = rh.cluster(  # using password
                name="cpu-cluster",
                ips=['<ip of the cluster>'],
                ssh_creds={'ssh_user': '<user>', 'password':'******'},
            )

    # Using a Cloud provider
    cluster = rh.ondemand_cluster(
                  name="cpu-cluster",
                  instance_type="CPU:8",
                  provider="cheapest",       # "AWS", "GCP", "Azure", "Lambda", or "cheapest" (default)
                  autostop_mins=60,          # Optional, defaults to default_autostop_mins; -1 suspends autostop
              )
    # Launch the cluster
    cluster.up()

    # Using SageMaker as the compute provider
    cluster = rh.sagemaker_cluster(
                  name="sm-cluster",
                  profile="sagemaker" # AWS profile with a role ARN configured for SageMaker
              )
    # Launch the cluster
    cluster.up()

You can set default configs for future cluster constructions. These
defaults are associated with either only your local environment (if you
don’t login to Runhouse), or can be reused across devices (if they are
saved to your Runhouse account).

.. code:: python

    rh.configs.set('use_spot', False)
    rh.configs.set('default_autostop', 30)

    rh.configs.upload_defaults()


.. parsed-literal::

    INFO | 2023-05-18 12:48:20,821 | Uploaded defaults for user to rns.


Useful Cluster APIs
~~~~~~~~~~~~~~~~~~~

To run CLI or Python commands on the cluster:

.. code:: python

    cluster.run(['pip install numpy && pip freeze | grep numpy'])


.. parsed-literal::

    INFO | 2023-05-18 13:59:54,417 | Running command on cpu-cluster: pip install numpy && pip freeze | grep numpy
    Requirement already satisfied: numpy in /opt/conda/lib/python3.10/site-packages (1.24.3)
    numpy==1.24.3


.. parsed-literal::

    [(0,
      'Requirement already satisfied: numpy in /opt/conda/lib/python3.10/site-packages (1.24.3)\nnumpy==1.24.3\n',
      '')]



.. code:: python

    cluster.run_python(['import numpy', 'print(numpy.__version__)'])


.. parsed-literal::

    INFO | 2023-05-18 14:00:01,581 | Running command on cpu-cluster: python3 -c "import numpy; print(numpy.__version__)"
    1.24.3



.. parsed-literal::

    [(0, '1.24.3\n', '')]



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

To open a port, if you want to run an application on the cluster that
requires a port to be open, e.g. Tensorboard, Gradio:

.. code:: python

    cluster.ssh_tunnel(local_port=7860, remote_port=7860)

Function
--------

Runhouse’s Function API lets you define functions to be run on remote
hardware (your cluster above!). Simply pass in a local (or a GitHub)
function, the intended remote hardware, and any dependencies; Runhouse
will handle the rest for you.

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
    getpid_remote = rh.function(fn=getpid).to(system=cluster)


.. parsed-literal::

    INFO | 2023-06-26 13:40:45,990 | Writing out function function to /Users/caroline/Documents/runhouse/runhouse/docs/notebooks/basics/getpid_fn.py. Please make sure the function does not rely on any local variables, including imports (which should be moved inside the function body).
    INFO | 2023-06-26 13:40:45,993 | Setting up Function on cluster.
    INFO | 2023-06-26 13:40:57,014 | Checking server test-cluster
    INFO | 2023-06-26 13:40:57,989 | Server test-cluster is up.
    INFO | 2023-06-26 13:40:59,299 | Installing packages on cluster test-cluster: ['Package: runhouse']
    INFO | 2023-06-26 13:41:00,872 | Function setup complete.


To run the function, simply call it just as you would a local function,
and the function automatically runs on your specified hardware!

.. code:: python

    print(f"local function result: {getpid()}")
    print(f"remote function result: {getpid_remote()}")


.. parsed-literal::

    local function result: 73128
    INFO | 2023-06-26 13:41:05,026 | Running getpid via HTTP
    INFO | 2023-06-26 13:41:05,290 | Submitted remote call to cluster for 20230626_134105
    :job_id:17000000
    :task_name:get_fn_from_pointers
    :job_id:17000000
    INFO | 2023-06-26 17:41:05,959 | Loaded Runhouse config from /home/ubuntu/.rh/config.yaml
    :task_name:get_fn_from_pointers
    INFO | 2023-06-26 17:41:06,566 | Appending /home/ubuntu/runhouse/docs/notebooks/basics to sys.path
    INFO | 2023-06-26 17:41:06,566 | Importing module getpid_fn
    INFO | 2023-06-26 13:41:06,446 | Time to call remote function: 1.42 seconds
    remote function result: 183873


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

    INFO | 2023-05-18 14:00:56,859 | Setting up Function on cluster.
    INFO | 2023-05-18 14:00:56,863 | Installing packages on cluster cpu-cluster: ['GitPackage: https://github.com/run-house/runhouse.git@v0.0.4']
    INFO | 2023-05-18 14:00:59,540 | Function setup complete.


.. code:: python

    pid_git_remote()


.. parsed-literal::

    INFO | 2023-05-18 14:01:01,496 | Running getpid via gRPC
    INFO | 2023-05-18 14:01:01,867 | Time to send message: 0.37 seconds




.. parsed-literal::

    24065



Additional Function Call Types
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

    WARNING | 2023-06-26 13:41:14,945 | /Users/caroline/Documents/runhouse/runhouse/runhouse/rns/function.py:421: UserWarning: `remote()` is deprecated, use `run()` instead
      warnings.warn("`remote()` is deprecated, use `run()` instead")

    INFO | 2023-06-26 13:41:14,946 | Running getpid via HTTP
    INFO | 2023-06-26 13:41:15,199 | Submitted remote call to cluster for 20230626_134114


You can use ``.get`` to retrive the value of a reference.

.. code:: python

    getpid_remote.get(getpid_remote_ref)


.. parsed-literal::

    :job_id:17000000
    INFO | 2023-06-26 17:41:15,889 | Loaded Runhouse config from /home/ubuntu/.rh/config.yaml
    :task_name:get_fn_from_pointers
    INFO | 2023-06-26 17:41:16,491 | Appending /home/ubuntu/runhouse/docs/notebooks/basics to sys.path
    INFO | 2023-06-26 17:41:16,491 | Importing module getpid_fn
    :job_id:17000000
    :task_name:get_fn_from_pointers




.. parsed-literal::

    183941



You can also directly pass in the ref to another function, and it will
be automatically dereferenced once on the cluster.

.. code:: python

    getpid_remote(getpid_remote_ref)


.. parsed-literal::

    INFO | 2023-06-26 13:41:28,636 | Running getpid via HTTP
    INFO | 2023-06-26 13:41:28,900 | Submitted remote call to cluster for 20230626_134128
    :job_id:17000000
    INFO | 2023-06-26 17:41:29,561 | Loaded Runhouse config from /home/ubuntu/.rh/config.yaml
    :task_name:get_fn_from_pointers
    INFO | 2023-06-26 17:41:30,161 | Appending /home/ubuntu/runhouse/docs/notebooks/basics to sys.path
    INFO | 2023-06-26 17:41:30,161 | Importing module getpid_fn
    :job_id:17000000
    :task_name:get_fn_from_pointers
    INFO | 2023-06-26 13:41:29,998 | Time to call remote function: 1.36 seconds




.. parsed-literal::

    367951



``.repeat``
^^^^^^^^^^^

To repeat the function call multiple times, call ``.repeat`` and pass in
the number of times to repeat the function. The function calls take
place across multiple processes, so we see that there are several
process IDs being returned.

.. code:: python

    getpid_remote.repeat(num_repeats=10)


.. parsed-literal::

    INFO | 2023-06-26 13:41:39,838 | Running getpid via HTTP
    INFO | 2023-06-26 13:41:42,960 | Time to call remote function: 3.12 seconds




.. parsed-literal::

    [184080,
     184084,
     184085,
     184082,
     184078,
     184081,
     184083,
     184079,
     184190,
     184285]



``.enqueue``
^^^^^^^^^^^^

This queues up the function call on the cluster. It ensures a function
call doesn’t run simultaneously with other calls, but will wait until
the execution completes.

.. code:: python

    [getpid_remote.enqueue() for _ in range(3)]


.. parsed-literal::

    INFO | 2023-06-26 13:41:45,818 | Running getpid via HTTP
    INFO | 2023-06-26 13:41:47,255 | Time to call remote function: 1.43 seconds
    INFO | 2023-06-26 13:41:47,256 | Running getpid via HTTP
    INFO | 2023-06-26 13:41:48,676 | Time to call remote function: 1.42 seconds
    INFO | 2023-06-26 13:41:48,677 | Running getpid via HTTP
    INFO | 2023-06-26 13:41:50,102 | Time to call remote function: 1.42 seconds




.. parsed-literal::

    [184727, 184791, 184857]



``.map`` and ``.starmap``
^^^^^^^^^^^^^^^^^^^^^^^^^

These are ways to parallelize a function. ``.map`` maps a function over
a list of arguments, while ``.starmap`` unpacks the elements of the
iterable while mapping.

.. code:: python

    a_map = [1, 2]
    b_map = [2, 5]
    getpid_remote.map(a_map, b_map)


.. parsed-literal::

    INFO | 2023-06-26 13:41:53,812 | Running getpid via HTTP
    INFO | 2023-06-26 13:41:55,238 | Time to call remote function: 1.42 seconds




.. parsed-literal::

    [184927, 184932]



.. code:: python

    starmap_args = [[1, 2], [1, 3], [1, 4]]
    getpid_remote.starmap(starmap_args)


.. parsed-literal::

    INFO | 2023-06-26 13:41:57,452 | Running getpid via HTTP
    INFO | 2023-06-26 13:41:57,597 | Time to call remote function: 0.14 seconds




.. parsed-literal::

    [184928, 184928, 184930]



Function Logging
~~~~~~~~~~~~~~~~

``stream_logs``
^^^^^^^^^^^^^^^

To stream logs to local during the remote function call, pass in
``stream_logs=True`` to the function call.

.. code:: python

    getpid_remote(stream_logs=True)


.. parsed-literal::

    INFO | 2023-06-26 13:42:02,197 | Running getpid via HTTP
    INFO | 2023-06-26 13:42:02,444 | Submitted remote call to cluster for 20230626_134202
    :job_id:17000000
    INFO | 2023-06-26 17:42:02,530 | Loaded Runhouse config from /home/ubuntu/.rh/config.yaml
    :task_name:get_fn_from_pointers
    INFO | 2023-06-26 17:42:03,133 | Appending /home/ubuntu/runhouse/docs/notebooks/basics to sys.path
    INFO | 2023-06-26 17:42:03,133 | Importing module getpid_fn
    :job_id:17000000
    :task_name:get_fn_from_pointers
    INFO | 2023-06-26 13:42:03,014 | Time to call remote function: 0.82 seconds




.. parsed-literal::

    185046



Function logs are also automatically output onto a log file on cluster
it is run on. You can refer to `Runhouse Logging
Docs <https://runhouse-docs.readthedocs-hosted.com/en/latest/debugging_logging.html>`__
for more information on accessing these logs.

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

Base Env
^^^^^^^^

The basic Env resource just consists of a list of Packages, or strings
that represent the packages. You can additionally add any environment
variables by providing a Dict or ``.env`` local file path, and also set
the working directory to be synced over (defaults to base GitHub repo).

.. code:: python

    env = rh.env(reqs=["numpy", reqs_package, git_package], env_vars={"USER": "*****"})

When you send an environment object to a cluster, the environment is
automatically set up (packages are installed) on the cluster.

.. code:: python

    env_on_cluster = env.to(system=cluster)

Conda Env
^^^^^^^^^

The CondaEnv resource represents a Conda environment that can be used to
set up reproducible Conda envs across clusters.

There are several ways to construct a Runhouse CondaEnv object using
``rh.conda_env``, by passing in any of the following into the
``conda_env`` parameter:

1. A yaml file corresponding to a conda environment config
2. A dict corresponding to a conda environment config
3. Name of an existing conda env on your local machine
4. Leaving the argument empty. In this case, we’ll construct a new Conda
   environment for you, using the list you pass into ``reqs``.

Beyond the conda config, you can also add any additional requirements
you’d like to install in the environment by adding
``reqs = List[packages]``.

.. code:: python

    # 1. config yaml file
    conda_env = rh.conda_env(conda_env="conda_env.yml", reqs=["numpy", "diffusers"], name="yaml_env")
    # 2. config dict
    conda_dict = {"name": "conda_env", "channels": ["conda-forge"], "dependencies": ["python=3.10.0"]}
    conda_env = rh.env(conda_env=conda_dict, name="dict_env")
    # 3. local conda env
    conda_env = rh.conda_env(conda_env="local_conda_env", name="from_local_env")
    # 4. empty, construct from reqs
    conda_env = rh.conda_env(reqs=["numpy", "diffusers"], name="new_env")

As with the base env, we can set up a conda env on the cluster with:

.. code:: python

    conda_env_on_cluster = conda_env.to(system=cluster)

Previously in the cluster section, we mentioned several cluster APIs
such as running CLI or Python commands. These all run on the base
environment in the examples above, but now that we’ve defined a Conda
env, let’s demonstrate how we can accomplish this inside a Conda env on
the cluster:

.. code:: python

    # run Python command within the conda env
    cluster.run_python("import diffusers", 'print(diffusers.__version__)', env=conda_env)

    # install additional package on given env
    cluster.install_packages(["pandas"], env=conda_env)

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

Now that you understand the basics, feel free to play around with more
complicated scenarios! You can also check out our additional API and
example usage tutorials on our `docs
site <https://runhouse-docs.readthedocs-hosted.com/en/latest/index.html>`__.

Cluster Termination
-------------------

To terminate the cluster, you can call ``sky down cluster-name`` in CLI
or ``cluster_obj.teardown()`` in Python.

.. code:: python

    !sky down cpu-cluster
    # or
    cluster.teardown()
