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

.. code:: ipython3

    !pip install runhouse

.. code:: ipython3

    import runhouse as rh

Optionally, to login to Runhouse to sync any secrets.

.. code:: ipython3

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

Each cluster must be provided with a unique ``name`` identifier during
construction. This ``name`` parameter is used for saving down or loading
previous saved clusters, and also used for various CLI commands for the
cluster.

.. code:: ipython3

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

.. code:: ipython3

    rh.configs.set('use_spot', False)
    rh.configs.set('default_autostop', 30)

    rh.configs.upload_defaults()

Useful Cluster APIs
~~~~~~~~~~~~~~~~~~~

To run CLI or Python commands on the cluster:

.. code:: ipython3

    cluster.run(['pip install numpy && pip freeze | grep numpy'])


.. parsed-literal::

    INFO | 2023-08-29 03:35:44.910826 | Running command on cpu-cluster: pip install numpy && pip freeze | grep numpy
    Warning: Permanently added '34.205.23.213' (ED25519) to the list of known hosts.


.. parsed-literal::

    Requirement already satisfied: numpy in /opt/conda/lib/python3.10/site-packages (1.25.2)
    numpy==1.25.2


.. parsed-literal::

    [(0,
      'Requirement already satisfied: numpy in /opt/conda/lib/python3.10/site-packages (1.25.2)\nnumpy==1.25.2\n',
      "Warning: Permanently added '34.205.23.213' (ED25519) to the list of known hosts.\r\n")]



.. code:: ipython3

    cluster.run_python(['import numpy', 'print(numpy.__version__)'])


.. parsed-literal::

    INFO | 2023-08-29 03:35:50.911455 | Running command on cpu-cluster: python3 -c "import numpy; print(numpy.__version__)"


.. parsed-literal::

    1.25.2



.. parsed-literal::

    [(0, '1.25.2\n', '')]



To ssh into the cluster:

.. code:: ipython3

    # Python
    cluster.ssh()

    # CLI
    !ssh cpu-cluster

To tunnel a JupyterLab server into your local browser:

.. code:: ipython3

    # Python
    cluster.notebook()

    # CLI
    !runhouse notebook cpu-cluster

To open a port, if you want to run an application on the cluster that
requires a port to be open, e.g. Tensorboard, Gradio:

.. code:: ipython3

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

.. code:: ipython3

    # Local Function
    def getpid(a=0, b=0):
        import os
        return os.getpid() + a + b

To construct a function that runs ``getpid`` on a remote cluster, we
wrap it using ``rh.function``, and specify ``system=cluster``.

.. code:: ipython3

    # Remote Function
    getpid_remote = rh.function(fn=getpid).to(system=cluster)


.. parsed-literal::

    INFO | 2023-08-29 03:59:14.328987 | Writing out function function to /Users/caroline/Documents/runhouse/runhouse/docs/notebooks/basics/getpid_fn.py. Please make sure the function does not rely on any local variables, including imports (which should be moved inside the function body).
    INFO | 2023-08-29 03:59:14.332706 | Setting up Function on cluster.
    INFO | 2023-08-29 03:59:14.807140 | Connected (version 2.0, client OpenSSH_8.2p1)
    INFO | 2023-08-29 03:59:15.280859 | Authentication (publickey) successful!
    INFO | 2023-08-29 03:59:17.534412 | Found credentials in shared credentials file: ~/.aws/credentials
    INFO | 2023-08-29 03:59:18.002794 | Checking server cpu-cluster
    INFO | 2023-08-29 03:59:19.059074 | Server cpu-cluster is up.
    INFO | 2023-08-29 03:59:19.061851 | Copying package from file:///Users/caroline/Documents/runhouse/runhouse to: cpu-cluster
    INFO | 2023-08-29 03:59:20.822780 | Calling env_20230829_035913.install


.. parsed-literal::

    base servlet: Calling method install on module env_20230829_035913
    Installing package: Package: runhouse
    Installing Package: runhouse with method reqs.
    reqs path: runhouse/requirements.txt
    pip installing requirements from runhouse/requirements.txt with: -r runhouse/requirements.txt
    Running: /opt/conda/bin/python3.10 -m pip install -r runhouse/requirements.txt


.. parsed-literal::

    INFO | 2023-08-29 03:59:22.728154 | Time to call env_20230829_035913.install: 1.91 seconds
    INFO | 2023-08-29 03:59:22.981633 | Function setup complete.


To run the function, simply call it just as you would a local function,
and the function automatically runs on your specified hardware!

.. code:: ipython3

    print(f"local function result: {getpid()}")
    print(f"remote function result: {getpid_remote()}")


.. parsed-literal::

    INFO | 2023-08-29 03:59:43.821391 | Calling getpid.call


.. parsed-literal::

    local function result: 7592
    base servlet: Calling method call on module getpid


.. parsed-literal::

    INFO | 2023-08-29 03:59:44.078775 | Time to call getpid.call: 0.26 seconds


.. parsed-literal::

    remote function result: 1382396


Git Functions
~~~~~~~~~~~~~

A neat feature of Runhouse is the ability to take a function from a
Github repo, and create a wrapper around that function to be run on
remote. This saves you the effort of needing to clone or copy a
function. To do so, simply pass in the function url into
``rh.function``.

We’ve implemented the same ``getpid`` function
`here <https://github.com/run-house/runhouse/blob/main/docs/notebooks/sample_fn.py>`__.
Below, we demonstrate how we can directly use the GitHub link and
function name to run this function on remote hardware, without needing
to clone the repo ourselves or reimplement the function locally.

.. code:: ipython3

    pid_git_remote = rh.function(
        fn='https://github.com/run-house/runhouse/blob/main/docs/notebooks/sample_fn.py:getpid',
        system=cluster,
    )


.. parsed-literal::

    INFO | 2023-08-29 04:00:01.870718 | Setting up Function on cluster.
    INFO | 2023-08-29 04:00:01.873021 | Copying package from file:///Users/caroline/Documents/runhouse/runhouse to: cpu-cluster
    INFO | 2023-08-29 04:00:03.145979 | Copying package from file:///Users/caroline/Documents/runhouse/runhouse to: cpu-cluster
    INFO | 2023-08-29 04:00:04.625905 | Calling env_20230829_035957.install


.. parsed-literal::

    base servlet: Calling method install on module env_20230829_035957
    Installing package: GitPackage: https://github.com/run-house/runhouse.git@main
    Pulling: git -C ./runhouse fetch https://github.com/run-house/runhouse.git
    Checking out revision: git checkout main
    Installing GitPackage: https://github.com/run-house/runhouse.git@main with method local.
    Installing package: Package: runhouse
    Installing Package: runhouse with method reqs.
    reqs path: runhouse/requirements.txt
    pip installing requirements from runhouse/requirements.txt with: -r runhouse/requirements.txt
    Running: /opt/conda/bin/python3.10 -m pip install -r runhouse/requirements.txt
    Installing package: Package: runhouse
    Installing Package: runhouse with method reqs.
    reqs path: runhouse/requirements.txt
    pip installing requirements from runhouse/requirements.txt with: -r runhouse/requirements.txt
    Running: /opt/conda/bin/python3.10 -m pip install -r runhouse/requirements.txt


.. parsed-literal::

    INFO | 2023-08-29 04:00:08.100045 | Time to call env_20230829_035957.install: 3.47 seconds
    INFO | 2023-08-29 04:00:08.275688 | Function setup complete.


.. code:: ipython3

    pid_git_remote(1)


.. parsed-literal::

    INFO | 2023-08-29 04:00:12.015937 | Calling getpid.call


.. parsed-literal::

    base servlet: Calling method call on module getpid


.. parsed-literal::

    INFO | 2023-08-29 04:00:12.285294 | Time to call getpid.call: 0.27 seconds




.. parsed-literal::

    1382397



Function Call Types: ``.remote`` and ``.run``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can use ``fn.remote()`` to have the function return a remote object,
rather than the proper result. This is a convenient way to avoid passing
large objects back and forth to your laptop, or to run longer execution
in notebooks without locking up the kernel.

.. code:: ipython3

    getpid_remote_obj = getpid_remote.remote()
    getpid_remote_obj


.. parsed-literal::

    INFO | 2023-08-29 04:42:17.026532 | Calling getpid.call


.. parsed-literal::

    base servlet: Calling method call on module getpid


.. parsed-literal::

    INFO | 2023-08-29 04:42:17.900012 | Time to call getpid.call: 0.87 seconds




.. parsed-literal::

    <runhouse.rns.blobs.blob.Blob at 0x154dab3d0>



To retrieve the data from the returned remote object, you can call
``.fetch()`` on the remote object.

.. code:: ipython3

    getpid_remote_obj.fetch()


.. parsed-literal::

    INFO | 2023-08-29 04:42:18.626515 | Getting getpid_call_20230829_044209_708686
    INFO | 2023-08-29 04:42:18.780105 | Time to get getpid_call_20230829_044209_708686: 0.15 seconds




.. parsed-literal::

    1382396



To run a function async, use ``fn.run()``, which returns a ``run_key``
that can be used to retrieve the results and logs at a later point.

.. code:: ipython3

    getpid_run_key = getpid_remote.run()
    getpid_run_key


.. parsed-literal::

    INFO | 2023-08-29 04:42:20.182323 | Calling getpid.call
    INFO | 2023-08-29 04:42:20.318719 | Time to call getpid.call: 0.14 seconds




.. parsed-literal::

    'getpid_call_20230829_044212_868665'



To retrieve the result of the function run, you can call
``cluster.get()`` and pass in the ``run_key``.

.. code:: ipython3

    cluster.get(getpid_run_key)


.. parsed-literal::

    INFO | 2023-08-29 04:42:28.747188 | Getting getpid_call_20230829_044212_868665
    INFO | 2023-08-29 04:42:28.875886 | Time to get getpid_call_20230829_044212_868665: 0.13 seconds




.. parsed-literal::

    1382396



Function Logging
~~~~~~~~~~~~~~~~

``stream_logs``
^^^^^^^^^^^^^^^

To stream logs to local during the remote function call, pass in
``stream_logs=True`` to the function call.

.. code:: ipython3

    getpid_remote(stream_logs=True)


.. parsed-literal::

    INFO | 2023-08-29 04:43:17.812658 | Calling getpid.call


.. parsed-literal::

    base servlet: Calling method call on module getpid


.. parsed-literal::

    INFO | 2023-08-29 04:43:18.107531 | Time to call getpid.call: 0.29 seconds




.. parsed-literal::

    1382396



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

.. code:: ipython3

    pip_package = rh.Package.from_string("pip:numpy")
    conda_package = rh.Package.from_string("conda:torch")
    reqs_package = rh.Package.from_string("reqs:./")
    git_package = rh.GitPackage(git_url='https://github.com/huggingface/diffusers.git',
                                install_method='pip',
                                revision='v0.11.1')

You can also send packages between local, remote, and file storage.

.. code:: ipython3

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

.. code:: ipython3

    env = rh.env(reqs=["numpy", reqs_package, git_package], env_vars={"USER": "*****"})

When you send an environment object to a cluster, the environment is
automatically set up (packages are installed) on the cluster.

.. code:: ipython3

    env_on_cluster = env.to(system=cluster)


.. parsed-literal::

    INFO | 2023-08-29 04:44:06.955053 | Copying package from file:///Users/caroline/Documents/runhouse/runhouse to: cpu-cluster
    INFO | 2023-08-29 04:44:08.250678 | Copying package from file:///Users/caroline/Documents/runhouse/runhouse to: cpu-cluster
    INFO | 2023-08-29 04:44:09.741572 | Calling env_20230829_044402._set_env_vars


.. parsed-literal::

    base servlet: Calling method _set_env_vars on module env_20230829_044402


.. parsed-literal::

    INFO | 2023-08-29 04:44:10.028261 | Time to call env_20230829_044402._set_env_vars: 0.29 seconds
    INFO | 2023-08-29 04:44:10.029212 | Calling env_20230829_044402.install


.. parsed-literal::

    base servlet: Calling method install on module env_20230829_044402
    Installing package: Package: numpy
    Installing Package: numpy with method pip.
    Running: pip install numpy
    Installing package: Package: runhouse
    Installing Package: runhouse with method reqs.
    reqs path: runhouse/requirements.txt
    pip installing requirements from runhouse/requirements.txt with: -r runhouse/requirements.txt
    Running: /opt/conda/bin/python3.10 -m pip install -r runhouse/requirements.txt
    Installing package: GitPackage: https://github.com/huggingface/diffusers.git@v0.11.1
    Pulling: git -C ./diffusers fetch https://github.com/huggingface/diffusers.git
    Checking out revision: git checkout v0.11.1
    Installing GitPackage: https://github.com/huggingface/diffusers.git@v0.11.1 with method pip.
    Running: pip install ./diffusers
    Installing package: Package: runhouse
    Installing Package: runhouse with method reqs.
    reqs path: runhouse/requirements.txt
    pip installing requirements from runhouse/requirements.txt with: -r runhouse/requirements.txt
    Running: /opt/conda/bin/python3.10 -m pip install -r runhouse/requirements.txt


.. parsed-literal::

    INFO | 2023-08-29 04:44:19.111342 | Time to call env_20230829_044402.install: 9.08 seconds


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

.. code:: ipython3

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

.. code:: ipython3

    conda_env_on_cluster = conda_env.to(system=cluster)


.. parsed-literal::

    INFO | 2023-08-29 04:48:21.600485 | Copying package from file:///Users/caroline/Documents/runhouse/runhouse to: cpu-cluster
    INFO | 2023-08-29 04:48:23.132095 | Calling new_env.install


.. parsed-literal::

    new_env servlet: Calling method install on module new_env
    Env already installed, skipping


.. parsed-literal::

    INFO | 2023-08-29 04:48:24.358608 | Time to call new_env.install: 1.23 seconds


Previously in the cluster section, we mentioned several cluster APIs
such as running CLI or Python commands. These all run on the base
environment in the examples above, but now that we’ve defined a Conda
env, let’s demonstrate how we can accomplish this inside a Conda env on
the cluster:

.. code:: ipython3

    # run Python command within the conda env
    cluster.run_python(['import numpy', 'print(numpy.__version__)'], env=conda_env)


.. parsed-literal::

    INFO | 2023-08-29 05:14:08.725396 | Running command on cpu-cluster: conda run -n new_env python3 -c "import numpy; print(numpy.__version__)"


.. parsed-literal::

    1.25.2


.. parsed-literal::

    [(0, '1.25.2\n\n', '')]



.. code:: ipython3

    # install additional package on given env
    cluster.install_packages(["pandas"], env=conda_env)

Putting it all together – Cluster, Function, Env
------------------------------------------------

Now that we understand how clusters, functions, and
packages/environments work, we can go on to implement more complex
functions that require external dependencies, and seamlessly run them on
a remote cluster.

.. code:: ipython3

    def add_lists(list_a, list_b):
      import numpy as np

      return np.add(np.array(list_a), np.array(list_b))

Note that in the function defined, we include the import statement
``import numpy as np`` within the function. The import needs to be
inside the function definition in notebook or interactive environments,
but can be outside the function if being used in a Python script.

.. code:: ipython3

    env = rh.env(reqs=["numpy"])
    add_lists_remote = rh.function(fn=add_lists).to(system=cluster, env=env)

    list_a = [1, 2, 3]
    list_b = [2, 3, 4]
    add_lists_remote(list_a, list_b)


.. parsed-literal::

    INFO | 2023-08-29 05:20:27.959315 | Writing out function function to /Users/caroline/Documents/runhouse/runhouse/docs/notebooks/basics/add_lists_fn.py. Please make sure the function does not rely on any local variables, including imports (which should be moved inside the function body).
    INFO | 2023-08-29 05:20:27.962973 | Setting up Function on cluster.
    INFO | 2023-08-29 05:20:27.965670 | Copying package from file:///Users/caroline/Documents/runhouse/runhouse to: cpu-cluster
    INFO | 2023-08-29 05:20:29.406978 | Calling env_20230829_052021.install


.. parsed-literal::

    base servlet: Calling method install on module env_20230829_052021
    Installing package: Package: numpy
    Installing Package: numpy with method pip.
    Running: pip install numpy
    Installing package: Package: runhouse
    Installing Package: runhouse with method reqs.
    reqs path: runhouse/requirements.txt
    pip installing requirements from runhouse/requirements.txt with: -r runhouse/requirements.txt
    Running: /opt/conda/bin/python3.10 -m pip install -r runhouse/requirements.txt


.. parsed-literal::

    INFO | 2023-08-29 05:20:32.575986 | Time to call env_20230829_052021.install: 3.17 seconds
    INFO | 2023-08-29 05:20:32.774676 | Function setup complete.
    INFO | 2023-08-29 05:20:32.791597 | Calling add_lists.call


.. parsed-literal::

    base servlet: Calling method call on module add_lists


.. parsed-literal::

    INFO | 2023-08-29 05:20:33.086075 | Time to call add_lists.call: 0.29 seconds




.. parsed-literal::

    array([3, 5, 7])



Now that you understand the basics, feel free to play around with more
complicated scenarios! You can also check out our additional API and
example usage tutorials on our `docs
site <https://runhouse-docs.readthedocs-hosted.com/en/latest/index.html>`__.

Cluster Termination
-------------------

To terminate the cluster, you can call ``sky down cluster-name`` in CLI
or ``cluster_obj.teardown()`` in Python.

.. code:: ipython3

    !sky down cpu-cluster
    # or
    cluster.teardown()
