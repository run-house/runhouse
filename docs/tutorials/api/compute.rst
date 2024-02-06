Compute: Clusters, Functions & Modules, Packages & Envs
=======================================================

.. raw:: html

    <p><a href="https://colab.research.google.com/github/run-house/runhouse/blob/stable/docs/notebooks/api/compute.ipynb">
    <img height="20px" width="117px" src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a></p>

Runhouse has several abstractions to provide a seamless flow of code and
execution across local and remote compute. The abstractions are cloud
provider-agnostic, and provide living, reusable services.

The Cluster, Function, and Module APIs blur the line between program
execution and deployment.

The Env and Package APIs help to provide convenient dependency isolation
and management.

Install Runhouse
----------------

.. code:: ipython3

    !pip install runhouse[sky]

.. code:: ipython3

    import runhouse as rh

Optionally, to login to Runhouse to sync any secrets.

.. code:: ipython3

    !runhouse login

Cluster
-------

A cluster is the most basic form of compute in Runhouse, largely
representing a group of instances or VMs connected with Ray. They
largely fall in two categories: 1. Static clusters, which are accessed
via IP addresses and SSH credentials. 2. On-Demand clusters, which are
automatically spun up and down for you with your local cloud account.

Runhouse provides various APIs for interacting with remote clusters,
such as terminating an on-demand cloud cluster or running remote CLI or
Python commands from your local dev environment.

Initialize your Cluster
~~~~~~~~~~~~~~~~~~~~~~~

Each cluster must be provided with a unique ``name`` identifier during
construction. This ``name`` parameter is used for saving down or loading
previous saved clusters, and also used for various CLI commands for the
cluster.

.. code:: ipython3

    # Static cluster
    cluster = rh.cluster(  # using private key
                  name="cpu-cluster",
                  ips=['<ip of the cluster>'],
                  ssh_creds={'ssh_user': '<user>', 'ssh_private_key':'<path_to_key>'},
              )

    # On-demand cluster
    cluster = rh.ondemand_cluster(
                  name="cpu-cluster",
                  instance_type="CPU:8",
                  provider="cheapest",       # "AWS", "GCP", "Azure", "Lambda", or "cheapest" (default)
                  autostop_mins=60,          # Optional, defaults to default_autostop_mins; -1 suspends autostop
              )
    # Launch the cluster, only supported for on-demand clusters
    cluster.up_if_not()
    cluster.run(["echo started!"])

Useful Cluster APIs
~~~~~~~~~~~~~~~~~~~

To run CLI or Python commands on the cluster:

.. code:: ipython3

    cluster.run(['pip install numpy && pip freeze | grep numpy'])


.. parsed-literal::

    Requirement already satisfied: numpy in /opt/conda/lib/python3.10/site-packages (1.26.3)
    numpy==1.26.3


.. code:: ipython3

    cluster.run_python(['import numpy', 'print(numpy.__version__)'])


.. parsed-literal::

    1.26.3


To ssh into the cluster (which you probably want to do in an interactive
shell rather than a notebook):

::

   # Python
   >>> cluster.ssh()

   # CLI
   $ ssh cpu-cluster

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
wrap it using ``rh.function``, and send it to a cluster.

.. code:: ipython3

    # Remote Function
    getpid_remote = rh.function(fn=getpid).to(system=cluster)


To run the function, simply call it just as you would a local function,
and the function automatically runs on your specified hardware!

.. code:: ipython3

    print(f"local function result: {getpid()}")
    print(f"remote function result: {getpid_remote()}")


.. parsed-literal::

    INFO | 2024-01-12 16:33:21.308681 | Calling getpid.call


.. parsed-literal::

    local function result: 68830
    base_env servlet: Calling method call on module getpid


.. parsed-literal::

    INFO | 2024-01-12 16:33:21.668737 | Time to call getpid.call: 0.36 seconds


.. parsed-literal::

    remote function result: 31069


``stream_logs``
^^^^^^^^^^^^^^^

By default, logs and stdout will stream back to you as the function
runs. If you’re quite latency sensitive, you may see a slight
performance gain if you disable it by passing ``stream_logs=False`` to
the function call:

.. code:: ipython3

    getpid_remote(stream_logs=False)

.. parsed-literal::

    31069



Function logs are also automatically output onto a log file on cluster
it is run on. You can refer to `Runhouse Logging
Docs <https://www.run.house/docs/debugging_logging>`__ for more
information on accessing these logs.

Modules
-------

A ``Function`` is actually a subclass of a more generic Runhouse concept
called a ``Module``, which represents the class analogue to a function.
Like ``Function``, you can send a ``Module`` to a remote cluster and
interact with it natively by calling its methods, but it can also
persist and utilize live state via instance methods. This is a
superpower of Runhouse - often introducing state into a service means
spinning up, connecting, and securing auxiliary services like Redis,
Celery, etc. In Runhouse, state is built in, and lives natively
in-memory in Python so it’s ridiculously fast.

Converting existing class to Runhouse Module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you have an existing non-Runhouse class that you would like to run
remotely, you can convert it into a ``Module`` via the ``module``
factory (not the lowercase m in ``rh.module``):

::

   from package import Model

   RemoteModel = rh.module(cls=Model, system=cluster)
   remote_model = RemoteModel(model_id="bert-base-uncased", device="cuda")
   remote_model.predict("Hello world!")  # Runs on cluster

Creating your own Runhouse Module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can also construct a new ``Module`` simply by subclassing
``rh.Module`` (note the uppercase M). Note that because you’ll construct
this class locally prior to sending it to a remote cluster, if there is
a computationally heavy operation such as loading a dataset or model
that you only want to be done remotely, you probably want to wrap that
operation in an instance method and call it on the module after you’ve
sent it to your remote compute. One way of doing so is through lazy
initialization, as in the data property of the module below.

When working in a notebook setting, we define the class in another file,
``pid_module.py``, because module code is synced to the cluster and
there isn’t a robust standard for extracting code from notebooks. In
normal Python, you can use any Module as you would a normal Python
class.

.. code:: ipython3

    %%writefile pid_module.py

    import os
    import runhouse as rh

    class PIDModule(rh.Module):
        def __init__(self, a: int=0):
            super().__init__()
            self.a = a

        @property
        def data(self):
            if not hasattr(self, '_data'):
                self._data = load_dataset()
            return self._data

        def getpid(self):
            return os.getpid() + self.a


.. parsed-literal::

    Overwriting pid_module.py


.. code:: ipython3

    from pid_module import PIDModule

    remote_module = PIDModule(a=5).to(cluster)

.. code:: ipython3

    remote_module.getpid()


.. parsed-literal::

    INFO | 2024-01-12 16:52:41.394668 | Calling PIDModule.getpid


.. parsed-literal::

    base_env servlet: Calling method getpid on module PIDModule


.. parsed-literal::

    INFO | 2024-01-12 16:52:41.633281 | Time to call PIDModule.getpid: 0.24 seconds


.. parsed-literal::

    31074



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

    local_package = rh.Package.from_string("./")  # ./ defaults to the current git root, but you can also pass an abs path

    # package_on_s3 = local_package.to(system="s3", path="/s3/path/to/folder")
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


Conda Env
^^^^^^^^^

The CondaEnv resource represents a Conda environment that can be used to
set up reproducible Conda envs across clusters.

There are several ways to construct a Runhouse CondaEnv object using
``rh.conda_env``, by passing in any of the following into the
``conda_env`` parameter:

1. A yaml file corresponding to a conda environment config

::

   conda_env = rh.conda_env(conda_env="conda_env.yml", reqs=["numpy", "diffusers"], name="yaml_env")

2. A dict corresponding to a conda environment config

::

   conda_dict = {"name": "conda_env", "channels": ["conda-forge"], "dependencies": ["python=3.10.0"]}
   conda_env = rh.env(conda_env=conda_dict, name="dict_env")

3. Name of an existing conda env on your local machine

::

   conda_env = rh.conda_env(conda_env="local_conda_env", name="from_local_env")

4. Leaving the argument empty. In this case, we’ll construct a new Conda
   environment for you, using the list you pass into ``reqs``.

::

   conda_env = rh.conda_env(reqs=["numpy", "diffusers"], name="new_env")

Beyond the conda config, you can also add any additional requirements
you’d like to install in the environment by adding
``reqs = List[packages]``.

.. code:: ipython3

    conda_env = rh.conda_env(reqs=["numpy", "diffusers"], name="new_env")

As with the base env, we can set up a conda env on the cluster with
(note, this command might appear to hang, but it may be updating conda
in the backgroud for a few minutes the first time you run it):

.. code:: ipython3

    conda_env_on_cluster = conda_env.to(system=cluster)

Previously in the cluster section, we mentioned several cluster APIs
such as running CLI or Python commands. These all run on the base
environment in the examples above, but now that we’ve defined a Conda
env, let’s demonstrate how we can accomplish this inside a Conda env on
the cluster:

.. code:: ipython3

    # run Python command within the conda env
    cluster.run_python(['import numpy', 'print(numpy.__version__)'], env=conda_env)


.. parsed-literal::

    Warning: Permanently added '3.83.88.203' (ED25519) to the list of known hosts.


.. parsed-literal::

    1.26.3


.. code:: ipython3

    # install additional package on given env
    cluster.install_packages(["pandas"], env=conda_env)

Putting it all together – Cluster, Function/Module, Env
-------------------------------------------------------

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

.. code:: ipython3

    list_a = [1, 2, 3]
    list_b = [2, 3, 4]
    add_lists_remote(list_a, list_b)


.. parsed-literal::

    INFO | 2024-01-12 16:52:00.149572 | Calling add_lists.call


.. parsed-literal::

    base_env servlet: Calling method call on module add_lists


.. parsed-literal::

    INFO | 2024-01-12 16:52:00.433690 | Time to call add_lists.call: 0.28 seconds


.. parsed-literal::

    array([3, 5, 7])



Now that you understand the basics, feel free to play around with more
complicated scenarios! You can also check out our additional API and
example usage tutorials on our `docs
site <https://www.run.house/docs>`__.

Cluster Termination
-------------------

To terminate the cluster, you can call ``sky down cluster-name`` in CLI
or ``cluster_obj.teardown()`` in Python.

.. code:: ipython3

    !sky down cpu-cluster
    # or
    cluster.teardown()
