Envs and Packages
=================

.. raw:: html

    <p><a href="https://colab.research.google.com/github/run-house/notebooks/blob/stable/docs/api-envs.ipynb">
    <img height="20px" width="117px" src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a></p>

The Runhouse Env and Package abstractions help to provide convenient
dependency isolation and management across your dev environments and
applications. By specifying the runtime environment associated with each
of your Runhouse functions and apps, ensure consistency and
reproducibility no matter where you deploy your code from/to.

Packages
--------

A Runhouse package represents a package or dependency that can be shared
between environments/clusters or file storage, and is core to the
Runhouse environment. This can be the standard PyPI or Conda package, a
requirements.txt file, a custom local package, or even a Git package.

.. code:: ipython3

    import runhouse as rh

.. code:: ipython3

    pip_package = rh.Package.from_string("pip:numpy")
    conda_package = rh.Package.from_string("conda:torch")
    reqs_package = rh.Package.from_string("reqs:./")
    git_package = rh.GitPackage(git_url='https://github.com/huggingface/diffusers.git',
                                install_method='pip',
                                revision='v0.11.1')

Envs
----

The Runhouse environment represents a whole compute environment,
consisting of packages, environment variables, and any secrets necessary
for performing tasks within the environment. It defines the environment
on which Runhouse functions and modules run.

Currently, both bare metal environments and Conda environments are
supported. Docker environment support is planned.

Bare Metal Envs
~~~~~~~~~~~~~~~

Envs can be constructed with the ``rh.env()`` factory function,
optionally taking in a name, requirements (packages), environment
variables, and secrets.

.. code:: ipython3

    env = rh.env(
            name="fn_env",
            reqs=["numpy", "torch"],
            env_vars={"USER": "*****"},
            secrets=["aws"],
    )

If no environment name is provided, when the environment is sent to a
cluster, the dependencies and variables of the environment will be
installed and synced on top of the cluster’s default env. However,
Without a name, the env resource itself can not be accessed and does not
live in the cluster’s object store.

Conda Envs
~~~~~~~~~~

Conda Envs can be created using ``rh.conda_env``. There are a couple of
ways to construct a Conda Env:

-  ``.yml`` file corresponding to conda config
-  dict corresponding to conda config
-  name of already set up local conda env
-  passing in reqs as a list

Additional package dependencies can be passed in through the ``reqs``
argument, and env vars, secrets, and working dir is supported just as in
the bare metal env.

.. code:: ipython3

    conda_env = rh.conda_env(conda_env="conda_env.yml", reqs=["numpy", "diffusers"], name="yaml_env")

    conda_dict = {"name": "conda_env", "channels": ["conda-forge"], "dependencies": ["python=3.10.0"]}
    conda_env = rh.env(conda_env=conda_dict, name="dict_env")

    conda_env = rh.conda_env(conda_env="local_conda_env", name="from_local_env")

    conda_env = rh.conda_env(reqs=["numpy", "diffusers"], name="new_env")

Envs on the Cluster
~~~~~~~~~~~~~~~~~~~

Runhouse environments are generic environments, and the object itself is
not associated with a cluster. However, it is easy to set up an
environment on the cluster, by simply calling the ``env.to(cluster)``
API, or by sending your module/function to the env with the
``<rh_fn>.to(cluster=cluster, env=env)`` API, which will construct and
cache the environment on the remote cluster.

.. code:: ipython3

    # Function, cluster, and env setup
    def np_sum(a, b):
        import numpy as np
        return np.sum([a, b])

    cluster = rh.ondemand_cluster("rh-cluster", instance_type="CPU:2+").up_if_not()
    env = rh.env(name="np_env", reqs=["numpy"])

.. code:: ipython3

    remote_np_sum = rh.function(np_sum).to(cluster, env=env)


.. parsed-literal::
    :class: code-output

    INFO | 2024-02-28 21:24:52.915177 | Writing out function to /Users/caroline/Documents/runhouse/notebooks/docs/np_sum_fn.py. Please make sure the function does not rely on any local variables, including imports (which should be moved inside the function body).
    INFO | 2024-02-28 21:25:03.923658 | SSH tunnel on to server's port 32300 via server's ssh port 22 already created with the cluster.
    INFO | 2024-02-28 21:25:04.162828 | Server rh-cluster is up.
    INFO | 2024-02-28 21:25:04.166104 | Copying package from file:///Users/caroline/Documents/runhouse/notebooks to: rh-cluster


.. parsed-literal::
    :class: code-output

    INFO | 2024-02-28 21:25:07.356780 | Calling np_env.install


.. parsed-literal::
    :class: code-output

    ----------
    [36mrh-cluster[0m
    ----------
    [36mInstalling Package: numpy with method pip.
    [0m[36mRunning: pip install numpy
    [0m[36mInstalling Package: notebooks with method reqs.
    [0m[36mreqs path: notebooks/requirements.txt
    [0m[36mnotebooks/requirements.txt not found, skipping
    [0m

.. parsed-literal::
    :class: code-output

    INFO | 2024-02-28 21:25:09.601131 | Time to call np_env.install: 2.24 seconds


.. parsed-literal::
    :class: code-output

    INFO | 2024-02-28 21:25:16.987243 | Sending module np_sum to rh-cluster


.. code:: ipython3

    remote_np_sum(2, 3)


.. parsed-literal::
    :class: code-output

    INFO | 2024-02-28 21:38:18.997808 | Calling np_sum.call
    INFO | 2024-02-28 21:38:20.047907 | Time to call np_sum.call: 1.05 seconds




.. parsed-literal::
    :class: code-output

    5



On the cluster, each environment is associated with its own Ray Actor
servlet, which handles all the activities within the environment
(installing packages, getting or putting objects, calling functions,
etc). Each env servlet has its own local object store where objects
persist in Python, and lives in its own process, reducing interprocess
overhead and eliminating launch overhead for calls made in the same env.

Syncing your local code
~~~~~~~~~~~~~~~~~~~~~~~

You may be wondering how the actual code that you have written and sent
to Runhouse gets synced to the cluster, if it is not included in the
env. When you import a function and send it to the env, we locate the
function’s import site and find the package it’s a part of. We do this
by searching for any “.git”, “setup.py”, “setup.cfg”, “pyproject.toml”,
or “requirements.txt”, and then sync the first directory we find that
represents a package. Any directory with a ``requirements.txt`` that is
synced up will also have those reqs installed. *We do not store this
code on our servers at all, it is just synced onto your own cluster.*

You can also sync a specific folder of your own choosing, and it will be
synced and added to the remote Python path, resulting in any Python
packages in that directory being importable. For example:

.. code:: ipython3

    env = rh.env(
            name="fn_env_with_local_package",
            reqs=["numpy", "torch", "~/path/to/package"],
    )

Cluster Default Env
^^^^^^^^^^^^^^^^^^^

The cluster also has a concept of a base default env, which is the
environment on which the runhouse server will be started from. It is the
environment in which cluster calls and computations, such as commands
and functions, will default to running on, if no other env is specified.

During cluster initialization, you can specify the default env for the
cluster. It is constructed as with any other runhouse env, using
``rh.env()``, and contains any package installations, commands to run,
or env vars to set prior to starting the Runhouse server, or even a
particular conda env to isolate your Runhouse environment. If no default
env is specified, runs on the base environment on the cluster (after
sourcing bash).

.. code:: ipython3

    import runhouse as rh

.. code:: ipython3

    default_env = rh.conda_env(
        name="cluster_default",
        reqs=["skypilot"],  # to enable autostop, which requires skypilot library
        env_vars={"my_token": "TOKEN_VAL"}
    )
    cluster = rh.ondemand_cluster(
        name="rh-cpu",
        instance_type="CPU:2+",
        provider="aws",
        default_env=default_env,
    )
    cluster.up_if_not()

Now, as we see in the examples below, running a command or sending over
a function without specifying an env will default the default conda env
that we have specified for the cluster.

.. code:: ipython3

    cluster.run("conda env list | grep '*'")


.. parsed-literal::
    :class: code-output

    INFO | 2024-05-20 18:08:42.460946 | Calling cluster_default._run_command


.. parsed-literal::
    :class: code-output

    [36mRunning command in cluster_default: conda run -n cluster_default conda env list | grep '*'
    [0m[36mcluster_default       *  /opt/conda/envs/cluster_default
    [0m

.. parsed-literal::
    :class: code-output

    INFO | 2024-05-20 18:08:45.130137 | Time to call cluster_default._run_command: 2.67 seconds




.. parsed-literal::
    :class: code-output

    [(0, 'cluster_default       *  /opt/conda/envs/cluster_default\n', '')]



.. code:: ipython3

    def check_import():
        import sky
        return "import succeeded"

.. code:: ipython3

    check_remote_import = rh.function(check_import).to(cluster)

.. code:: ipython3

    check_remote_import()


.. parsed-literal::
    :class: code-output

    INFO | 2024-05-20 18:30:05.128009 | Calling check_import.call
    INFO | 2024-05-20 18:30:05.691348 | Time to call check_import.call: 0.56 seconds




.. parsed-literal::
    :class: code-output

    'import succeeded'
