Envs and Packages
=================

.. raw:: html

    <p><a href="https://colab.research.google.com/github/run-house/notebooks/blob/stable/docs/api_envs.ipynb">
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
consisting of packages, environment variables, the working directory,
and any secrets necessary for performing tasks within the environment.
It defines the environment on which Runhouse functions and modules run.

Currently, both bare metal environments and Conda environments are
supported. Docker environment support is planned.

Bare Metal Envs
~~~~~~~~~~~~~~~

Envs can be constructed with the ``rh.env()`` factory function,
optionally taking in a name, requirements (packages), environment
variables, secrets, and working directory.

.. code:: ipython3

    env = rh.env(
            name="fn_env",
            reqs=["numpy", "torch"],
            working_dir="./",  # current working dir
            env_vars={"USER": "*****"},
            secrets=["aws"],
    )

If no environment name is provided, it defaults to ``"base_env"``, which
corresponds to the base, catch-all environment on the cluster. If
multiple “base_env” environments are sent to a cluster, the dependencies
and variables will continue to be synced on top of the existing base
environment.

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
not associated with a cluster. However, it is a core component of
Runhouse services, like functions and modules, which are associated with
a cluster. As such, it is set up remotely when these services are sent
over to the cluster – packags are installed, working directory and env
vars/secrets synced over, and cached on the cluster.

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
