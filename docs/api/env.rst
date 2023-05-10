Env
====================================
An Env is a Runhouse primitive that represents an compute environment.


Env Factory Method
~~~~~~~~~~~~~~~~~~~

.. autofunction:: runhouse.env

Env Class
~~~~~~~~~~

.. autoclass:: runhouse.Env
   :members:
   :exclude-members:

    .. automethod:: __init__

Conda Env Class
~~~~~~~~~~~~~~~

.. autoclass:: runhouse.CondaEnv
    :members:
    :exclude-members:

      .. automethod:: __init__


API Usage
~~~~~~~~~

To initalize an Env, use ``rh.env``.

.. code:: python

    env = rh.env(name="env_from_reqs", reqs="requirements.txt")
    # or
    env = rh.env(name="env_from_list", reqs=['torch', 'numpy', 'diffusers'])

To create a CondaEnv, additionally pass in a ``conda_env`` argument, which can take in a local path to a conda yaml file,
a dict corresponding to the conda environment, or a local conda environment name. You can also add additional packages to
install on top of the environment using the ``reqs`` parameter.

.. code:: python

    # from a yaml file, such as `conda env export` output
    conda_env = rh.env(conda_env="conda_env.yaml", reqs=["pip:diffusers"])
    # using a dict corresponding to the conda env
    conda_env = rh.env(conda_env={"name": "new-conda-env", "channels": ["defaults"], "dependencies": "pip", {"pip": "diffusers"})
    # from a local conda env
    conda_env = rh.env(conda_env="local-conda-env-name")


To setup and use an environment on a cluster for running a Function:

.. code:: python

    env = rh.env(...)
    gpu = rh.cluster(name="rh-a10x")

    rh_func = rh.function(name="sd_generate")
    rh_func.to(gpu, env)

You can also directly send and setup the environment on a cluster using:

.. code:: python

    env.to(gpu)

To install packages on the environment on the system at any point:

.. code:: python

    to_install = ["pip:diffusers"]
    system.install_packages(to_install, env)
