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


API Usage
~~~~~~~~~

To initalize an Env, use ``rh.env``:

.. code:: python
    # Python env
    python_env = rh.env(name="pyenv_from_list", py_env="requirements.txt")
    # or
    python_env = rh.env(name="pyenv_from_list", py_env=['torch', 'numpy', 'diffusers'])
To setup and use an environment on a cluster, or for running a Function:

.. code:: python
    env = rh.env(...)
    gpu = rh.cluster(name="rh-a10x")
    env.to(gpu)
    rh_func = rh.function(name="sd_generate")
    rh_func.to(env)  # equivalent to env.to(rh_func.system), when running the function
