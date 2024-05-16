Functions and Modules
=====================

.. raw:: html

    <p><a href="https://colab.research.google.com/github/run-house/notebooks/blob/stable/docs/api-modules.ipynb">
    <img height="20px" width="117px" src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a></p>

Runhouse makes Python functions and modules portable. Runhouse functions
and modules are wrappers around Python code for functions and classes,
that can live on remote compute and be run remotely. Once constructed,
they can be called natively in Python from your local environment, and
they come with a suite of built-in, ready-to-use features like logging,
streaming, and mapping.

Setup
-----

We first construct a Runhouse Cluster resource, which is the compute to
which we will be sending and running our remote Python code on. You can
read more in the `Cluster
tutorial <https://www.run.house/docs/tutorials/api-clusters>`__.

.. code:: ipython3

    import runhouse as rh

.. code:: ipython3

    cluster = rh.cluster(
        name="rh-cluster",
        instance_type="CPU:2+",
        provider="aws",
    )
    cluster.up_if_not()

Runhouse Functions
------------------

A Runhouse Function wraps a function, and can be send to remote hardware
to be run as a subroutine or service.

Let’s start by defining a Python function locally. This function uses
the ``numpy`` package to return the sum of the two input arguments.

.. code:: ipython3

    def np_sum(a, b):
        import numpy as np
        return np.sum([a, b])

We set up the function on the cluster by

-  wrapping it with ``rh.function(np_env)``
-  sending it ``.to(cluster)``
-  specifying dependencies with ``env=["numpy"]``

When this is called, the underlying code is synced over and dependencies
are set up.

.. code:: ipython3

    remote_np_sum = rh.function(np_sum).to(cluster, env=["numpy"])


.. parsed-literal::
    :class: code-output

    INFO | 2024-02-27 20:21:54.329646 | Because this function is defined in a notebook, writing it out to a file to make it importable. Please make sure the function does not rely on any local variables, including imports (which should be moved inside the function body). Functions defined in Python files can be used normally.
    INFO | 2024-02-27 20:21:55.378194 | Server rh-cluster is up.
    INFO | 2024-02-27 20:21:55.384844 | Copying package from file:///Users/caroline/Documents/runhouse/notebooks to: rh-cluster
    INFO | 2024-02-27 20:22:06.614361 | Calling base_env.install


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

    INFO | 2024-02-27 20:22:09.486367 | Time to call base_env.install: 2.87 seconds
    INFO | 2024-02-27 20:22:18.091062 | Sending module np_sum to rh-cluster


Running the function remotely is as simple as if you were running it
locally. Below, the function runs remotely on the cluster, and returns
the results to your local environment.

.. code:: ipython3

    remote_np_sum(1, 5)


.. parsed-literal::
    :class: code-output

    INFO | 2024-02-27 20:49:41.688705 | Calling np_sum.call
    INFO | 2024-02-27 20:49:42.944473 | Time to call np_sum.call: 1.26 seconds




.. parsed-literal::
    :class: code-output

    6



Runhouse Modules
----------------

A Function is a subclass of a more generic Runhouse concept called a
Module, which represents the class analogue to a function. Like a
Function, you can send a Module to a remote cluster and interact with it
natively by calling its methods, but it can also persist and utilize
live state via instance methods.

Introducing state into a service means being able to spin up, connect,
and secure auxiliary services like Redis, Celery, etc. In Runhouse,
state is built in, and lives natively in-memory in Python so it’s
ridiculously fast.

Converting Existing Class to Runhouse Module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you have a native Python class that you would like to run remotely,
you can directly convert it into a Runhouse Module via the ``rh.module``
factory function.

-  Pass in the Python class to ``rh.module()``
-  Call ``.to(cluster)`` to sync the class across to the cluster
-  Create a class instance and call their functions just as you would a
   locally defined class. The function runs remotely, and returns the
   result locally.

.. code:: ipython3

    from transformers import AutoModel

    RemoteModel = rh.module(AutoModel).to(my_gpu)
    remote_model = AutoModel.from_pretrained("google-bert/bert-base-uncased")
    remote_model.predict()

Constructing your own rh.Module Class
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can also construct a Module from scratch by subclassing
``rh.Module``.

Note that the class is constructed locally prior to sending it to a
remote cluster. If there is a computationally heavy operation such as
loading a dataset or model that you only want to take place remotely,
you probably want to wrap that operation in an instance method and call
it only after it’s sent to remote compute. One such way is through lazy
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
    :class: code-output

    Writing pid_module.py


We can directly import the Module, and call ``.to(cluster)`` on it. Then
use it as you would with any local Python class, except that this it is
being run on the cluster.

.. code:: ipython3

    from pid_module import PIDModule

    remote_module = PIDModule(a=5).to(cluster)
    remote_module.getpid()


.. parsed-literal::
    :class: code-output

    INFO | 2024-02-27 20:56:19.187985 | Copying package from file:///Users/caroline/Documents/runhouse/notebooks to: rh-cluster
    INFO | 2024-02-27 20:56:24.220264 | Calling base_env.install


.. parsed-literal::
    :class: code-output

    [36mInstalling Package: notebooks with method reqs.
    [0m[36mreqs path: notebooks/requirements.txt
    [0m[36mnotebooks/requirements.txt not found, skipping
    [0m

.. parsed-literal::
    :class: code-output

    INFO | 2024-02-27 20:56:25.343078 | Time to call base_env.install: 1.12 seconds
    INFO | 2024-02-27 20:56:35.126382 | Sending module PIDModule to rh-cluster
    INFO | 2024-02-27 20:56:44.887485 | Calling PIDModule.getpid
    INFO | 2024-02-27 20:56:45.938380 | Time to call PIDModule.getpid: 1.05 seconds




.. parsed-literal::
    :class: code-output

    31607
