Local Quick Start
=================

.. raw:: html

    <p><a href="https://colab.research.google.com/github/run-house/notebooks/blob/stable/docs/quick-start-local.ipynb">
    <img height="20px" width="117px" src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a></p>

In `Cloud Quick Start <https://www.run.house/docs/tutorials/quick-start-cloud>`__,
we demonstrate how to deploy a local function to a remote cluster using
Runhouse. In this local-only version, we show how to use Runhouse to set
up a local web server, and deploy an arbitrary Python function to it.

Runhouse Server Setup
---------------------

First install Runhouse with ``pip install runhouse``

.. code:: ipython3

    !pip install runhouse

Next, start the Runhouse server locally on CLI with
``runhouse restart``, and use ``runhouse status`` to print the status
and details of the server. For printing cluster's status outside the cluser, its name should provided as well: ``runhouse status <cluster_name>``.

.. code:: ipython3

    !runhouse restart

.. code:: ipython3

    !runhouse status

*CPU cluster*

.. parsed-literal::
    :class: code-output

    /sashab/rh-basic-cpu
    😈 Runhouse Daemon is running 🏃
    Runhouse v0.0.28
    server pid: 29395
    • server port: 32300
    • den auth: True
    • server connection type: ssh
    • backend config:
      • resource subtype: OnDemandCluster
      • domain: None
      • server host: 0.0.0.0
      • ips: ['52.207.212.159']
      • resource subtype: OnDemandCluster
      • autostop mins: autostop disabled
    Serving 🍦 :
    • _cluster_default_env (runhouse.Env)
      This environment has only python packages installed, if such provided. No resources were found.
    • sd_env (runhouse.Env) | pid: 29716 | node: head (52.207.212.159)
      CPU: 0.0% | Memory: 0.13 / 8 Gb (1.65%)
      This environment has only python packages installed, if such provided. No resources were found.
    • np_pd_env (runhouse.Env) | pid: 29578 | node: head (52.207.212.159)
      CPU: 0.0% | Memory: 0.13 / 8 Gb (1.71%)
      • /sashab/summer (runhouse.Function)
      • mult (runhouse.Function)

*GPU cluster*

.. parsed-literal::
    :class: code-output

    /sashab/rh-basic-gpu
    😈 Runhouse Daemon is running 🏃
    Runhouse v0.0.28
    server pid: 29486
    • server port: 32300
    • den auth: True
    • server connection type: ssh
    • backend config:
      • resource subtype: OnDemandCluster
      • domain: None
      • server host: 0.0.0.0
      • ips: ['35.171.157.49']
      • resource subtype: OnDemandCluster
      • autostop mins: autostop disabled
    Serving 🍦 :
    • _cluster_default_env (runhouse.Env)
      This environment has only python packages installed, if such provided. No resources were found.
    • np_pd_env (runhouse.Env) | pid: 29672 | node: head (35.171.157.49)
      CPU: 0.0% | Memory: 0.13 / 16 Gb (0.85%)
      • /sashab/summer (runhouse.Function)
      • mult (runhouse.Function)
    • sd_env (runhouse.Env) | pid: 29812 | node: head (35.171.157.49)
      CPU: 1.0% | Memory: 4.47 / 16 Gb (28.95%)
      GPU: 0.0% | Memory: 6.89 / 23 Gb (29.96%)
      • sd_generate (runhouse.Function)

Local Python Function
---------------------

Let’s first define a simple Python function that we want to send to the
server. This function returns the process ID it runs on, and optionally
takes in a parameter, which it adds to the process ID prior to returning
it.

.. code:: ipython3

    def get_pid(a=0):
        import os
        return os.getpid() + int(a)

Deployment
----------

Standing up your Python code on the server is simple with the Runhouse
API. Wrap the function with ``rh.function``, and then use
``.to(rh.here)`` to sync it to the server.

.. note::

   Make sure that any code in your Python file that’s meant to only run
   locally is placed within a ``if __name__ == "__main__":`` block.
   Otherwise, that code will run when Runhouse attempts to import your
   code remotely. For example, you wouldn’t want
   ``function.to(rh.here)`` to run again on the server. This is not
   necessary when using a notebook. Please see our `examples
   directory <https://github.com/run-house/runhouse/tree/main/examples>`__
   for implementation details.

.. code:: ipython3

    import runhouse as rh

.. code:: ipython3

    server_fn = rh.function(get_pid).to(rh.here)


.. parsed-literal::
    :class: code-output

    INFO | 2024-02-26 22:14:53.460361 | Because this function is defined in a notebook, writing it out to a file to make it importable. Please make sure the function does not rely on any local variables, including imports (which should be moved inside the function body). Functions defined in Python files can be used normally.
    INFO | 2024-02-26 22:14:53.523591 | Sending module get_pid to local Runhouse daemon


The ``get_pid`` function we defined above now exists on the server.

Remote Function Call
~~~~~~~~~~~~~~~~~~~~

You can call the server function just as you would any other Python
function, with ``server_fn()``, and it runs on the server and returns
the result to our local environment.

Below, we run both the local and server versions of this function, which
give different results and confirms that the functions are indeed being
run on different processes.

.. code:: ipython3

    print(f"Local PID {get_pid()}")
    print(f"Server PID {server_fn()}")


.. parsed-literal::
    :class: code-output

    Local PID 27818
    Server PID 19846


HTTP Endpoint and Curl
~~~~~~~~~~~~~~~~~~~~~~

In addition to calling the function directly in Python, we can also
access it with a curl call or open it up in a browser.

.. code:: ipython3

    server_fn.endpoint()




.. parsed-literal::
    :class: code-output

    'http://0.0.0.0:32300/get_pid'



.. code:: ipython3

    !curl "http://0.0.0.0:32300/get_pid/call"


.. parsed-literal::
    :class: code-output

    {"data":"19846","error":null,"traceback":null,"output_type":"result_serialized","serialization":"json"}

To pass in the optional function parameter:

.. code:: ipython3

    !curl "http://0.0.0.0:32300/get_pid/call?a=1"


.. parsed-literal::
    :class: code-output

    {"data":"19847","error":null,"traceback":null,"output_type":"result_serialized","serialization":"json"}
