Local Quick Start
=================

.. raw:: html

    <p><a href="https://colab.research.google.com/github/run-house/notebooks/blob/stable/docs/getting_started/local_quick_start.ipynb">
    <img height="20px" width="117px" src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a></p>

In `Cloud Quick
Start <run.house/docs/tutorials/quick_start_cloud>`__,
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
and details of the server.

.. code:: ipython3

    !runhouse restart

.. code:: ipython3

    !runhouse status


.. parsed-literal::
    :class: code-output

    [1;38;5;63m😈 Runhouse Daemon is running 🏃[0m
    • server_port: [1;36m32300[0m
    • den_auth: [3;91mFalse[0m
    • server_connection_type: none
    • backend config:
            • use_local_telemetry: [3;91mFalse[0m
            • domain: [3;35mNone[0m
            • server_host: [1;92m0.0.0.0[0m
            • ips: [1m[[0m[32m'0.0.0.0'[0m[1m][0m
            • resource_subtype: Cluster
    [1mServing 🍦 :[0m
    [3;4mbase [0m[1;3;4m([0m[3;4mEnv[0m[1;3;4m)[0m[3;4m:[0m
    This environment has no resources.
    [0m

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

.. code:: ipython3

    import runhouse as rh

.. code:: ipython3

    server_fn = rh.function(get_pid).to(rh.here)


.. parsed-literal::
    :class: code-output

    INFO | 2024-02-26 22:14:53.460361 | Writing out function to /Users/caroline/Documents/runhouse/notebooks/docs/getting_started/get_pid_fn.py. Please make sure the function does not rely on any local variables, including imports (which should be moved inside the function body).
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
