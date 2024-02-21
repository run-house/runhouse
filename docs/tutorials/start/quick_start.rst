Getting Started
===============

.. raw:: html

    <p><a href="https://colab.research.google.com/github/run-house/runhouse/blob/stable/docs/notebooks/start/quick_start.ipynb">
    <img height="20px" width="117px" src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a></p>

In this basic example, we demonstrate how to use Runhouse to set up a
local server and deploy your own Python application to it.

Runhouse Server Setup
---------------------

First install runhouse with ``pip install runhouse``.

.. code:: ipython3

    !pip install runhouse

Next, start the runhouse server locally on CLI with
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

Run your Local Service
----------------------

Standing up your Python code on the server is simple with the Runhouse
API. Simply wrap it with ``rh.function`` or ``rh.module`` for a Python
function or class, respectively, and then sync it to the server using
``.to(rh.here)``.

For more specifics on the Runhouse API, please refer to the API
tutorials or API reference on `Runhouse
docs <https://www.run.house/docs>`__.

Sample Function: ``get_pid``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For our example, we use a simple Python function that returns the
process ID. It optionally takes in a parameter, which it adds to the
process ID prior to returning it.

.. code:: ipython3

    import runhouse as rh

.. code:: ipython3

    def get_pid(a=0):
        import os
        return os.getpid() + int(a)

Sync ``get_pid`` to the server
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following code sends our Python function, ``get_pid`` to

.. code:: ipython3

    server_get_pid = rh.function(get_pid).to(rh.here)


.. parsed-literal::
    :class: code-output

    INFO | 2024-02-21 19:12:40.047825 | Writing out function to /Users/caroline/Documents/runhouse/runhouse/docs/notebooks/start/get_pid_fn.py. Please make sure the function does not rely on any local variables, including imports (which should be moved inside the function body).
    INFO | 2024-02-21 19:12:40.083022 | Sending module get_pid to local Runhouse daemon


Remote Function Call
~~~~~~~~~~~~~~~~~~~~

We can call the function from the server just as you would a regular
Python function. As you can see in the following code, the regular
Python function ``get_pid()`` returns a different process ID than the
server ``server_get_pid()``.

.. code:: ipython3

    print(f"Local PID {get_pid()}")
    print(f"Server PID {server_get_pid()}")


.. parsed-literal::
    :class: code-output

    Local PID 3295
    Daemon PID 3391


HTTP Endpoint and Curl
~~~~~~~~~~~~~~~~~~~~~~

Here, we extract the function endpoint, and use it in a curl call. You
can also pass in variables to the curl call, or paste the http link in
your browser and see the result.

.. code:: ipython3

    server_get_pid.endpoint()


.. parsed-literal::
    :class: code-output

    'http://0.0.0.0:32300/get_pid'



.. code:: ipython3

    !curl "http://0.0.0.0:32300/get_pid/call"


.. parsed-literal::
    :class: code-output

    {"data":"3391","error":null,"traceback":null,"output_type":"result_serialized","serialization":"json"}

.. code:: ipython3

    !curl "http://0.0.0.0:32300/get_pid/call?a=1"


.. parsed-literal::
    :class: code-output

    {"data":"3392","error":null,"traceback":null,"output_type":"result_serialized","serialization":"json"}
