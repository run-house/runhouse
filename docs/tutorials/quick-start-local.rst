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
    Runhouse v0.0.34
    🤖 aws m6i.large cluster | 🌍 us-east-1 | 💸 $0.096/hr
    server pid: 29477
    • server port: 32300
    • den auth: False
    • server connection type: ssh
    • backend config:
        • resource subtype: OnDemandCluster
        • domain: None
        • server host: 0.0.0.0
        • ips: ['52.91.194.125']
        • autostop mins: autostop disabled
    CPU Utilization: 5.4%
    Serving 🍦 :
    • _cluster_default_env (runhouse.Env)
        This environment has only python packages installed, if provided. No resources were found.
    • np_pd_env (runhouse.Env) | pid: 29621 | node: head (52.91.194.125)
        CPU: 0.3% | Memory: 0.1 / 8 Gb (0.01%)
        • np_pd_env (runhouse.Env)
        • summer (runhouse.Function) Currently not running
        • mult (runhouse.Function) Running for 2.484918 seconds


*GPU cluster*

.. parsed-literal::
    :class: code-output

    /sashab/rh-basic-gpu
    😈 Runhouse Daemon is running 🏃
    Runhouse v0.0.34
    🤖 aws g5.xlarge cluster | 🌍 us-east-1 | 💰 $1.006/hr
    server pid: 29657
    • server port: 32300
    • den auth: False
    • server connection type: ssh
    • backend config:
        • resource subtype: OnDemandCluster
        • domain: None
        • server host: 0.0.0.0
        • ips: ['3.92.223.118']
        • autostop mins: autostop disabled
    CPU Utilization: 12.8% | GPU Utilization: 7.07%
    Serving 🍦 :
    • _cluster_default_env (runhouse.Env)
        This environment has only python packages installed, if provided. No resources were found.
    • np_pd_env (runhouse.Env) | pid: 29809 | node: head (3.92.223.118)
        CPU: 0.4% | Memory: 0.1 / 16 Gb (0.01%)
        • np_pd_env (runhouse.Env)
        • summer (runhouse.Function) Currently not running
        • mult (runhouse.Function) Currently not running
    • sd_env (runhouse.Env) | pid: 32054 | node: head (3.92.223.118)
        CPU: 40.1% | Memory: 2.87 / 16 Gb (0.19%)
        GPU Memory: 3.38 / 23 Gb (14.7%)
        • sd_env (runhouse.Env)
        • sd_generate (runhouse.Function) Running for 26.578614 seconds


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
