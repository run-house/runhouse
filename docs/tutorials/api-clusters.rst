Clusters
========

.. raw:: html

    <p><a href="https://colab.research.google.com/github/run-house/notebooks/blob/stable/docs/api-clusters.ipynb">
    <img height="20px" width="117px" src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a></p>

A cluster is the most basic form of compute in Runhouse, largely
representing a group of instances or VMs connected with Ray. They
largely fall in two categories:

1. *Static Clusters*: Any machine you have SSH access to, set up with IP
   addresses and SSH credentials.
2. *On-Demand Clusters*: Any cloud instance spun up automatically for
   you with your cloud credentials.

Runhouse provides various APIs for interacting with remote clusters,
such as terminating an on-demand cloud cluster or running remote CLI or
Python commands from your local dev environment.

Let’s start with a simple example using AWS. First, install ``runhouse``
with AWS dependencies:

.. code:: ipython3

    ! pip install "runhouse[aws]"

Make sure your AWS credentials are set up:

.. code:: ipython3

    ! aws configure
    ! sky check

On-Demand Clusters
------------------

We can start by using the ``rh.cluster`` factory function to create our
cluster. By specifying an ``instance_type``, Runhouse sets up an
On-Demand Cluster in AWS EC2 for us.

Each cluster must be provided with a unique ``name`` identifier during
construction. This ``name`` parameter is used for saving down or loading
previous saved clusters, and also used for various CLI commands for the
cluster.

Our ``instance_type`` here is defined as ``CPU:2``, which is the
accelerator type and count that we need (another example would be
``A10G:2``). We could alternatively specify a specific specific instance
type, such as ``p3.2xlarge`` or ``g4dn.xlarge`` (these are instance
types on AWS).

.. code:: ipython3

    import runhouse as rh

    aws_cluster = rh.cluster(name="test-cluster", instance_type="CPU:2")
    aws_cluster.up_if_not()

Next, we set up a basic function to throw up on our cluster. For more
information about Functions & Modules that you can put up on a cluster,
see `Functions &
Modules <https://www.run.house/docs/tutorials/api-modules>`__.

.. code:: ipython3

    def run_home(name: str):
        return f"Run home {name}!"

    remote_function = rh.function(run_home).to(aws_cluster)

After running ``.to``, your function is set up on the cluster to be
called from anywhere. When you call ``remote_function``, it executes
remotely on your AWS instance.

.. code:: ipython3

    remote_function("in cluster!")


.. parsed-literal::
    :class: code-output

    INFO | 2024-03-06 15:18:58.439252 | Calling run_home.call
    INFO | 2024-03-06 15:18:59.490122 | Time to call run_home.call: 1.05 seconds




.. parsed-literal::
    :class: code-output

    'Run home in cluster!!'



On-Demand Clusters with TLS exposed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the previous example, the cluster that was brought up in EC2 is only
accessible to the original user that has SSH credentials to the machine.
However, you can set up a cluster with ports exposed to open Internet,
and access objects and functions via ``curl``.

.. code:: ipython3

    tls_cluster = rh.cluster(name="tls-cluster",
                             instance_type="CPU:2",
                             open_ports=[443], # expose HTTPS port to public
                             server_connection_type="tls", # specify how runhouse communicates with this cluster
                             den_auth=False, # no authentication required to hit this cluster (NOT recommended)
    ).up_if_not()


.. parsed-literal::
    :class: code-output

    WARNING | 2024-03-06 15:19:05.297411 | /Users/rohinbhasin/work/runhouse/runhouse/resources/hardware/on_demand_cluster.py:317: UserWarning: Server is insecure and must be inside a VPC or have `den_auth` enabled to secure it.
      warnings.warn(



.. code:: ipython3

    remote_tls_function = rh.function(run_home).to(tls_cluster)

.. code:: ipython3

    remote_tls_function("Marvin")


.. parsed-literal::
    :class: code-output

    INFO | 2024-03-06 15:26:05.482586 | Calling run_home.call
    INFO | 2024-03-06 15:26:06.550625 | Time to call run_home.call: 1.07 seconds




.. parsed-literal::
    :class: code-output

    'Run home Marvin!'



.. code:: ipython3

    tls_cluster.address




.. parsed-literal::
    :class: code-output

    '54.172.178.196'



.. code:: ipython3

    ! curl "https://54.172.178.196/run_home/call?name=Marvin" -k


.. parsed-literal::
    :class: code-output

    {"data":"\"Run home Marvin!\"","error":null,"traceback":null,"output_type":"result_serialized","serialization":"json"}

This cluster is exposed to the open Internet, so anyone can hit it. If
you do want to share functions and apps publically, it’s recommended you
set ``den_auth=True`` when setting up your cluster, which requires a
user to run ``runhouse login`` in order to hit the cluster. We’ll enable
it now:

.. code:: ipython3

    tls_cluster.enable_den_auth()

.. code:: ipython3

    ! curl "https://54.172.178.196/run_home/call?name=Marvin" -k


.. parsed-literal::
    :class: code-output

    {"data":null,"error":raise PermissionError(\\nPermissionError: No Runhouse token provided. Try running `$ runhouse login` or visiting https://run.house/login to retrieve a token. If calling via HTTP, please provide a valid token in the Authorization header.\\n\"","output_type":"exception","serialization":null}

If we send our Runhouse Den token as a header, then the request is
valid:

.. code:: ipython3

    ! curl "https://54.172.178.196/run_home/call?name=Marvin" -k -H "Authorization: Bearer <YOUR TOKEN HERE>"


.. parsed-literal::
    :class: code-output

    {"data":"\"Run home Marvin!\"","error":null,"traceback":null,"output_type":"result_serialized","serialization":"json"}

Static Clusters
---------------

If you have existing machines within a VPC that you want to connect to,
you can simply provide the IP addresses and path to SSH credentials to
the machine.

.. code:: ipython3

    cluster = rh.cluster(  # using private key
                  name="cpu-cluster-existing",
                  ips=['<ip of the cluster>'],
                  ssh_creds={'ssh_user': '<user>', 'ssh_private_key':'<path_to_key>'},
              )

Useful Cluster Functions
------------------------

.. code:: ipython3

    tls_cluster.run(['pip install numpy && pip freeze | grep numpy'])


.. parsed-literal::
    :class: code-output

    Warning: Permanently added '54.172.178.196' (ED25519) to the list of known hosts.


.. parsed-literal::
    :class: code-output

    Requirement already satisfied: numpy in /opt/conda/lib/python3.10/site-packages (1.26.4)
    numpy==1.26.4




.. parsed-literal::
    :class: code-output

    [(0,
      'Requirement already satisfied: numpy in /opt/conda/lib/python3.10/site-packages (1.26.4)\nnumpy==1.26.4\n',
      "Warning: Permanently added '54.172.178.196' (ED25519) to the list of known hosts.\r\n")]



.. code:: ipython3

    tls_cluster.run_python(['import numpy', 'print(numpy.__version__)'])


.. parsed-literal::
    :class: code-output

    1.26.4




.. parsed-literal::
    :class: code-output

    [(0, '1.26.4\n', '')]
