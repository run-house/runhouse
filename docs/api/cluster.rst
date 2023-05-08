Cluster
====================================
A Cluster is a Runhouse primitive used for abstracting a particular hardware configuration.
This can be either an :ref:`on-demand cluster <OnDemandCluster>` (requires valid cloud credentials) or a BYO
(bring-your-own) cluster (requires IP address and ssh creds).

A cluster is assigned a name, through which it can be accessed and reused later on.

Cluster Factory Method
~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: runhouse.cluster

Cluster Class
~~~~~~~~~~~~~

.. autoclass:: runhouse.Cluster
  :members:
  :exclude-members:

    .. automethod:: __init__

OnDemandCluster Class
~~~~~~~~~~~~~~~~~~~~~
A OnDemandCluster is a cluster that uses SkyPilot functionality underneath to handle
various cluster properties.

.. autoclass:: runhouse.OnDemandCluster
   :members:
   :exclude-members:

    .. automethod:: __init__

Hardware Setup
~~~~~~~~~~~~~~
For BYO Clusters, no additional setup is required. You will just need to have the IP address for the cluster
and the path to SSH credentials ready to be used for the cluster initialization.

For On-Demand Clusters, which use SkyPilot to automatically spin up and down clusters on the cloud, you will
need to first set up cloud access on your local machine:

Run ``sky check`` to see which cloud providers are enabled, and how to set up cloud credentials for each of the
providers.

.. code-block:: cli

    sky check

For a more in depth tutorial on setting up individual cloud credentials, you can refer to
`SkyPilot setup docs <https://skypilot.readthedocs.io/en/latest/getting-started/installation.html#cloud-account-setup>`_.

Basic API Usage
~~~~~~~~~~~~~~~
For On-Demand Clusters, use the ``rh.cluster`` factory method as follows.

.. code-block:: python

  my_cluster = rh.cluster(name='rh-4-a100s',
                          instance_type='A100:4',    # or 'CPU:8', 'g5.2xlarge', etc
                          provider='gcp',            # Defaults to `cheapest` if empty
                          autostop_mins=-1,          # Defaults to default_autostop_mins, -1 suspends autostop
                          use_spot=True,             # You must have spot quota approved to use this
                          image_id='my_ami_string',  # Generally defaults to basic deep-learning AMIs
                          region='us-east-1'         # Looks for cheapest on your continent if empty
                          )

For BYO Clusters, which are specified by IP addresses and SSH credentials, use the ``rh.cluster`` factory
method as follows:

.. code-block:: python

  my_cluster = rh.cluster(ips=['<ip of the cluster>'],
                          ssh_creds={'ssh_user': '...', 'ssh_private_key':'<path_to_key>'},
                          name='rh-a10x')

For existing clusters, which can mean either a saved OnDemandCluster config (which will be brought back
up if needed), or a BYO or OnDemandCluster that is already up, you can simply pass in the name to
``rh.cluster`` as follows:

.. code-block:: python

  my_cluster = rh.cluster(name='~/my-local-a100')
  my_cluster = rh.cluster(name='@/my-a100-in-rh-rns')

  # or, if you just want to load the Cluster object without refreshing its status
  my_cluster = rh.cluster(name='^rh-v100', dryrun=True)

Advanced API Usage
~~~~~~~~~~~~~~~~~~
Additional utilites for on-demand Clusters include
:ref:`setting custom default configs <Setting Config Options>` and
:ref:`SkyPilot CLI commands for instance management <On-Demand Clusters>`.

To start an ssh session into the cluster so you can poke around or debug:

.. code-block:: cli

  $ # In CLI
  $ ssh my_cluster_name

.. code-block:: python

  # In Python
  my_cluster.ssh()

Function logs are output onto the cluster, which can be viewed at :code:`~/.rh/<cluster_name>_grpc_server.log`.

You can restart the RPC server, in the case that it crashes or you want to update a package that the
server has already imported. This runs much more quickly than shutting down and restarting a cluster.

.. code-block:: python

    my_cluster.restart_grpc_server()

For notebook users, to tunnel a JupyterLab server into your local browser:

.. code-block:: cli

  $ # In CLI
  $ runhouse notebook my_cluster

.. code-block:: python

  # In Python
  my_cluster.notebook()

To run a Shell command on the cluster:

.. code-block:: python

  my_cluster.run(['git clone ...', 'pip install ...'])

To run a Python on the cluster:

.. code-block:: python

  my_cluster.run_python(['import torch', 'print(torch.__version__)'])

To open a port, if you want to run an application on the cluster that requires a port to be open,
e.g. `Tensorboard <https://www.tensorflow.org/tensorboard/>`_, `Gradio <https://gradio.app/>`_:

.. code-block:: python

  my_cluster.ssh_tunnel(local_port=7860, remote_port=7860)
