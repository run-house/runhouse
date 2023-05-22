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
