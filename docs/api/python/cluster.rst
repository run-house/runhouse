Cluster
====================================
A Cluster is a Runhouse primitive used for abstracting a particular hardware configuration.
This can be either an :ref:`on-demand cluster <OnDemandCluster Class>` (requires valid cloud credentials), a
:ref:`BYO (bring-your-own) cluster <Cluster Factory Method>` (requires IP address and ssh creds), or a
:ref:`SageMaker cluster <SageMakerCluster Class>` (requires an ARN role).

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


SageMakerCluster Class
~~~~~~~~~~~~~~~~~~~~~
A SageMakerCluster is a cluster that uses a SageMaker instance under the hood.

Runhouse currently supports two core usage paths for SageMaker clusters:

- *Compute backend*: You can use SageMaker as a compute backend, just as you would a
  :ref:`BYO (bring-your-own) <Cluster Class>` or an :ref:`on-demand cluster <OnDemandCluster Class>` cluster.
  Runhouse will facilitate the creation of the SageMaker compute and will handle the creation of an SSH
  connection to the instance. You can then use the instance as you would any other compute backend.

.. raw:: html

  <br>

- *Dedicated training jobs*: You can use a SageMakerCluster class to run a training job on SageMaker compute.
  To do so, you will need to provide an
  `estimator <https://sagemaker.readthedocs.io/en/stable/api/training/estimators.html>`_.

.. note::

    Runhouse requires an AWS IAM role (either name or full ARN) whose credentials have adequate permissions to
    create create SageMaker endpoints and access AWS resources.

    This can be specified with the ``role`` attribute. If not provided, Runhouse will use the SageMaker default
    role configured in your local environment.

.. autoclass:: runhouse.SageMakerCluster
   :members:
   :exclude-members:

    .. automethod:: __init__


SageMaker Factory Method
~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: runhouse.sagemaker_cluster


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
