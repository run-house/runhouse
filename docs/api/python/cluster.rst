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

    Please see the :ref:`SageMaker Clusters <SageMaker Clusters>` section for more specific instructions and
    requirements for setting up the cluster.

.. autoclass:: runhouse.SageMakerCluster
   :members:
   :exclude-members:

    .. automethod:: __init__


SageMaker Factory Method
~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: runhouse.sagemaker_cluster


Hardware Setup
~~~~~~~~~~~~~~

BYO Clusters
------------
No additional setup is required. You will just need to have the IP address for the cluster
and the path to SSH credentials ready to be used for the cluster initialization.

On-Demand Clusters
------------------

On-Demand clusters use SkyPilot to automatically spin up and down clusters on the cloud. You will
need to first set up cloud access on your local machine:

Run ``sky check`` to see which cloud providers are enabled, and how to set up cloud credentials for each of the
providers.

.. code-block:: cli

    sky check

For a more in depth tutorial on setting up individual cloud credentials, you can refer to
`SkyPilot setup docs <https://skypilot.readthedocs.io/en/latest/getting-started/installation.html#cloud-account-setup>`_.

SageMaker Clusters
------------------

SageMaker clusters require `AWS CLI V2 <https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-welcome.html>`_ and
configuring the SageMaker IAM role with the
`AWS Systems Manager <https://docs.aws.amazon.com/systems-manager/latest/userguide/ssm-agent.html>`_.

**IAM Role**

In order to use SageMaker clusters, you must grant SageMaker the necessary permissions with an IAM role.
You can provide this role either by profile name or by full ARN, via an estimator, or with the :code:`AWS_PROFILE` environment
variable.

Please see the `AWS docs <https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-roles.html>`_ for more info
on creating and configuring the role.


**AWS CLI V2**

- `Uninstall <https://docs.aws.amazon.com/cli/latest/userguide/cliv2-migration-instructions.html#cliv2-migration-instructions-migrate>`_ AWS CLI V1

- `Install <https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html>`_ AWS CLI V2


To confirm the installation succeeded, run ``aws --version`` in the command line. You should see something like:

.. code-block:: cli

    aws-cli/2.13.8 Python/3.11.4 Darwin/21.3.0 source/arm64 prompt/off

**SSM Setup**

The AWS Systems Manager service is used to create SSH tunnels with the SageMaker cluster.

To install the AWS Session Manager Plugin, please see the `AWS docs <https://docs.aws.amazon.com/systems-manager/latest/userguide/session-manager-working-with-install-plugin.html>`_
or `SageMaker SSH Helper <https://https://github.com/aws-samples/sagemaker-ssh-helper>`_. The SSH Helper package
simplifies the process of creating SSH tunnels with SageMaker clusters. It is installed by default if
you are using the SageMaker dependency: :code:`pip install runhouse[sagemaker]`.

You can install the Session Manager using the command:

.. code-block:: cli

    sm-local-configure

To configure your SageMaker IAM role with the AWS Systems Manager, please
refer to `these instructions <https://github.com/aws-samples/sagemaker-ssh-helper/blob/main/IAM_SSM_Setup.md>`_.
