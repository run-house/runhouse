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


Cluster Hardware Setup
----------------------

No additional setup is required. You will just need to have the IP address for the cluster
and the path to SSH credentials ready to be used for the cluster initialization.


OnDemandCluster Class
~~~~~~~~~~~~~~~~~~~~~
A OnDemandCluster is a cluster that uses SkyPilot functionality underneath to handle
various cluster properties.

.. autoclass:: runhouse.OnDemandCluster
   :members:
   :exclude-members:

    .. automethod:: __init__

OnDemandCluster Hardware Setup
------------------------------

On-Demand clusters use SkyPilot to automatically spin up and down clusters on the cloud. You will
need to first set up cloud access on your local machine:

Run ``sky check`` to see which cloud providers are enabled, and how to set up cloud credentials for each of the
providers.

.. code-block:: cli

    sky check

For a more in depth tutorial on setting up individual cloud credentials, you can refer to
`SkyPilot setup docs <https://skypilot.readthedocs.io/en/latest/getting-started/installation.html#cloud-account-setup>`_.


SageMakerCluster Class
~~~~~~~~~~~~~~~~~~~~~
.. note::

    SageMaker support is an alpha and under active development. Please report any bugs or let us know of any
    feature requests.

A SageMakerCluster is a cluster that uses a SageMaker instance under the hood.

Runhouse currently supports two core usage paths for SageMaker clusters:

- **Compute backend**: You can use SageMaker as a compute backend, just as you would a
  :ref:`BYO (bring-your-own) <Cluster Class>` or an :ref:`on-demand cluster <OnDemandCluster Class>`.
  Runhouse will handle launching the SageMaker compute and creating the SSH connection
  to the cluster.

- **Dedicated training jobs**: You can use a SageMakerCluster class to run a training job on SageMaker compute.
  To do so, you will need to provide an
  `estimator <https://sagemaker.readthedocs.io/en/stable/api/training/estimators.html>`_.

.. note::

    Runhouse requires an AWS IAM role (either name or full ARN) whose credentials have adequate permissions to
    create create SageMaker endpoints and access AWS resources.

    Please see :ref:`SageMaker Hardware Setup` for more specific instructions and
    requirements for providing the role and setting up the cluster.

.. autoclass:: runhouse.SageMakerCluster
   :members:
   :exclude-members:

    .. automethod:: __init__

SageMaker Hardware Setup
------------------------

IAM Role
^^^^^^^^

SageMaker clusters require `AWS CLI V2 <https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-welcome.html>`_ and
configuring the SageMaker IAM role with the
`AWS Systems Manager <https://docs.aws.amazon.com/systems-manager/latest/userguide/ssm-agent.html>`_.


In order to launch a cluster, you must grant SageMaker the necessary permissions with an IAM role, which
can be provided either by name or by full ARN. You can also specify a profile explicitly or
with the :code:`AWS_PROFILE` environment variable.

For example, let's say your local :code:`~/.aws/config` file contains:

.. code-block:: ini

    [profile sagemaker]
    role_arn = arn:aws:iam::123456789:role/service-role/AmazonSageMaker-ExecutionRole-20230717T192142
    region = us-east-1
    source_profile = default

There are several ways to provide the necessary credentials when :ref:`initializing the cluster <SageMaker Factory Method>`:

- Providing the AWS profile name: :code:`profile="sagemaker"`
- Providing the AWS Role ARN directly: :code:`role="arn:aws:iam::123456789:role/service-role/AmazonSageMaker-ExecutionRole-20230717T192142"`
- Environment Variable: setting :code:`AWS_PROFILE` to :code:`"sagemaker"`

.. note::

    If no role or profile is provided, Runhouse will try using the :code:`default` profile. Note if this default AWS
    identity is not a role, then you will need to provide the :code:`role` or :code:`profile` explicitly.

.. tip::

    If you are providing an estimator, you must provide the role ARN explicitly as part of the estimator object.
    More info on estimators `here <https://sagemaker.readthedocs.io/en/stable/api/training/estimators.html>`_.

Please see the `AWS docs <https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-roles.html>`_ for further
instructions on creating and configuring an ARN Role.


AWS CLI V2
^^^^^^^^^^

Runhouse requires the AWS CLI V2 to be installed on your local machine.

- `Uninstall <https://docs.aws.amazon.com/cli/latest/userguide/cliv2-migration-instructions.html#cliv2-migration-instructions-migrate>`_ AWS CLI V1

- `Install <https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html>`_ AWS CLI V2


To confirm the installation succeeded, run ``aws --version`` in the command line. You should see something like:

.. code-block:: cli

    aws-cli/2.13.8 Python/3.11.4 Darwin/21.3.0 source/arm64 prompt/off

SSM Setup
^^^^^^^^^
The AWS Systems Manager service is used to create SSH tunnels with the SageMaker cluster.

To install the AWS Session Manager Plugin, please see the `AWS docs <https://docs.aws.amazon.com/systems-manager/latest/userguide/session-manager-working-with-install-plugin.html>`_
or `SageMaker SSH Helper <https://github.com/aws-samples/sagemaker-ssh-helper#step-4-connect-over-ssm>`_. The SSH Helper package
simplifies the process of creating SSH tunnels with SageMaker clusters. It is installed by default if
you are installing Runhouse with the SageMaker dependency: :code:`pip install runhouse[sagemaker]`.

You can also install the Session Manager by running the CLI command:

.. code-block:: cli

    sm-local-configure

To configure your SageMaker IAM role with the AWS Systems Manager, please
refer to `these instructions <https://github.com/aws-samples/sagemaker-ssh-helper/blob/main/IAM_SSM_Setup.md>`_.


SageMaker Factory Method
~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: runhouse.sagemaker_cluster


Cluster Authentication & Verification
====================================
Runhouse provides a couple of options to manage the connection to the Runhouse API server running on a cluster.

Server Connection
~~~~~~~~~~~~~~~~~

The below options can be specified with the ``server_connection_type`` parameter
when :ref:`initializing a cluster <Cluster Factory Method>`. By default the Runhouse API server will
be started on the cluster on port :code:`32300`.

- ``ssh``: Connects to the cluster via an SSH tunnel.
- ``tls``: Connects to the cluster via HTTPS (by default on port :code:`443`) and enforces verification via TLS
  certificates. Only users with a valid cert will be able to make requests to the API server.
- ``none``: Does not use any port forwarding or enforce any authentication. Connects to the cluster with HTTP by
  default on port :code:`80`.
- ``aws_ssm``: Uses the
  `AWS Systems Manager <https://docs.aws.amazon.com/systems-manager/latest/userguide/what-is-systems-manager.html>`_ to
  create an SSH tunnel to the cluster. *Note: this is currently only relevant for SageMaker Clusters*
- ``paramiko``: Uses `Paramiko <https://www.paramiko.org/>`_ to create an SSH tunnel to the cluster. This
  is relevant if you are using a cluster which require existing credentials (e.g. a password)


.. note::

    The ``tls`` connection type is the most secure and is recommended for production use if you are not running inside
    of a VPC.

Server Authentication
~~~~~~~~~~~~~~~~~~~~~

Runhouse allows you to authenticate users via their Runhouse token (generated when
:ref:`logging in <Login/Logout>`) and saved to local Runhouse configs in path: :code:`~/.rh/config.yaml`.

When :ref:`initializing a cluster <Cluster Factory Method>`, you can set the :code:`den_auth` parameter to :code:`True`
to enable token authentication. Runhouse will handle adding the token to each subsequent request as an auth header with
format: :code:`{"Authorization": "Bearer <token>"}`


TLS Certificates
----------------
Enabling TLS and `Runhouse Den Dashboard <https://www.run.house/dashboard>`_ Auth for the API server makes it incredibly fast
and easy to stand up a microservice with standard token authentication, allowing you to easily share Runhouse resources
with collaborators, teams, customers, etc.

Let's illustrate this with a simple example:

.. code-block:: python

    import runhouse as rh

    def np_array(num_list: list):
        import numpy as np

        return np.array(num_list)

    # Launch a cluster with TLS and Den Auth enabled
    cpu = rh.ondemand_cluster(instance_type="m5.xlarge",
                              provider="aws",
                              name="rh-cluster",
                              den_auth=True,
                              server_connection_type="tls").up_if_not()

    # Remote function stub which lives on the cluster
    remote_func = rh.function(np_array).to(cpu, env=["numpy"])

    # Save to Runhouse Den
    remote_func.save()

    # Share the function with another user
    remote_func.share("user1@gmail.com")


    # Run function on cluster
    # Note: only users with a Runhouse token and access to this function can run it
    res = remote_func([1,2,3])


.. note::

    For more examples on using clusters and functions see
    the :ref:`Compute Guide <Compute: Clusters, Functions, Packages, & Envs>`.

Nginx
-----
Runhouse gives you the option of using `Nginx <https://www.nginx.com/>`_ as a reverse proxy for the Runhouse API
server, which is a Fast API launched with `Uvicorn <https://www.uvicorn.org/>`_. Using Nginx provides you with a safer
and more conventional approach running the FastAPI app on a higher, non-privileged port (such as 32300, the default
Runhouse port) and then use Nginx as a reverse proxy to forward requests from the HTTP port (default: 80) or the
HTTPS port (default: 443).

.. note::

    Nginx is enabled by default when you launch a cluster with the :code:`server_port` set to either 80 or 443.
