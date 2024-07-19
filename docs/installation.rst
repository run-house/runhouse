Installation and Setup
======================

An overview of common installation and setup steps when working with Runhouse.

- Installing the Runhouse library
- Cloud providers and credentials
- Cluster lifecycle management
- Authenticate with Runhouse Den

Install the Runhouse library
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Optionally, begin by setting up a virtual env with `Conda <https://conda.io/projects/conda/en/latest/user-guide/install/index.html>`_
or your package and environment management tool of choice. This allows you to maintain an environment with a
specific version of Python and installed packages.

.. code-block:: shell

    $ conda create -n rh_env python=3.9.15
    $ conda activate rh_env

For local-only development, the Runhouse base package can be installed with:

.. code-block:: shell

    $ pip install runhouse

**Recommended**: To use Runhouse to launch on-demand clusters, please instead run the following command.
This additionally installs SkyPilot, which is used for launching fresh VMs through your cloud provider.

.. code-block:: shell

    $ pip install "runhouse[sky]"

Alternatively, if you plan to use Runhouse with a specific cloud provider, you can install the package with
that provider's CLI included - or choose multiple providers. SkyPilot will also be included with these.

.. code-block:: shell

    # Cloud-specific installation
    $ pip install "runhouse[aws]"
    # Include multiple providers
    $ pip install "runhouse[aws,gcp]"


Configure Cloud Credentials
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Runhouse uses `SkyPilot <https://github.com/skypilot-org/skypilot>`_ to launch and manage virtual machine instances on your cloud providers.

For each provider, we recommend you begin by checking your configuration with SkyPilot.

.. code-block:: shell

    $ sky check

This will show a detailed status of which cloud providers have already been configured properly.

AWS
---

Start by installing the AWS CLI. Follow this
`Getting Started <https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html>`_ guide
or use ``pip install "runhouse[aws]"`` to include it with the Runhouse library.

Next, configure AWS with the following command:

.. code-block:: shell

    $ aws configure

You'll be prompted to enter your AWS Access ID. This can be found by logging into
the `AWS Console <https://console.aws.amazon.com/console/home>`_. Click on the name
of your account in the top-left corner of the screen and then select
"Securtiy credentials" from the dropdown.

To verify that credentials are properly set up, run the SkyPilot command again:

.. code-block:: shell

    $ sky check

For more info: `SkyPilot AWS <https://skypilot.readthedocs.io/en/latest/cloud-setup/cloud-permissions/aws.html>`_

GCP
---

Begin by installing Runhouse with GCP:

.. code-block:: shell

    $ pip install "runhouse[gcp]"

Additionally, you'll need to install GCP tools. Run the following commands:

.. code-block:: shell

    $ pip install google-api-python-client
    $ conda install -c conda-forge google-cloud-sdk -y

Your GCP credentials may also need to be set:

.. code-block:: shell

    $ gcloud init
    # Run this if you don't have a credentials file.
    $ gcloud auth application-default login
    $ sky check


You'll be prompted to pick a cloud project to use after running ``gcloud init``.
If you don't have one ready yet, you can connect one later by listing your projects
with ``gcloud projects list`` and setting one
with ``gcloud config set project <PROJECT_ID>``.

For more info: `SkyPilot GCP <https://skypilot.readthedocs.io/en/latest/getting-started/installation.html#google-cloud-platform-gcp>`_

Azure
-----

If you have an Azure account, use the following commands to install Runhouse and configure your
credentials to launch instances on their cloud.

.. code-block:: shell

    $ pip install "runhouse[azure]"
    # Login
    $ az login
    # Set the subscription to use
    $ az account set -s <subscription_id>
    $ sky check

For more info: `SkyPilot Azure <https://skypilot.readthedocs.io/en/latest/getting-started/installation.html#azure>`_


Other Providers
---------------

For a full list of providers and configuration details: `SkyPilot documentation <https://skypilot.readthedocs.io/en/latest/getting-started/installation.html#cloud-account-setup>`_.
Some additional providers supported by Runhouse via SkyPilot include:

- Kubernetes
- Lambda Cloud
- & more

Cluster Lifecycle Management
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To ensure that you have full control over the availability of your clusters, and the cost associated with maintaining them,
you may find the following `SkyPilot commands <https://skypilot.readthedocs.io/en/latest/reference/cli.html#cluster-cli>`_ helpful:

- ``sky status`` - Displays a list of all your clusters by name.
- ``sky stop <NAME>`` - Stops an instance. This will not fully tear down the cluster in case you need to restart it.
- ``sky down <NAME>`` - Fully tears down the instance. Best for when you are done using a cluster entirely.
- ``ssh <NAME>`` - Easily SSH into your cluster. Here you can further examine Runhouse's server with ``runhouse status``.

Example output from ``sky status``:

.. parsed-literal::
    :class: code-output

    Clusters
    NAME                     LAUNCHED    RESOURCES                                        STATUS  AUTOSTOP    COMMAND
    rh-a10                   4 days ago  1x AWS(g5.2xlarge, {'A10G': 1}, ports=['8080'])  UP      30m (down)  llama3_tgi_ec2.py
    sky-3201-matthewkandler  4 days ago  1x AWS(g5.xlarge, {'A10G': 1})                   UP      -           sky launch --gpus A10G:1 ...

    Managed jobs
    No in-progress managed jobs. (See: sky jobs -h)

    Services
    No live services. (See: sky serve -h)


Authenticate with Runhouse Den
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To enable sharing features through Runhouse Den, you can log in to your
Runhouse Den account. Start by creating one via our `signup page <https://www.run.house/signup>`_.
You'll have the option to authenticate with your Google or Github account.

Once you've created an account, you'll be able to access your Runhouse token on your `account page <https://www.run.house//account>`_.

Login to Runhouse in your terminal with the following command:

.. code-block:: shell

    $ runhouse login

You'll be prompted to enter your token and, after your initial login, you will see the following propmts:

- ``Download your Runhouse config to your local .rh folder? [Y/n]:`` - Updates your local config file from your settings on the cloud
- ``Upload your local .rh config to Runhouse? [y/N]:`` - Use updates to your local config to modify your saved account settings

If you are running Runhouse strictly in code (like in a notebook), you can also login to your account with the Python API:

.. code-block:: python

    import runhouse as rh
    rh.login()
