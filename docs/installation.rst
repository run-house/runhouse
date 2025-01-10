Installation and Setup
======================

An overview of common installation and setup steps when working with Runhouse.

Install the Runhouse library
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Optionally, begin by setting up a virtual env with
`Conda <https://conda.io/projects/conda/en/latest/user-guide/install/index.html>`_ or your package and environment
management tool of choice. This allows you to maintain an environment with a specific version of Python and installed
packages.

.. code-block:: shell

    $ conda create -n rh_env python=3.9.15
    $ conda activate rh_env

The Runhouse base package can be installed with:

.. code-block:: shell

    $ pip install runhouse

To use Runhouse to launch on-demand clusters from your local environment, you can run the following command instead.
This additionally installs `SkyPilot <https://github.com/skypilot-org/skypilot>`__, which is used for launching fresh
VMs through your cloud provider. Or, to launch a cluster through the Den launcher, only the base package is necessary.

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

For `on-demand clusters </docs/tutorials/api-clusters#on-demand-clusters>`_, Runhouse uses SkyPilot to launch and
manage virtual machine instances on your cloud providers, so the credentials must be available to SkyPilot at launch
time, which can be configured as below. These steps are not necessary when using static clusters that can be accessed
with SSH credentials.

To see a detailed status of which cloud providers have already been configured properly, as well as steps to set up
unconfigured providers:

.. code-block:: shell

    $ sky check

You can also refer to the `SkyPilot documentation
<https://skypilot.readthedocs.io/en/latest/getting-started/installation.html#cloud-account-setup>`__. for a full list
of providers and configuration details.

On-Demand Cluster Lifecycle Management
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To ensure that you have full control over the availability of on-demand clusters, and the cost associated with
maintaining them, you may find the following `SkyPilot commands
<https://skypilot.readthedocs.io/en/latest/reference/cli.html#cluster-cli>`__ helpful:

- ``sky status`` - Displays a list of all your clusters by name. Add ``--refresh`` to refresh the clusters before
  displaying.

- ``sky stop <NAME>`` - Stops an instance. This will not fully tear down the cluster in case you need to restart it.

- ``sky down <NAME>`` - Fully tears down the instance. Best for when you are done using a cluster entirely.

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

To enable sharing features through Runhouse Den, you can log in to your Runhouse Den account. Start by creating one via
our `signup page <https://www.run.house/signup>`__. You'll have the option to authenticate with your Google or Github
account.

Once you've created an account, you'll be able to access your Runhouse token on your `account page
<https://www.run.house//account>`__.

Login to Runhouse in your terminal with the following command, optionally including the ``--sync-secrets`` flag to
upload or download any credentials you want to save into your account, or download into your local environment:

.. code-block:: shell

    $ runhouse login

You'll be prompted to enter your token and, after your initial login, you will see the following propmts:

- ``Download your Runhouse config to your local .rh folder? [Y/n]:`` - Updates your local config file from your settings on the cloud
- ``Upload your local .rh config to Runhouse? [y/N]:`` - Use updates to your local config to modify your saved account settings

If you are running Runhouse strictly in code (like in a notebook), you can also login to your account with the Python API:

.. code-block:: python

    import runhouse as rh
    rh.login()
