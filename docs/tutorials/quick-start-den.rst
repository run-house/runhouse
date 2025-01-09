Installation and Compute Setup
==============================

.. raw:: html

    <p><a href="https://colab.research.google.com/github/run-house/notebooks/blob/stable/docs/quick-start-den.ipynb">
    <img height="20px" width="117px" src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a></p>

It will take just ~15 minutes to setup Runhouse and be ready to launch
your first cluster. You will need a service account or user account
authorized to launch compute on your cloud provider.

If you do not currently have access to a cloud account, but still want
to try the Runhouse APIs, review how to start a Runhouse server on your
`local
machine <https://www.run.house/docs/tutorials/quick-start-local>`__ for
a quick experiment instead.

Installing Runhouse
-------------------

To use Runhouse to launch on-demand clusters, run the following
installation command.

.. code:: ipython3

    !pip install "runhouse"

If you want to try using Runhouse by launching from your local machine
with local credentials, you can add SkyPilot and the cloud provider of
your choice as additional installs.

.. code:: ipython3

    !pip install "runhouse[gcp, sky]"

Account Creation & Login
------------------------

You can create an account on the `Runhouse <https://www.run.house>`__
website or by calling the login command in Python or CLI.

To login, call ``runhouse login`` in CLI or run the following in Python.

.. code:: ipython3

    import runhouse as rh

    rh.login(token='your token here - from https://www.run.house/account')

Enabling Compute Launch
-----------------------

In order to launch the clusters, you will to provide one or more sources
of compute. This is typically one or more of:

- **Elastic Compute from Cloud Provider**: Save a Service Account to
  Runhouse. For more information about the required service account
  permissions, review our documentation for `service
  accounts <https://www.run.house/docs/tutorials/service-account-requirements>`__.
- **Kubernetes Cluster**: Save your Kubeconfig to Runhouse
- **On Premise VM**: Provide SSH Key or Username/Password

For the first option, to set up Runhouse with your service account,
simply run the following Python:

.. code:: ipython3

    gcp_secret = rh.provider_secret(name="gcp", path="Local path to your service account key, e.g. /Users/username/Downloads/runhouse-service-account.json")
    gcp_secret.save()

Saved Service Accounts and Kubeconfigs are always treated as secrets and
stored securely. For more information on Secrets management, refer to
the `Secrets
Tutorial <https://www.run.house/docs/tutorials/api-secrets>`__. For
Runhouse cloud users, Secrets are securely stored in
`Vault <https://www.vaultproject.io/>`__.

Runhouse Enterprise users may have other secrets configurations, and
Runhouse supports additional configurations to work with your
organization’s cloud settings out-of-the-box, such as setting up
Runhouse to launch within a specific VPC.

Local Launching
---------------

It is also possible to leverage SkyPilot to launch clusters from your
local machine and use Runhouse as a library only. In the local setting,
we use Skypilot under the hood to launch elastic compute. Assuming you
are already logged in locally via CLI to your cloud provider of choice,
this should work out-of-the-box with no further configuration.

Review Skypilot documentation to understand how to setup your local
credentials, and you can run CLI command ``sky check`` after installing
Runhouse with Skypilot enabled to confirm you have access to the cloud.
If you are authenticated locally with your cloud CLI, and your account
has the right permissions to launch clusters, Skypilot’s check should
reflect that.

Launching Clusters
------------------

You are now ready to launch clusters with Runhouse. Simply specify the
resources you want to launcb, and in this example, we will bring up a
simple 2 CPU 1 node cluster.

.. code:: ipython3

    cluster = rh.ondemand_cluster(
        name="rh-cluster",
        num_cpus="2",
        provider="gcp",
        launcher="den" # Set to `local` if you are launching from your local machine
    ).up_if_not()

Autostop should already be enabled by default on your Runhouse-launched
clusters (which you can control in configurations or for your
organization). If we want to tear this cluster down:

.. code:: ipython3

    cluster.teardown()

Now you’re ready to start working with the Runhouse APIs. Jump over to
the `API Quick Start
guide <https://www.run.house/docs/tutorials/quick-start-den>`__.
