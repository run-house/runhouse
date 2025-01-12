Installation and Compute Setup
==============================

It will take just a few minutes to setup Runhouse, including the Runhouse installation, account
creation, and configuring of compute credenitals (e.g. service or user account for a cloud
credential).

Runhouse is compatible with:

* On-demand clusters launched through Runhouse
* On-demand clusters launched locally
* Existing static clusters

If you do not currently have access to a cloud account, but still want to try the Runhouse APIs,
review `how to use Runhouse locally <https://www.run.house/docs/tutorials/quick-start-local>`__
for a quick experiment instead.

Installing Runhouse
-------------------

The Runhouse package can be installed with:

.. code::

    !pip install "runhouse"

The base package is sufficient to be able to launch on-demand clusters through Runhouse. If you
you prefer to launch from your local machine with local credentials, you can specify SkyPilot and
the cloud provider of your choice as additional installs:

.. code::

    !pip install "runhouse[gcp, sky]"

Account Creation & Login
------------------------

You can create an account in the `Runhouse website <https://www.run.house>`__, or by calling the
login command in Python or CLI, which will redirect you to the sign up page. To continue logging
in on your machine, paste in your generated Runhouse token when prompted for it.

.. code::

    !runhouse login

.. code:: ipython3

    import runhouse as rh

    rh.login(token="generated_token_from_signup")

Access to Compute
-----------------

In order to use Runhouse, you must be able to access compute resources, which can take any form
(e.g. VMs, elastic compute, Kubernetes). You should think about all the compute resources you have
as a single pool, from which Runhouse allows you to launch ephemeral clusters to execute your code.

* **Elastic Compute**: Specify a service account from your cloud provider and Runhouse launches and
  manages clusters for you (including enabling telemetry, auto-stop, etc).

* **Kubernetes**: All you need is a kubeconfig to launch Runhouse clusters out of your existing
  Kubernetes clusters.

* **Existing VM**: Runhouse supports a variety of authentication methods to access existing
  compute, including SSH with keys or passwords.

You can specify cloud credentials or kube configs in the form of a Runhouse secret object, and
save it into your Runhouse account, where they will be securely stored in `Vault
<https://www.vaultproject.io/>`__. For more information on Secrets management, refer to
the `Secrets Tutorial <https://www.run.house/docs/tutorials/api-secrets>`__.

.. code:: ipython3

    gcp_creds = rh.provider_secret(provider="gcp", path="local_path/to/gcp-service-account.json")
    gcp_creds.save()

    kube_config = rh.provider_secret(provider="kubernetes", path="~/.kube/config")
    kube_config.save()

Runhouse Enterprise users may have other secrets configurations, and Runhouse supports additional
configurations to work with your organization’s cloud settings out-of-the-box, such as setting up
Runhouse to launch within a specific VPC.

Launching Clusters
------------------

Den Launcher
~~~~~~~~~~~~
The Den launcher allows you to launch clusters in your own cloud via the Runhouse control plane.
We recommend this approach for a couple of reasons:

* **Resource Management**: Clusters launched through Runhouse are automatically persisted in the
  Den UI, making it easier to track and manage your entire collection of resources.

* **Distributed Workflows / Production Pipelines**: Launch clusters as part of distributed workflows
  or pipelines without needing to configure cloud credentials in your environment.

To enable the Den launcher, you can set :code:`launcher="den"` in the :ref:`cluster factory <Cluster Factory Methods>`, or update
your local :ref:`runhouse config <Setting Config Options>` with :code:`launcher: den` to apply the setting globally
across all subsequent clusters created.

Local Launcher
~~~~~~~~~~~~~~

It is also possible to leverage SkyPilot to launch clusters elastic compute from your local
machine and use Runhouse as a library only. If you are already logged in locally via CLI to your
cloud provider of choice, this should work out-of-the-box with no further configuration.

To launch locally without an account, review `Skypilot's cloud setup documentation
<https://docs.skypilot.co/en/latest/getting-started/installation.html#cloud-account-setup>`__
to understand how to set up and check your local credentials. Run the CLI command ``sky check``
after installing Runhouse with Skypilot enabled to confirm you have sufficient access to the cloud.

Launching API
~~~~~~~~~~~~~

You are now ready to launch clusters with Runhouse. Simply specify the resources you want to
launch. In this example, we will bring up a 2 CPU 1 node cluster.

.. code:: ipython3

    cluster = rh.ondemand_cluster(
        name="rh-cluster",
        num_cpus="2",
        provider="gcp",
        launcher="den" # Set to `local` if you are launching from your local machine
    ).up_if_not()

A default autostop of 60 min is automatically enabled on your Runhouse-launched clusters. You can
configure this in you or your organization's configurations (``~/.rh/config.yaml``), or by
specifying ``autostop_mins=desired_autostop`` in the cluster constructor.

To tear this cluster down:

.. code:: ipython3

    cluster.teardown()

Now you’re ready to start working with the Runhouse APIs. Jump over to the `API Quick Start
guide <https://www.run.house/docs/tutorials/quick-start-den>`__ to start deploying and executing
code on your cluster.
