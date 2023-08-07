Quick Start Guide
=================

.. raw:: html

    <p><a href="https://colab.research.google.com/github/run-house/runhouse/blob/stable/docs/notebooks/basics/quick_start.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a></p>


This tutorials walks through Runhouse setup (installation, hardware
setup, and optional login) and goes through an example that demonstrates
how to user Runhouse to bridge the gap between local and remote compute,
and create Resources that can be saved, reused, and shared.

Installation
------------

Runhouse can be installed with:

.. code:: python

    !pip install runhouse

If using Runhouse with a cloud provider, you can additionally install
cloud packages (e.g. the right versions of tools like boto, gsutil,
etc.):

::

   $ pip install "runhouse[aws]"
   $ pip install "runhouse[gcp]"
   $ pip install "runhouse[azure]"
   # Or
   $ pip install "runhouse[all]"

To import runhouse:

.. code:: python

    import runhouse as rh

Cluster Setup
-------------

Runhouse provides APIs and Secrets management to make it easy to
interact with your clusters. This can be either an existing, on-prem
cluster you have access to, or cloud instances that Runhouse spins
up/down for you (through your own cloud account).

**Note that Runhouse is NOT managed compute. Everything runs inside your
own compute and storage, using your credentials.**

Bring-Your-Own Cluster
~~~~~~~~~~~~~~~~~~~~~~

If you are using an existing, on-prem cluster, no additional setup is
needed. Just have your cluster IP address and path to SSH credentials
ready:

.. code:: python

    # using private key
    cluster = rh.cluster(
                name="cpu-cluster",
                ips=['<ip of the cluster>'],
                ssh_creds={'ssh_user': '<user>', 'ssh_private_key':'<path_to_key>'},
            )

    # using password
    cluster = rh.cluster(
                name="cpu-cluster",
                ips=['<ip of the cluster>'],
                ssh_creds={'ssh_user': '<user>', 'password':'******'},
            )

On-Demand Cluster
~~~~~~~~~~~~~~~~~

For on-demand clusters through cloud accounts (e.g. AWS, Azure, GCP,
LambdaLabs), Runhouse uses
`SkyPilot <https://github.com/skypilot-org/skypilot>`__ for much of the
heavy lifting with launching and terminating cloud instances.

To set up your cloud credentials locally to be able to use on-demand
cloud clusters, you can either:

1. Use SkyPilot’s CLI command ``!sky check``, which provides
   instructions on logging in or setting up your local config file,
   depending on the provider (further SkyPilot instructions
   `here <https://skypilot.readthedocs.io/en/latest/getting-started/installation.html#cloud-account-setup>`__)

2. Use Runhouse’s Secrets API to sync your secrets down into the
   appropriate local config.

.. code:: python

    # SkyPilot CLI
    !sky check


.. code:: python

    # Runhouse Secrets
    # Lambda Labs:
    rh.Secrets.save_provider_secrets(secrets={"lambda": {"api_key": "*******"}})

    # AWS:
    rh.Secrets.save_provider_secrets(secrets={"aws": {"access_key": "******", "secret_key": "*******"}})

    # GCP:
    !gcloud init
    !gcloud auth application-default login
    !cp -r /content/.config/* ~/.config/gcloud

    # Azure
    !az login
    !az account set -s <subscription_id>

To check that the provider credentials are properly configured locally,
run ``sky check`` to confirm that the cloud provider is enabled

.. code:: python

    !sky check

To create a cluster instance, use the ``rh.cluster()`` factory function.
We go more in depth about how to launch the cluster, and run a function
on it later in this tutorial.

.. code:: python

    cluster = rh.ondemand_cluster(
                  name="cpu-cluster",
                  instance_type="CPU:8",
                  provider="cheapest",      # options: "AWS", "GCP", "Azure", "Lambda", or "cheapest"
              )

Secrets and Portability
-----------------------

Using Runhouse with only the OSS Python package is perfectly fine, but
you can unlock some unique portability features by creating an (always
free) `account <https://www.run.house/>`__ and saving down your secrets
and/or resource metadata there.

Think of the OSS-package-only experience as akin to Microsoft Office,
while creating an account will make your cloud resources sharable and
accessible from anywhere like Google Docs.

For instance, if you previously set up cloud provider credentials in
order for launching on-demand clusters, simply call ``runhouse login``
or ``rh.login()`` and choose which of your secrets you want to sync into
your Runhouse account. Then, from any other environment, you can
download those secrets and use them immediately, without needing to set
up your local credentials again. To delete any local credentials or
remove secrets from Runhouse, you can call ``runhouse logout`` or
``rh.logout()``.

Some notes on security \* Our API servers only ever store light metadata
about your resources (e.g. folder name, cloud provider, storage bucket,
path). All actual data and compute stays inside your own cloud account
and never hits our servers. \* Secrets are stored in `Hashicorp
Vault <https://www.vaultproject.io/>`__ (an industry standard for
secrets management), never on our API servers, and our APIs simply call
into Vault’s APIs.

.. code:: python

    !runhouse login
    # or
    rh.login()

Getting Started Example
-----------------------

In the following example, we demonstrate Runhouse’s simple but powerful
compute APIs to run locally defined functions on a remote cluster
launched through Runhouse, bridging the gap between local and remote.
Additionally, save, reuse, and share any of your Runhouse Resources.

Please first make sure that you have successfully followed the
Installation and Cluster Setup sections above prior to running this
example.

.. code:: python

    import runhouse as rh

Running local functions on remote hardware
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

First let’s define a simple local function which returns the number of
CPUs available.

.. code:: python

    def num_cpus():
        import multiprocessing
        return f"Num cpus: {multiprocessing.cpu_count()}"

    num_cpus()




.. parsed-literal::

    'Num cpus: 2'



Next, instantiate the cluster that we want to run this function on. This
can be either an existing cluster where you pass in an IP address and
SSH credentials, or a cluster associated with supported Cloud account
(AWS, GCP, Azure, LambdaLabs), where it is automatically launched (and
optionally terminated) for you.

.. code:: python

    # Using an existing, bring-your-own cluster
    cluster = rh.cluster(
                  name="cpu-cluster",
                  ips=['<ip of the cluster>'],
                  ssh_creds={'ssh_user': '<user>', 'ssh_private_key':'<path_to_key>'},
              )

    # Using a Cloud provider
    cluster = rh.ondemand_cluster(
                  name="cpu-cluster",
                  instance_type="CPU:8",
                  provider="cheapest",      # options: "AWS", "GCP", "Azure", "Lambda", or "cheapest"
              )


.. parsed-literal::

    INFO | 2023-05-05 14:02:33,950 | Loaded Runhouse config from /root/.rh/config.yaml
    INFO | 2023-05-05 14:02:33,956 | Attempting to load config for /carolineechen/cpu-cluster from RNS.
    INFO | 2023-05-05 14:02:34,754 | No config found in RNS: {'detail': 'Resource does not exist'}


If using a cloud cluster, we can launch the cluster with ``.up()`` or
``.up_if_not()``.

Note that it may take a few minutes for the cluster to be launched
through the Cloud provider and set up dependencies.

.. code:: python

    cluster.up_if_not()

Now that we have our function and remote cluster set up, we’re ready to
see how to run this function on our cluster!

We wrap our local function in ``rh.function``, and associate this new
function with the cluster. Now, whenever we call this new function, just
as we would call any other Python function, it runs on the cluster
instead of local.

.. code:: python

    num_cpus_cluster = rh.function(name="num_cpus_cluster", fn=num_cpus).to(system=cluster, reqs=["./"])


.. parsed-literal::

    INFO | 2023-05-05 14:31:58,659 | Attempting to load config for /carolineechen/num_cpus_cluster from RNS.
    INFO | 2023-05-05 14:31:59,470 | No config found in RNS: {'detail': 'Resource does not exist'}
    INFO | 2023-05-05 14:31:59,473 | Writing out function function to /content/num_cpus_fn.py. Please make sure the function does not rely on any local variables, including imports (which should be moved inside the function body).
    INFO | 2023-05-05 14:31:59,476 | Setting up Function on cluster.
    INFO | 2023-05-05 14:31:59,479 | Copying local package content to cluster <cpu-cluster>
    INFO | 2023-05-05 14:32:04,026 | Installing packages on cluster cpu-cluster: ['./']
    INFO | 2023-05-05 14:32:04,402 | Function setup complete.


.. code:: python

    num_cpus_cluster()


.. parsed-literal::

    INFO | 2023-05-05 14:32:06,397 | Running num_cpus_cluster via gRPC
    INFO | 2023-05-05 14:32:06,766 | Time to send message: 0.37 seconds




.. parsed-literal::

    'Num cpus: 8'



Saving, Reusing, and Sharing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Runhouse supports saving down the metadata and configs for resources
like clusters and functions, so that you can load them from a different
environment, or share it with your collaborators.

.. code:: python

    num_cpus_cluster.save()


.. parsed-literal::

    INFO | 2023-05-05 14:32:31,248 | Saving config to RNS: {'name': '/carolineechen/cpu-cluster', 'resource_type': 'cluster', 'resource_subtype': 'OnDemandCluster', 'instance_type': 'CPU:8', 'num_instances': None, 'provider': 'cheapest', 'autostop_mins': 30, 'use_spot': False, 'image_id': None, 'region': None, 'sky_state': {'name': 'cpu-cluster', 'launched_at': 1683295614, 'handle': {'cluster_name': 'cpu-cluster', 'cluster_yaml': '~/.sky/generated/cpu-cluster.yml', 'head_ip': '3.87.203.10', 'launched_nodes': 1, 'launched_resources': {'cloud': 'AWS', 'instance_type': 'm6i.2xlarge', 'use_spot': False, 'disk_size': 256, 'region': 'us-east-1', 'zone': 'us-east-1a'}}, 'last_use': '/usr/local/lib/python3.10/dist-packages/ipykernel_launcher.py -f /root/.local/share/jupyter/runtime/kernel-729e54ec-f20d-48a4-8603-099468cb0df6.json', 'status': 'UP', 'autostop': 30, 'to_down': True, 'owner': 'AIDASQMZKHMBGKPSNXGMZ', 'metadata': {}, 'cluster_hash': 'b5ff32eb-425d-42af-ac6c-801be1f399de', 'public_key': '~/.ssh/sky-key.pub', 'ssh_creds': {'ssh_user': 'ubuntu', 'ssh_private_key': '~/.ssh/sky-key', 'ssh_control_name': 'cpu-cluster', 'ssh_proxy_command': None}}}
    INFO | 2023-05-05 14:32:32,079 | Config updated in RNS for Runhouse URI <resource/carolineechen:cpu-cluster>
    INFO | 2023-05-05 14:32:32,083 | Saving config to RNS: {'name': '/carolineechen/num_cpus_cluster', 'resource_type': 'function', 'resource_subtype': 'Function', 'system': '/carolineechen/cpu-cluster', 'reqs': ['./'], 'setup_cmds': [], 'fn_pointers': ('content', 'num_cpus_fn', 'num_cpus')}
    INFO | 2023-05-05 14:32:32,871 | Saving new resource in RNS for Runhouse URI <resource/carolineechen:num_cpus_cluster>




.. parsed-literal::

    <runhouse.rns.function.Function at 0x7fb3b7ca1ff0>



.. code:: python

    num_cpus_cluster.share(
        users=["<email_to_runhouse_account>"],
        access_type="write",
    )

Now, you, or whoever you shared it with, can reload this function from
anther dev environment (like a different Colab, local, or on a cluster),
as long as you are logged in to your Runhouse account.

.. code:: python

    reloaded_function = rh.function(name="num_cpus_cluster")
    reloaded_function()


.. parsed-literal::

    INFO | 2023-05-05 14:32:34,922 | Attempting to load config for /carolineechen/num_cpus_cluster from RNS.
    INFO | 2023-05-05 14:32:35,708 | Attempting to load config for /carolineechen/cpu-cluster from RNS.
    INFO | 2023-05-05 14:32:36,785 | Setting up Function on cluster.
    INFO | 2023-05-05 14:32:48,041 | Copying local package content to cluster <cpu-cluster>
    INFO | 2023-05-05 14:32:50,491 | Installing packages on cluster cpu-cluster: ['./']
    INFO | 2023-05-05 14:32:50,862 | Function setup complete.
    INFO | 2023-05-05 14:32:50,863 | Running num_cpus_cluster via gRPC
    INFO | 2023-05-05 14:32:51,271 | Time to send message: 0.41 seconds




.. parsed-literal::

    'Num cpus: 8'



Terminate the Cluster
~~~~~~~~~~~~~~~~~~~~~

To terminate the cluster, you can run:

.. code:: python

    cluster.teardown()


Summary
~~~~~~~

In this tutorial, we demonstrated how to use runhouse to create
references to remote clusters, run local functions on the cluster, and
save/share and reuse functions with a Runhouse account.

Runhouse also lets you: - Send and save data (folders, blobs, tables)
between local, remote, and file storage - Send, save, and share dev
environments - Reload and reuse saved resources (both compute and data)
from different environments (with a Runhouse account) - … and much more!
