Quick Start Guide
=================

.. raw:: html

    <p><a href="https://colab.research.google.com/github/run-house/runhouse/blob/stable/docs/notebooks/api/quick_start.ipynb">
    <img height="20px" width="117px" src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a></p>


This tutorials walks through Runhouse setup (installation, hardware
setup, and optional login) and goes through an example that demonstrates
how to use Runhouse to bridge the gap between local and remote compute,
and create Resources that can be saved, reused, and shared.

Installation
------------

Runhouse can be installed with:

.. code:: ipython3

    !pip install runhouse

If using Runhouse with a cloud provider, you can additionally install
cloud packages (e.g. the right versions of tools like boto, gsutil,
etc.):

.. code:: shell

   $ pip install "runhouse[aws]"
   $ pip install "runhouse[gcp]"
   $ pip install "runhouse[azure]"
   $ pip install "runhouse[sagemaker]"
   # Or
   $ pip install "runhouse[all]"

To import runhouse:

.. code:: ipython3

    import runhouse as rh

.. code:: ipython3

    # Optional: to sync over secrets from your Runhouse account
    # !runhouse login

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
needed. Just have your cluster IP address and path to SSH credentials or
password ready:

.. code:: ipython3

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

.. note::

    For more information see the :ref:`Cluster Class` section.

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

.. code:: ipython3

    # SkyPilot CLI
    !sky check

.. code:: ipython3

    # Runhouse Secrets
    # Lambda Labs:
    rh.provider_secret("lambda", values={"api_key": "*******"}).write()

    # AWS:
    rh.provider_secret("aws", values={"access_key": "******", "secret_key": "*******"}).write()

    # GCP:
    !gcloud init
    !gcloud auth application-default login
    !cp -r /content/.config/* ~/.config/gcloud

    # Azure
    !az login
    !az account set -s <subscription_id>

To check that the provider credentials are properly configured locally,
run ``sky check`` to confirm that the cloud provider is enabled

.. code:: ipython3

    !sky check

To create a cluster instance, use the ``rh.cluster()`` factory function
for an existing cluster, or ``rh.ondemand_cluster`` for an on-demand
cluster. We go more in depth about how to launch the cluster, and run a
function on it later in this tutorial.

.. code:: ipython3

    cluster = rh.ondemand_cluster(
                  name="cpu-cluster",
                  instance_type="CPU:8",
                  provider="cheapest",      # options: "AWS", "GCP", "Azure", "Lambda", or "cheapest"
              ).save()

.. note::

    For more information and hardware setup see the :ref:`OnDemandCluster Class` section.

SageMaker Cluster
~~~~~~~~~~~~~~~~~

Runhouse facilitates easy access to existing or new SageMaker compute.
Just provide your SageMaker execution role ARN, profile name, or have it configured in your local environment.

.. code:: ipython3

    # Create a SageMaker instance which uses the "sagemaker" profile configured locally
    # Once the instance is up, we can run a function or send other resources to the SageMaker compute, just as we would
    # for any other cluster type.
    cluster = rh.sagemaker_cluster(name='sm-cluster', profile="sagemaker").save()

    # Alternatively, provide an Estimator and launch a dedicated training job once the cluster is launched
    pytorch_estimator = PyTorch(entry_point='train.py',
                                role='arn:aws:iam::123456789012:role/MySageMakerRole',
                                source_dir='/Users/myuser/dev/sagemaker',
                                framework_version='1.8.1',
                                py_version='py36',
                                instance_type='ml.p3.2xlarge')

    cluster = rh.sagemaker_cluster(name='sagemaker-cluster',
                                   estimator=pytorch_estimator).save()

.. note::

    For more information and hardware setup see the :ref:`SageMakerCluster Class` section.

Secrets and Portability
-----------------------

Using Runhouse with only the OSS Python package is perfectly fine, but
you can unlock some unique portability features by creating an (always
free) `account <https://www.run.house/account>`__ and saving down your secrets
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

Some notes on security:

* Our API servers only ever store light metadata about your resources (e.g. folder name, cloud provider, storage bucket, path). All actual data and compute stays inside your own cloud account and never hits our servers.
* Secrets are stored in `Hashicorp Vault <https://www.vaultproject.io/>`__ (an industry standard for secrets management), never on our API servers, and our APIs simply call into Vault’s APIs.

.. code:: ipython3

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
`Installation <installation_>`_ and `Cluster Setup <#cluster-setup>`_ sections above prior to running this
example.

.. code:: ipython3

    import runhouse as rh

Running local functions on remote hardware
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

First let’s define a simple local function which returns the number of
CPUs available.

.. code:: ipython3

    def num_cpus():
        import multiprocessing
        return f"Num cpus: {multiprocessing.cpu_count()}"

    num_cpus()




.. parsed-literal::
    :class: code-output

    'Num cpus: 10'



Next, instantiate the cluster that we want to run this function on. This
can be either an existing cluster where you pass in an IP address and
SSH credentials, or a cluster associated with supported Cloud account
(AWS, GCP, Azure, LambdaLabs), where it is automatically launched (and
optionally terminated) for you.

.. code:: ipython3

    # Using an existing, bring-your-own cluster
    cluster = rh.cluster(
                  name="cpu-cluster",
                  ips=['<ip of the cluster>'],
                  ssh_creds={'ssh_user': '<user>', 'ssh_private_key':'<path_to_key>'},
              )

    # Using a Cloud provider
    cluster = rh.cluster(
                  name="cpu-cluster",
                  instance_type="CPU:8",
                  provider="cheapest",      # options: "AWS", "GCP", "Azure", "Lambda", or "cheapest"
                  autostop_mins=60,         # optional, defaults to default_autostop_mins; -1 suspends autostop
              )

If using a cloud cluster, we can launch the cluster with ``.up()`` or
``.up_if_not()``.

Note that it may take a few minutes for the cluster to be launched
through the Cloud provider and set up dependencies.

.. code:: ipython3

    cluster.up_if_not()

Now that we have our function and remote cluster set up, we’re ready to
see how to run this function on our cluster!

We wrap our local function in ``rh.function``, and associate this new
function with the cluster. Now, whenever we call this new function, just
as we would call any other Python function, it runs on the cluster
instead of local.

.. code:: ipython3

    num_cpus_cluster = rh.function(name="num_cpus_cluster", fn=num_cpus).to(system=cluster, reqs=["./"])


.. parsed-literal::
    :class: code-output

    INFO | 2023-08-29 03:03:52.826786 | Writing out function function to /Users/caroline/Documents/runhouse/runhouse/docs/notebooks/basics/num_cpus_fn.py. Please make sure the function does not rely on any local variables, including imports (which should be moved inside the function body).
    /Users/caroline/Documents/runhouse/runhouse/runhouse/rns/function.py:106: UserWarning: ``reqs`` and ``setup_cmds`` arguments has been deprecated. Please use ``env`` instead.
      warnings.warn(
    INFO | 2023-08-29 03:03:52.832445 | Setting up Function on cluster.
    INFO | 2023-08-29 03:03:53.271019 | Connected (version 2.0, client OpenSSH_8.2p1)
    INFO | 2023-08-29 03:03:53.546892 | Authentication (publickey) successful!
    INFO | 2023-08-29 03:03:53.557504 | Checking server cpu-cluster
    INFO | 2023-08-29 03:03:54.942843 | Server cpu-cluster is up.
    INFO | 2023-08-29 03:03:54.948097 | Copying package from file:///Users/caroline/Documents/runhouse/runhouse to: cpu-cluster
    INFO | 2023-08-29 03:03:56.480770 | Calling env_20230829_030349.install


.. parsed-literal::
    :class: code-output

    base servlet: Calling method install on module env_20230829_030349
    Installing package: Package: runhouse
    Installing Package: runhouse with method reqs.
    reqs path: runhouse/requirements.txt
    pip installing requirements from runhouse/requirements.txt with: -r runhouse/requirements.txt
    Running: /opt/conda/bin/python3.10 -m pip install -r runhouse/requirements.txt


.. parsed-literal::
    :class: code-output

    INFO | 2023-08-29 03:03:58.230209 | Time to call env_20230829_030349.install: 1.75 seconds
    INFO | 2023-08-29 03:03:58.462054 | Function setup complete.


.. code:: ipython3

    num_cpus_cluster()


.. parsed-literal::
    :class: code-output

    INFO | 2023-08-29 03:04:01.105011 | Calling num_cpus_cluster.call


.. parsed-literal::
    :class: code-output

    base servlet: Calling method call on module num_cpus_cluster


.. parsed-literal::
    :class: code-output

    INFO | 2023-08-29 03:04:01.384439 | Time to call num_cpus_cluster.call: 0.28 seconds




.. parsed-literal::
    :class: code-output

    'Num cpus: 8'



Saving, Reusing, and Sharing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Runhouse supports saving down the metadata and configs for resources
like clusters and functions, so that you can load them from a different
environment, or share it with your collaborators.

.. code:: ipython3

    num_cpus_cluster.save()


.. parsed-literal::
    :class: code-output

    <runhouse.resources.function.Function at 0x104634ee0>



.. code:: ipython3

    num_cpus_cluster.share(
        users=["<email_to_runhouse_account>"],
        access_level="write",
    )

Now, you, or whoever you shared it with, can reload this function from
another dev environment (like a different Colab, local, or on a cluster),
as long as you are logged in to your Runhouse account.

.. code:: ipython3

    reloaded_function = rh.function(name="num_cpus_cluster")
    reloaded_function()


.. parsed-literal::
    :class: code-output

    INFO | 2023-08-29 03:04:24.820884 | Checking server cpu-cluster
    INFO | 2023-08-29 03:04:25.850301 | Server cpu-cluster is up.
    INFO | 2023-08-29 03:04:25.852478 | Calling num_cpus_cluster.call


.. parsed-literal::
    :class: code-output

    base servlet: Calling method call on module num_cpus_cluster


.. parsed-literal::
    :class: code-output

    INFO | 2023-08-29 03:04:26.127098 | Time to call num_cpus_cluster.call: 0.27 seconds




.. parsed-literal::
    :class: code-output

    'Num cpus: 8'



Terminate the Cluster
~~~~~~~~~~~~~~~~~~~~~

To terminate the cluster, you can run:

.. code:: ipython3

    cluster.teardown()



.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #008000; text-decoration-color: #008000">⠇</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">Terminating </span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">cpu-cluster</span>
    </pre>


Summary
~~~~~~~

In this tutorial, we demonstrated how to use runhouse to create
references to remote clusters, run local functions on the cluster, and
save/share and reuse functions with a Runhouse account.

Runhouse also lets you:

- Send and save data (folders, blobs, tables) between local, remote, and file storage
- Send, save, and share dev environments
- Reload and reuse saved resources (both compute and data) from different environments (with a Runhouse account)
- … and much more!
