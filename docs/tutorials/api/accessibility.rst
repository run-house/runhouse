Accessibility: Resource and Secrets Management
==============================================


.. raw:: html

    <p><a href="https://colab.research.google.com/github/run-house/runhouse/blob/stable/docs/notebooks/basics/accessibility.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a></p>

Runhouse provides a suite of accessibility features that makes it easy
to keep track of and access your data, code, or secrets from anywhere.
The Runhouse RNS (resource naming system) keeps track of lightweight
metadata for your Resources, making it possible to save, reload, and
reuse them.

Anyone can take advantage of these accessibility features locally, and
by creating a (free) Runhouse account, you can further access your
resources and secrets from any environment or device you log into.

This tutorial covers the following topics:

1. Secrets (Credentials) Management
2. Configs
3. Local RNS
4. Runhouse RNS

.. code:: ipython3

    import runhouse as rh


Secrets Management
------------------

The Secrets API provides a simple interface for storing and retrieving
secrets to a allow a more seamless experience when accessing resources
across environments. It provides a simple interface for storing and
retrieving secrets from a variety of providers (e.g. AWS, Azure, GCP,
Hugging Face, Github, etc.) as well as SSH Keys and custom secrets, and
stores them in Hashicorp Vault (and never on Runhouse servers).

The
`API <https://runhouse-docs.readthedocs-hosted.com/en/main/api/python/secrets.html>`__
handles secrets interactions between

* config files
* environment variables
* Python variables
* Vault

To use secrets locally without creating account, you can use
``rh.Secrets.save_provider_secrets()`` to properly sync down your
credentials into the proper local default paths expected by Runhouse, to
perform operations such as launching your cloud clusters.

.. code:: ipython3

    rh.Secrets.save_provider_secrets(secrets={
        "aws": {"access_key": "******", "secret_key": "*******"},
        "lambda": {"api_key": "******"}
    })

If you have a runhouse account, which you can create
`here <run.house/login>`__ or by calling either the ``runhouse login``
CLI command or ``rh.login()`` Python command, you can sync secrets (to
Vault) associated your account, and download existing secrets or upload
new secrets from your environment.

.. code:: ipython3

    # show supported builtin providers
    rh.Secrets.builtin_providers()


.. parsed-literal::

    [<runhouse.rns.secrets.aws_secrets.AWSSecrets at 0x106743b20>,
     <runhouse.rns.secrets.azure_secrets.AzureSecrets at 0x1067439d0>,
     <runhouse.rns.secrets.gcp_secrets.GCPSecrets at 0x123cb9cd0>,
     <runhouse.rns.secrets.huggingface_secrets.HuggingFaceSecrets at 0x123cb9af0>,
     <runhouse.rns.secrets.lambda_secrets.LambdaSecrets at 0x123cb9e50>,
     <runhouse.rns.secrets.sky_secrets.SkySecrets at 0x123cd37c0>,
     <runhouse.rns.secrets.ssh_secrets.SSHSecrets at 0x123cd3880>,
     <runhouse.rns.secrets.github_secrets.GitHubSecrets at 0x123cd38e0>]



.. code:: ipython3

    # Upload secrets into Vault
    rh.Secrets.save_provider_secrets(secrets={"azure": {"subscription_id": "12345"}})
    rh.Secrets.download_into_env()

    !cat ~/.azure/clouds.config


.. parsed-literal::

    WARNING | 2023-06-21 08:03:55,081 | Received secrets ['azure'] which Runhouse did not auto-detect as configured. For cloud providers, you may want to run `sky check` to double check that they're enabled and to see instructions on how to enable them.
    INFO | 2023-06-21 08:03:55,084 | Getting secrets from Vault.
    WARNING | 2023-06-21 08:03:56,614 | Key id_rsa already exists, skipping.
    WARNING | 2023-06-21 08:03:56,615 | Key id_rsa.pub already exists, skipping.
    WARNING | 2023-06-21 08:03:56,970 | Received secrets ['gcp', 'lambda'] which Runhouse did not auto-detect as configured. For cloud providers, you may want to run `sky check` to double check that they're enabled and to see instructions on how to enable them.
    INFO | 2023-06-21 08:03:56,972 | Saved secrets from Vault to local config files
    [AzureCloud]
    subscription = 12345



If you already have secrets configured locally in a config file or in
your environment, you can also use ``rh.Secrets.put()`` to upload them
into Vault.

.. code:: ipython3

    rh.Secrets.put("aws", from_env=True)
    # rh.Secrets.put("aws", file_path="~/.aws/credentials")

If you have secrets from Vault that you’d like to sync to local, you can
use ``rh.Secrets.download_into_env()`` to download them into your local
config files, or ``rh.Secrets.get("azure")`` to get the secrets
dictionary.

There are also options when logging in through ``runhouse login`` or
``rh.login()``, to choose which secrets you want to upload into Vault,
and which ones to download down from Vault.

When you logout with ``runhouse logout`` or ``rh.logout()``, you can
choose to remove locally saved secrets or delete them from Vault.

Setting Config Options
----------------------

Runhouse stores user configs both locally in ``~/.rh/config.yaml`` and
remotely in the Runhouse database, letting you preserve your same config
across environments.

Some configs to consider setting: \*
``rh.configs.set('use_spot', True)``: Whether to use spot instances,
which are cheaper but can be reclaimed at any time. This is ``False`` by
default, because you’ll need to request spot quota from the cloud
providers to use spot instances.

-  ``rh.configs.set('default_autostop', 30)``: Default autostop time (or
   -1 for indefinitely) for the on-demand cluster, to dynamically stop
   the cluster after inactivity to save money. You can also call
   ``cluster.keep_warm(autostop=60)`` to control this for an existing
   cluster.

-  ``rh.configs.set('default_provider', 'cheapest')``: Default cloud
   provider to use for your on-demand cluster, or ``cheapest`` selects
   the cheapest provider for the desired hardware.

To save updated configs to Runhouse, to be accessed from elsewhere:

.. code:: ipython3

    rh.configs.upload_defaults()

Local RNS
---------

The Local RNS is a git-based approach that allows for local persistence
and versioning, or sharing across OSS projects. It lets you publish the
exact resource metadata in the same version tree as your code, and can
be a highly visible way to publish distribute resources, such as cloud
configurations and data artifacts, to OSS users.

Local Resources live in the current local folder; they are saved down
into the `rh` folder of the current Git working directory.

If you are not logged into a Runhouse account, calling `.save()` will
save down resources locally by default. If you are logged into a Runhouse
account however, Resources will be saved into Runhouse RNS by default, so
if you would like to specify creating a local resource, you can do so by
explicitly setting the resource name to begin with `~/` to signal that it
lives in the current folder.

.. code:: ipython3

    my_resource = rh.ondemand_cluster(name='~/aws_cluster', instance_type='V100:1', provider='aws')
    my_resource.save()


.. parsed-literal::

    INFO | 2023-06-21 22:15:57,611 | Saving config for ~/aws_cluster to: /Users/caroline/Documents/runhouse/runhouse/rh/aws_cluster/config.json




.. parsed-literal::

    <runhouse.resources.hardware.on_demand_cluster.OnDemandCluster at 0x1661c7040>



.. code:: ipython3

    !cat /Users/caroline/Documents/runhouse/runhouse/rh/aws_cluster/config.json


.. parsed-literal::

    {
        "name": "~/aws_cluster",
        "resource_type": "cluster",
        "resource_subtype": "OnDemandCluster",
        "instance_type": "V100:1",
        "num_instances": null,
        "provider": "aws",
        "autostop_mins": 30,
        "use_spot": false,
        "image_id": null,
        "region": null,
        "sky_state": null
    }

To load a resource, you can call ``rh.load('resource_name')``, or use
the resource factory method, passing in only the name.

.. code:: ipython3

    del my_resource

    rh.load("~/aws_cluster")


.. parsed-literal::

    INFO | 2023-06-21 22:20:03,710 | Loading config from local file /Users/caroline/Documents/runhouse/runhouse/rh/aws_cluster/config.json




.. parsed-literal::

    <runhouse.resources.hardware.on_demand_cluster.OnDemandCluster at 0x1231023d0>



.. code:: ipython3

    rh.ondemand_cluster(name="~/aws_cluster")


.. parsed-literal::

    INFO | 2023-06-21 22:20:20,156 | Loading config from local file /Users/caroline/Documents/runhouse/runhouse/rh/aws_cluster/config.json




.. parsed-literal::

    <runhouse.resources.hardware.on_demand_cluster.OnDemandCluster at 0x12324b400>



Runhouse RNS
------------

The Runhouse RNS is a key-value metadata store that allows resources to
be shared across users or environments, and does not need to be backed
by Git. It works anywhere with an internet connection and Python
interpreter, making it more portable. The RNS is also backed by a
management dashboard to view and manage all resources, including
creation and update history.

To use the Runhouse RNS, you will need a `Runhouse
account <https://www.run.house/login>`__.

The following resource, whose name ``my_blob`` does not begin with
``~/``, will be saved into the Runhouse RNS.

.. code:: ipython3

    import pickle
    data = pickle.dumps(list(range(10)))

    my_resource = rh.blob(data, name="my_blob", system="s3").write()  # write data to linked s3
    my_resource.save()


.. parsed-literal::

    INFO | 2023-06-21 22:38:05,351 | Creating new s3 folder if it does not already exist in path: /runhouse-blob/d57201aa760b4893800c7e3782117b3b/carolineechen
    INFO | 2023-06-21 22:38:05,368 | Found credentials in shared credentials file: ~/.aws/credentials
    INFO | 2023-06-21 22:38:06,305 | Creating new s3 folder if it does not already exist in path: /runhouse-blob/d57201aa760b4893800c7e3782117b3b/carolineechen
    INFO | 2023-06-21 22:38:06,462 | Saving config to RNS: {'name': '/carolineechen/my_blob', 'resource_type': 'blob', 'resource_subtype': 'Blob', 'path': '/runhouse-blob/d57201aa760b4893800c7e3782117b3b/carolineechen/my_blob', 'system': 's3'}
    INFO | 2023-06-21 22:38:07,078 | Config updated in RNS for Runhouse URI <resource/carolineechen:my_blob>




.. parsed-literal::

    <runhouse.resources.blob.Blob at 0x16703ee80>



This resource can then be reloaded and reused not only from local, but
also from any other environment, cluster, or device that you’re logged
into!

.. code:: ipython3

    del my_resource

    loaded = rh.load("my_blob")
    pickle.loads(loaded.data)


.. parsed-literal::

    INFO | 2023-06-21 22:38:10,598 | Attempting to load config for /carolineechen/my_blob from RNS.
    INFO | 2023-06-21 22:38:10,936 | Creating new s3 folder if it does not already exist in path: /runhouse-blob/d57201aa760b4893800c7e3782117b3b/carolineechen
    INFO | 2023-06-21 22:38:10,970 | Found credentials in shared credentials file: ~/.aws/credentials




.. parsed-literal::

    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]



The portability is extended to any teammates or person you want to share
your resource with. Simply call ``.share()`` on the resource, and pass
in the emails (must be associated with a Runhouse account) of the people
to share it with. Further customize their resource access, and whether
to notify them.

.. code:: ipython3

    loaded.share(
        users=["teammate1@email.com"],
        access_type="write",
    )


.. parsed-literal::

    INFO | 2023-06-21 22:38:14,252 | Attempting to load config for /carolineechen/my_blob from RNS.




.. parsed-literal::

    ({}, {'teammate1@email.com': 'write'})
