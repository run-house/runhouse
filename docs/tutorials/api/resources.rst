Resource Management
===================

.. raw:: html

    <p><a href="https://colab.research.google.com/github/run-house/runhouse/blob/stable/docs/notebooks/api/resources.ipynb">
    <img height="20px" width="117px" src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a></p>

Runhouse provides a suite of accessibility features that makes it easy
to keep track of and access your data, code, or secrets from anywhere.
The Runhouse RNS (resource naming system) keeps track of lightweight
metadata for your Resources, making it possible to save, reload, and
reuse them.

Anyone can take advantage of these accessibility features locally, and
by creating a (free) Runhouse account, you can further access your
resources and secrets from any environment or device you log into.

This tutorial covers the following topics: 1. Configs 2. Local RNS 3.
Runhouse RNS

.. code:: ipython3

    import runhouse as rh

Setting Config Options
----------------------

Runhouse stores user configs both locally in ``~/.rh/config.yaml`` and
remotely in the Runhouse database, letting you preserve your same config
across environments.

Some configs to consider setting:

- ``rh.configs.set('use_spot', True)``: Whether to use spot instances,
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
into the ``rh`` folder of the current Git working directory.

If you are not logged into a Runhouse account, calling ``.save()`` will
save down resources locally by default. If you are logged into a
Runhouse account however, Resources will be saved into Runhouse RNS by
default, so if you would like to specify creating a local resource, you
can do so by explicitly setting the resource name to begin with ``~/``
to signal that it lives in the current folder.

.. code:: ipython3

    my_resource = rh.ondemand_cluster(name='~/aws_cluster', instance_type='V100:1', provider='aws')
    my_resource.save()


.. parsed-literal::
    :class: code-output

    INFO | 2023-06-21 22:15:57,611 | Saving config for ~/aws_cluster to: /Users/caroline/Documents/runhouse/runhouse/rh/aws_cluster/config.json




.. parsed-literal::
    :class: code-output

    <runhouse.resources.hardware.on_demand_cluster.OnDemandCluster at 0x1661c7040>



.. code:: ipython3

    !cat /Users/caroline/Documents/runhouse/runhouse/rh/aws_cluster/config.json


.. parsed-literal::
    :class: code-output

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
    :class: code-output

    INFO | 2023-06-21 22:20:03,710 | Loading config from local file /Users/caroline/Documents/runhouse/runhouse/rh/aws_cluster/config.json




.. parsed-literal::
    :class: code-output

    <runhouse.resources.hardware.on_demand_cluster.OnDemandCluster at 0x1231023d0>



.. code:: ipython3

    rh.cluster(name="~/aws_cluster")


.. parsed-literal::
    :class: code-output

    INFO | 2023-06-21 22:20:20,156 | Loading config from local file /Users/caroline/Documents/runhouse/runhouse/rh/aws_cluster/config.json




.. parsed-literal::
    :class: code-output

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
    :class: code-output

    INFO | 2023-06-21 22:38:05,351 | Creating new s3 folder if it does not already exist in path: /runhouse-blob/d57201aa760b4893800c7e3782117b3b/carolineechen
    INFO | 2023-06-21 22:38:05,368 | Found credentials in shared credentials file: ~/.aws/credentials
    INFO | 2023-06-21 22:38:06,305 | Creating new s3 folder if it does not already exist in path: /runhouse-blob/d57201aa760b4893800c7e3782117b3b/carolineechen
    INFO | 2023-06-21 22:38:06,462 | Saving config to RNS: {'name': '/carolineechen/my_blob', 'resource_type': 'blob', 'resource_subtype': 'Blob', 'path': '/runhouse-blob/d57201aa760b4893800c7e3782117b3b/carolineechen/my_blob', 'system': 's3'}
    INFO | 2023-06-21 22:38:07,078 | Config updated in RNS for Runhouse URI <resource/carolineechen:my_blob>




.. parsed-literal::
    :class: code-output

    <runhouse.resources.blob.Blob at 0x16703ee80>



This resource can then be reloaded and reused not only from local, but
also from any other environment, cluster, or device that you’re logged
into!

.. code:: ipython3

    del my_resource

    loaded = rh.load("my_blob")
    pickle.loads(loaded.data)


.. parsed-literal::
    :class: code-output

    INFO | 2023-06-21 22:38:10,598 | Attempting to load config for /carolineechen/my_blob from RNS.
    INFO | 2023-06-21 22:38:10,936 | Creating new s3 folder if it does not already exist in path: /runhouse-blob/d57201aa760b4893800c7e3782117b3b/carolineechen
    INFO | 2023-06-21 22:38:10,970 | Found credentials in shared credentials file: ~/.aws/credentials




.. parsed-literal::
    :class: code-output

    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]



The portability is extended to any teammates or person you want to share
your resource with. Simply call ``.share()`` on the resource, and pass
in the emails (must be associated with a Runhouse account) of the people
to share it with. Further customize their resource access, and whether
to notify them.

.. code:: ipython3

    loaded.share(
        users=["teammate1@email.com"],
        access_level="write",
    )


.. parsed-literal::
    :class: code-output

    INFO | 2023-06-21 22:38:14,252 | Attempting to load config for /carolineechen/my_blob from RNS.




.. parsed-literal::
    :class: code-output

    ({}, {'teammate1@email.com': 'write'})
