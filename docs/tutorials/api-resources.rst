Resource Management
===================

.. raw:: html

    <p><a href="https://colab.research.google.com/github/run-house/notebooks/blob/stable/docs/api-resources.ipynb">
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

.. note::

    If you are logged in and would like to turn off automatically saving resources to Runhouse RNS, you can
    set ``save_to_den`` to ``false`` in your local ``~/.rh/config.yaml`` file.

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

Runhouse RNS, or Den, is a key-value metadata store that allows
resources to be shared across users or environments, and does not need
to be backed by Git. It works anywhere with an internet connection and
Python interpreter, making it more portable. The RNS is also backed by a
management dashboard to view and manage all resources, including
creation and update history.

To use Den you will need a `Runhouse
account <https://www.run.house/login>`__.

Simply call ``.save()`` on any Runhouse resource to save it to Den.

Below is an example of how you connect to an existing cluster, run
commands on the cluster remotely, and share the cluster for another user
to connect to.

.. code:: ipython3

    # Load a cluster which has already been launched and saved in Runhouse Den
    # rh.cluster(name="aws-cpu", provider="aws", instance_type="m6i.large").save()

    cpu_cluster = rh.cluster(name="/jlewitt1/aws-cpu")
    print(cpu_cluster.is_up())


.. parsed-literal::
    :class: code-output

    INFO | 2024-08-18 06:50:57.377788 | Running command on aws-cpu: echo "hello"



.. parsed-literal::
    :class: code-output

    Output()



.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"></pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">
    </pre>



.. parsed-literal::
    :class: code-output

    INFO | 2024-08-18 06:51:07.370306 | Running command on aws-cpu: echo "hello"


.. parsed-literal::
    :class: code-output

    True


.. code:: ipython3

    # Put an object into the cluster's object store and reload it
    cpu_cluster.put("k1", "v1")
    print(cpu_cluster.get("k1"))


.. parsed-literal::
    :class: code-output

    INFO | 2024-08-17 20:24:44.166333 | Running command on aws-cpu: echo "hello"
    INFO | 2024-08-17 20:24:48.699220 | Running forwarding command: ssh -T -L 32300:localhost:32300 -i ~/.ssh/sky-key -o Port=10022 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o IdentitiesOnly=yes -o ExitOnForwardFailure=yes -o ServerAliveInterval=5 -o ServerAliveCountMax=3 -o ConnectTimeout=30s -o ForwardAgent=yes -o ProxyCommand='ssh -T -L 32300:localhost:32300 -i ~/.ssh/sky-key -o Port=22 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o IdentitiesOnly=yes -o ExitOnForwardFailure=yes -o ServerAliveInterval=5 -o ServerAliveCountMax=3 -o ConnectTimeout=30s -o ForwardAgent=yes -W %h:%p ubuntu@3.14.144.103' root@localhost


.. parsed-literal::
    :class: code-output

    v1


This resource can then be reloaded and reused not only from local, but
also from any other environment, cluster, or device that you’re logged
into!

The portability is extended to any teammates or person you want to share
your resource with. Simply call ``.share()`` on the resource, and pass
in the emails (must be associated with a Runhouse account) of the people
to share it with. Further customize their resource access, and whether
to notify them.

.. code:: ipython3

    cpu_cluster.share(
        users=["teammate1@email.com"],
        access_level="write",
    )


.. parsed-literal::
    :class: code-output

    INFO | 2024-08-18 06:51:39.797150 | Saving config for aws-cpu-ssh-secret to Den
    INFO | 2024-08-18 06:51:39.972763 | Saving secrets for aws-cpu-ssh-secret to Vault
    INFO | 2024-08-18 06:51:40.190996 | Saving config to RNS: {'name': '/jlewitt1/aws-cpu_default_env', 'resource_type': 'env', 'resource_subtype': 'Env', 'provenance': None, 'visibility': 'private', 'env_vars': {}, 'env_name': 'aws-cpu_default_env', 'compute': {}, 'reqs': ['ray==2.30.0'], 'working_dir': None}
    INFO | 2024-08-18 06:51:40.368442 | Saving config to RNS: {'name': '/jlewitt1/aws-cpu', 'resource_type': 'cluster', 'resource_subtype': 'OnDemandCluster', 'provenance': None, 'visibility': 'private', 'ips': ['3.14.144.103'], 'server_port': 32300, 'server_connection_type': 'ssh', 'den_auth': False, 'ssh_port': 22, 'client_port': 32300, 'creds': '/jlewitt1/aws-cpu-ssh-secret', 'api_server_url': 'https://api.run.house', 'default_env': '/jlewitt1/aws-cpu_default_env', 'instance_type': 'CPU:2+', 'provider': 'aws', 'open_ports': [], 'use_spot': False, 'image_id': 'docker:nvcr.io/nvidia/pytorch:23.10-py3', 'region': 'us-east-2', 'stable_internal_external_ips': [('172.31.5.134', '3.14.144.103')], 'sky_kwargs': {'launch': {'retry_until_up': True}}, 'launched_properties': {'cloud': 'aws', 'instance_type': 'm6i.large', 'region': 'us-east-2', 'cost_per_hour': 0.096, 'docker_user': 'root'}, 'autostop_mins': -1}
    INFO | 2024-08-18 06:51:40.548233 | Sharing cluster credentials, which enables the recipient to SSH into the cluster.
    INFO | 2024-08-18 06:51:40.551277 | Saving config for aws-cpu-ssh-secret to Den
    INFO | 2024-08-18 06:51:40.728345 | Saving secrets for aws-cpu-ssh-secret to Vault
    INFO | 2024-08-18 06:51:41.150745 | Saving config to RNS: {'name': '/jlewitt1/aws-cpu_default_env', 'resource_type': 'env', 'resource_subtype': 'Env', 'provenance': None, 'visibility': 'private', 'env_vars': {}, 'env_name': 'aws-cpu_default_env', 'compute': {}, 'reqs': ['ray==2.30.0'], 'working_dir': None}
    INFO | 2024-08-18 06:51:42.006030 | Saving config for aws-cpu-ssh-secret to Den
    INFO | 2024-08-18 06:51:42.504070 | Saving secrets for aws-cpu-ssh-secret to Vault
    INFO | 2024-08-18 06:51:42.728653 | Saving config to RNS: {'name': '/jlewitt1/aws-cpu_default_env', 'resource_type': 'env', 'resource_subtype': 'Env', 'provenance': None, 'visibility': 'private', 'env_vars': {}, 'env_name': 'aws-cpu_default_env', 'compute': {}, 'reqs': ['ray==2.30.0'], 'working_dir': None}
    INFO | 2024-08-18 06:51:42.906615 | Saving config to RNS: {'name': '/jlewitt1/aws-cpu', 'resource_type': 'cluster', 'resource_subtype': 'OnDemandCluster', 'provenance': None, 'visibility': 'private', 'ips': ['3.14.144.103'], 'server_port': 32300, 'server_connection_type': 'ssh', 'den_auth': False, 'ssh_port': 22, 'client_port': 32300, 'creds': '/jlewitt1/aws-cpu-ssh-secret', 'api_server_url': 'https://api.run.house', 'default_env': '/jlewitt1/aws-cpu_default_env', 'instance_type': 'CPU:2+', 'provider': 'aws', 'open_ports': [], 'use_spot': False, 'image_id': 'docker:nvcr.io/nvidia/pytorch:23.10-py3', 'region': 'us-east-2', 'stable_internal_external_ips': [('172.31.5.134', '3.14.144.103')], 'sky_kwargs': {'launch': {'retry_until_up': True}}, 'launched_properties': {'cloud': 'aws', 'instance_type': 'm6i.large', 'region': 'us-east-2', 'cost_per_hour': 0.096, 'docker_user': 'root'}, 'autostop_mins': -1}




.. parsed-literal::
    :class: code-output

    ({}, {'teammate1@email.com': 'write'}, ['teammate1@email.com'])
