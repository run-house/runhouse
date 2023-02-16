Clusters
====================================

A :ref:`Cluster` represents a set of machines which can be sent code or data, or a machine spec that could be spun up in the
event that we have some code or data to function to the machine.
Generally they are `Ray clusters <https://docs.ray.io/en/latest/cluster/getting-started.html>`_ under the hood.

There are a few kinds of clusters today:

BYO Cluster
~~~~~~~~~~~
This is a machine or group of machines specified by IP addresses and SSH credentials, which can be dispatched code
or data through the Runhouse APIs. This is useful if you have an on-prem instance, or an account with `Paperspace <https://www.paperspace.com/>`_,
`CoreWeave <https://www.coreweave.com/>`_, or another vertical provider, or simply want to spin up machines
yourself through the cloud UI.

You can use the :ref:`Cluster Factory Method` constructor like so:

.. code-block:: python

    gpu = rh.cluster(ips=['<ip of the cluster>'],
                     ssh_creds={'ssh_user': '...', 'ssh_private_key':'<path_to_key>'},
                     name='rh-a10x')


On-Demand Clusters
~~~~~~~~~~~~~~~~~~
Runhouse can spin up and down boxes for you as needed using `SkyPilot <https://github.com/skypilot-org/skypilot/>`_.
When you define a SkyPilot "cluster,"
you're primarily defining the configuration for us to spin up the compute resources on-demand.
When someone then calls a function or similar, we'll spin the box back up for you. You can also create these through the
cluster factory constructor:

You can use the cluster factory constructor like so:

.. code-block:: python

    gpu = rh.cluster(name='rh-4-a100s',
                     instance_type='A100:4',    # Can also be 'CPU:8' or cloud-specific strings, like 'g5.2xlarge'
                     provider='gcp',            # defaults to default_provider or cheapest if left empty
                     autostop_mins=-1,          # Defaults to 30 mins or default_autostop_mins, -1 suspends autostop
                     use_spot=True,             # You must have spot quota approved to use this
                     image_id='my_ami_string',  # Generally defaults to basic deep-learning AMIs through SkyPilot
                     region='us-east-1'         # Looks for cheapest on your continent if empty
                     )



SkyPilot also provides an excellent suite of CLI commands for basic instance management operations.
Some important ones are:

:code:`sky status --refresh`: Get the status of the clusters you launched from this machine.
This will not pull the status for all the machines you've launched from various environments.
We plan to add this feature soon.

:code:`sky down --all`: This will take down (terminate, without persisting the disk image) all clusters in the local
SkyPilot context (the ones that show when you run sky status --refresh). However, the best way to confirm that you
don't have any machines left running is always to check the cloud provider's UI.

:code:`sky down <cluster_name>`: This will take down a specific cluster.

:code:`ssh <cluster_name>`: This will ssh into the head node of the cluster.
SkyPilot cleverly adds the host information to your :code:`~/.ssh/config file`, so ssh will just work.

:code:`sky autostop -i <minutes, or -1> <cluster_name>`: This will set the cluster to autostop after that many minutes of inactivity.
By default this number is 10 minutes, but you can set it to -1 to disable autostop entirely. You can set your default autostop in :code:`~/.rh/config.yaml`.


Existing Clusters
~~~~~~~~~~~~~~~~~~
"Existing cluster" can mean either a saved :ref:`OnDemandCluster` config, which will be brought back up if needed,
or a BYO or OnDemandCluster that's already up. If you save the Cluster to the :ref:`Resource Name System (RNS)`,
you'll be able to dispatch to it from any environment. Multiple users or environments can function requests to a cluster
without issue, and either the OS or Ray (depending on the call to the cluster) will handle the resource contention.

You can load an existing cluster by name from local or Runhouse RNS simply by:

.. code-block:: python

    gpu = rh.cluster(name='~/my-local-a100')
    gpu = rh.cluster(name='@/my-a100-in-rh-rns')
    gpu = rh.cluster(name='^rh-v100')  # Loads a builtin cluster config

    # or, if you just want to load the Cluster object without refreshing its status
    gpu = rh.cluster(name='^rh-v100', dryrun=True)



Advanced Cluster Usage
~~~~~~~~~~~~~~~~~~~~~~

To start an ssh session into the cluster so you can poke around or debug:

.. code-block:: console

    $ ssh rh-v100

or in Python:

.. code-block:: python

    my_cluster.ssh()
    # or
    # my_function.ssh()

If you prefer to work in notebooks, you can tunnel a JupyterLab server into your local browser:

.. code-block:: console

    $ runhouse notebook my_cluster

or in Python:

.. code-block:: python

    my_function.notebook()
    # or
    my_cluster.notebook()


The :ref:`Notebooks` section goes more in depth on notebooks.

To run a shell command on the cluster:

.. code-block:: python

    gpu.run(['git clone ...', 'pip install ...'])

This is useful for installing more complex dependencies. :code:`gpu.run_setup(...)` will make sure the command is
only run once when the cluster is first created.

To run any Python on the cluster:

.. code-block:: python

    gpu.run_python(['import torch', 'print(torch.__version__)'])

This is useful for debugging, or for running a script that you don't want to function to the cluster
(e.g. because it has too many dependencies).

If you want to run an application on the cluster that requires a port to be open,
e.g. `Tensorboard <https://www.tensorflow.org/tensorboard/>`_, `Gradio <https://gradio.app/>`_.

.. code-block:: python

    gpu.ssh_tunnel(local_port=7860, remote_port=7860)
