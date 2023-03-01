Compute
====================================

The :ref:`Function`, :ref:`Cluster`, and :ref:`Package` APIs allow a seamless flow of code and execution across local and remote compute.
They blur the line between program execution and deployment, providing both a path of least resistence for running a
sub-routine on specific hardware, while unceremoniously turning that sub-routine into a reusable service.

They also provide convenient dependency isolation and management, provider-agnostic provisioning and termination,
and rich debugging and accessibility interfaces built-in.

Functions
---------

Runhouse allows you to send function code to a cluster, but still interact with it as a native runnable :ref:`Function` object.
When you do this, the following steps occur:

1. We check if the cluster is up, and bring up the cluster if not (only possible for :ref:`OnDemandClusters <OnDemandCluster>`)
2. We check that the cluster's gRPC server has started to handle requests to do things like install packages, run modules, get previously executed results, etc. If it hasn't, we install Runhouse on the cluster and start the gRPC server. The gRPC server initializes Ray.
3. We collect the dependencies from the :code:`reqs` parameter and install them on the cluster via :code:`cluster.install_packages()`. By default, we'll sync over the working git repo and install its :code:`requirements.txt` if it has one.


When you run your function module, we send a gRPC request to the cluster with the module name and function entrypoint to run.
The gRPC server adds the module to its python path, imports the module, grabs the function entrypoint, runs it,
and returns your results.

We plan to support additional form factors for modules beyond "remote Python function" shortly, including HTTP endpoints, custom ASGIs, and more.


Clusters
--------
A :ref:`Cluster` represents a set of machines which can be sent code or data, or a machine spec that could be spun up in the
event that we have some code or data to send to the machine.
Generally they are `Ray clusters <https://docs.ray.io/en/latest/cluster/getting-started.html>`_ under the hood.

There are a few kinds of clusters today:

BYO Cluster
~~~~~~~~~~~
This is a machine or group of machines specified by IP addresses and SSH credentials, which can be dispatched code
or data through the Runhouse APIs. This is useful if you have an on-prem instance, or an account with `Paperspace <https://www.paperspace.com/>`_,
`CoreWeave <https://www.coreweave.com/>`_, or another vertical provider, or simply want to spin up machines
yourself through the cloud UI.


On-Demand Clusters
~~~~~~~~~~~~~~~~~~
Runhouse can spin up and down boxes for you as needed using `SkyPilot <https://github.com/skypilot-org/skypilot/>`_.
When you define a SkyPilot "cluster,"
you're primarily defining the configuration for us to spin up the compute resources on-demand.
When someone then calls a function or similar, we'll spin the box back up for you.

SkyPilot also provides an excellent suite of CLI commands for basic instance management operations.
Some important ones are:

:code:`sky status --refresh`: Get the status of the clusters you launched from this machine.
This will not pull the status for all the machines you've launched from various environments.
We plan to add this feature soon.

:code:`sky down --all`: This will take down (terminate, without persisting the disk image) all clusters in the local
SkyPilot context (the ones that show when you run :code:`sky status --refresh`). However, the best way to confirm that you
don't have any machines left running is always to check the cloud provider's UI.

:code:`sky down <cluster_name>`: This will take down a specific cluster.

:code:`ssh <cluster_name>`: This will ssh into the head node of the cluster.
SkyPilot cleverly adds the host information to your :code:`~/.ssh/config file`, so ssh will just work.

:code:`sky autostop -i <minutes, or -1> <cluster_name>`: This will set the cluster to autostop after that many minutes of inactivity.
You can set it to -1 to disable autostop entirely. You can set your default autostop in :code:`~/.rh/config.yaml`.


Existing Clusters
~~~~~~~~~~~~~~~~~~
"Existing cluster" can mean either a saved :ref:`OnDemandCluster` config, which will be brought back up if needed,
or a BYO or OnDemandCluster that's already up. If you save the Cluster to the :ref:`Resource Name System (RNS)`,
you'll be able to dispatch to it from any environment. Multiple users or environments can send requests to a cluster
without issue, and either the OS or Ray (depending on the call to the cluster) will handle the resource contention.

You can load an existing cluster by name from local or Runhouse RNS simply by:

.. code-block:: python

    gpu = rh.cluster(name='~/my-local-a100')
    gpu = rh.cluster(name='@/my-a100-in-rh-rns')
    gpu = rh.cluster(name='^rh-v100')  # Loads a builtin cluster config

    # or, if you just want to load the Cluster object without refreshing its status
    gpu = rh.cluster(name='^rh-v100', dryrun=True)


Packages
--------
A :ref:`Package` represents the way we share code between various systems (ex: s3, cluster, local),
and back up the working directory to create a function that can be easily accessible and portable.
This allows Runhouse to load your code onto the cluster on the fly, as well as do basic registration and dispatch of
the :ref:`Function`.

At a high level, we dump the list of packages into gRPC, and the packages are installed on the gRPC server
on the cluster.

We currently provide four general package install methods: local, requiements.txt, pip, and conda.

GitPackage
~~~~~~~~~~

Runhouse offers support for using a GitHub URL as GitPackage object, a subclass of :ref:`Package`.
Instead of cloning down code from GitHub and copying it directly into your existing code base, you can provide a link
to a specific :code:`git_url` (with support for a :code:`revision` version), and Runhouse handles all the installations
for you.
