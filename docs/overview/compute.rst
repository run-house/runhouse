Compute
====================================

The :ref:`Function`, :ref:`Cluster`, and :ref:`Package` APIs allow a seamless flow of code and execution across local and remote compute.
They blur the line between program execution and deployment, providing both a path of least resistence for running a
sub-routine on specific hardware, while unceremoniously turning that sub-routine into a reusable service.

They also provide convenient dependency isolation and management, provider-agnostic provisioning and termination,
and rich debugging and accessibility interfaces built-in.

Functions
---------

Runhouse allows you to send function code to a cluster, but still interact with it as a native runnable :ref:`Function` object
(see `tutorial 01 <https://github.com/run-house/tutorials/tree/main/t01_Stable_Diffusion/>`_).
When you do this, the following steps occur:

1. We check if the cluster is up, and bring up the cluster if not (only possible for :ref:`OnDemandClusters <OnDemandCluster>`)
2. We check that the cluster's gRPC server has started to handle requests to do things like install packages, run modules, get previously executed results, etc. If it hasn't, we install Runhouse on the cluster and start the gRPC server. The gRPC server initializes Ray.
3. We collect the dependencies from the :code:`reqs` parameter and install them on the cluster via :code:`cluster.install_packages()`. By default, we'll sync over the working git repo and install its :code:`requirements.txt` if it has one.


When you run your function module, we function a gRPC request to the cluster with the module name and function entrypoint to run.
The gRPC server adds the module to its python path, imports the module, grabs the function entrypoint, runs it,
and returns your results.

You can stream in logs from the cluster as your module runs by passing :code:`stream_logs=True` into your call line:


.. code-block:: python

    images = generate_gpu('A dog.', num_images=1, steps=50, stream_logs=True)


We plan to support additional form factors for modules beyond "remote Python function" shortly, including HTTP endpoints, custom ASGIs, and more.


Advanced Function Usage
~~~~~~~~~~~~~~~~~~~~~~~
There are a number of ways to call a Function beyond just :code:`__call__`.

:code:`.remote` will call the function async (using Ray) and return a reference (`Ray ObjectRef <https://docs.ray.io/en/latest/ray-core/objects.html>`_)
to the object on the cluster. You can pass the ref into another function and it will be automatically
dereferenced once on the cluster. This is a convenient way to avoid passing large objects back and forth to your
laptop, or to run longer execution in notebooks without locking up the kernel.

.. code-block:: python

    images_ref = generate_gpu.remote('A dog.', num_images=1, steps=50)
    images = rh.get(images_ref)
    # or
    my_other_function(images_ref)


:code:`.enqueue` will queue up your function call on the cluster to make sure it doesn't run simultaneously with other
calls, but will wait until the execution completes.

:code:`.map` and :code:`.starmap` are easy way to parallelize your function (again using Ray on the cluster).

.. code-block:: python

    generate_gpu.map(['A dog.', 'A cat.', 'A biscuit.'], num_images=[1]*3, steps=[50]*3)


will run the function on each of the three prompts, and return a list of the results.
Note that the :code:`num_images` and :code:`steps` arguments are broadcasted to each prompt, so the first prompt will get 1 image.


.. code-block:: python

    generate_gpu.starmap([('A dog.', 1), ('A cat.', 2), ('A biscuit.', 3)], steps=50)

is the same as :code:`map` as above, but we can pass the arguments as a list of tuples, and the steps argument as a
single value, since it's the same for all three prompts.


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
SkyPilot context (the ones that show when you run :code:`sky status --refresh`). However, the best way to confirm that you
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
you'll be able to dispatch to it from any environment. Multiple users or environments can send requests to a cluster
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

This is useful for debugging, or for running a script that you don't want to send to the cluster
(e.g. because it has too many dependencies).

If you want to run an application on the cluster that requires a port to be open,
e.g. `Tensorboard <https://www.tensorflow.org/tensorboard/>`_, `Gradio <https://gradio.app/>`_.

.. code-block:: python

    gpu.ssh_tunnel(local_port=7860, remote_port=7860)


Packages
--------
A :ref:`Package` represents the way we share code between various systems (ex: s3, cluster, local),
and back up the working directory to create a function that can be easily accessible and portable.
This allows Runhouse to load your code onto the cluster on the fly, as well as do basic registration and dispatch of
the :ref:`Function`.

At a high level, we dump the list of packages into gRPC, and the packages are installed on the gRPC server
on the cluster.

We currently provide four package install methods:

- :code:`local`: Install packages to a Folder or a given path to a directory
- :code:`reqs`: Install a :code:`requirements.txt` file from the working directory.
- :code:`pip`: Runs :code:`pip install` for the provided packages.
- :code:`conda`: Runs :code:`conda install` for the provided packages.


GitPackage
~~~~~~~~~~

Runhouse offers support for using a GitHub URL as GitPackage object, a subclass of :ref:`Package`.
Instead of cloning down code from GitHub and copying it directly into your existing code base, you can provide a link
to a specific :code:`git_url` (with support for a :code:`revision` version), and Runhouse handles all the installations
for you!

For example:

.. code-block:: python

    rh.GitPackage(git_url='https://github.com/huggingface/diffusers.git',
                  install_method='pip',
                  revision='v0.11.1')


See a more detailed example of working with a GitPackage in our `Dreambooth Tutorial <https://github.com/run-house/tutorials/blob/main/t02_Dreambooth/p01_dreambooth_train.py/>`_
