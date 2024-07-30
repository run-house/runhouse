Docker Runtime Env
==================

Runhouse integrates with
`SkyPilot <https://skypilot.readthedocs.io/en/latest/docs/index.html>`__
to enable automatic setup of an existing Docker container when you
launch your `on-demand
cluster <https://www.run.house/docs/api/python/cluster#ondemandcluster-class>`__.
When you specify a Docker image for an on-demand cluster, the container
is automatically built and set up remotely on the cluster. The Runhouse
server will start directly inside the remote container.

**NOTE:** This guide details the setup and usage for on-demand clusters
only. Docker container is also supported for Sagemaker clusters, and it
is not yet supported for static clusters.

Cluster & Docker Setup
----------------------

**NOTE:** Docker support for on-demand clusters is currently an Alpha
feature. The syntax and setup is subject to change, and this page will
be updated with any changes.

Public Docker Image
~~~~~~~~~~~~~~~~~~~

To specify the Docker container, pass it to the ``image_id`` field of
the ondemand_cluster factory, in the following format:
``docker:<registry>/<image>:<tag>``.

.. code:: ipython3

    docker_cluster = rh.ondemand_cluster(
        name="pytorch_cluster",
        image_id="docker:nvcr.io/nvidia/pytorch:23.10-py3",
        instance_type="CPU:2+",
        provider="aws",
    )

Private Docker Image
~~~~~~~~~~~~~~~~~~~~

To use a Docker image hosted on a private registry, such as ECR, you
need to pass in the following environment variables. These environment
variables will propagate through to SkyPilot, which will use them while
launching and setting up the cluster and base container.

Values used in:
``docker login -u <user> -p <password> <registry server>``:

* SKYPILOT_DOCKER_USERNAME: ``<user>``

* SKYPILOT_DOCKER_PASSWORD: ``<password>``

* SKYPILOT_DOCKER_SERVER: ``<registry server>``

For instance, to use the PyTorch2.2 ECR Framework provided
`here <https://github.com/aws/deep-learning-containers/blob/master/available_images.md#ec2-framework-containers-tested-on-ec2-ecs-and-eks-only>`__,
you must first set your environment variables somewhere your program can
access.

.. code:: cli

   export SKYPILOT_DOCKER_USERNAME=AWS
   export SKYPILOT_DOCKER_PASSWORD=$(aws ecr get-login-password --region us-east-1)
   export SKYPILOT_DOCKER_SERVER=763104351884.dkr.ecr.us-east-1.amazonaws.com

Then, instantiate the on-demand cluster and fill in the ``image_id``
field.

.. code:: ipython3

    ecr_cluster = rh.ondemand_cluster(
        name="ecr_pytorch_cluster",
        image_id="docker:pytorch-training:2.2.0-cpu-py310-ubuntu20.04-ec2",
        instance_type="CPU:2+",
        provider="aws",
    )

Launching the Cluster
^^^^^^^^^^^^^^^^^^^^^

You can then launch the docker cluster with ``ecr_cluster.up()``. If for
any reason the docker pull fails on the cluster (for instance, due to
incorrect credentials or permission error), you must first teardown the
cluster with ``ecr_cluster.teardown()`` or
``sky stop ecr_pytorch_cluster`` in CLI before re-launching the cluster
with new credentials in order for them to propagate through.

Advanced Usage and Details
--------------------------

When the cluster is launched, the Docker container is set up on the
cluster. SkyPilot will be set up in it’s own separate base Conda
environment, while Runhouse will be installed and the server set up
directly on the Docker container. This way, any commands or
functions/classes run through Runhouse will be run directly on the
container, with access to any of its dependencies and setup.

SSH
~~~

To SSH directly onto the container, where the Runhouse server is
started, you can use ``runhouse ssh <cluster_name>``.

If you simply use ``ssh <cluster_name>``, the base environment set up by
SkyPilot will be activated by default. In this case, you would need to
additionally call ``conda deactivate`` to land in the base Docker
container.

User and Container Name
~~~~~~~~~~~~~~~~~~~~~~~

By default, the remote Docker container, which is set up through
Skypilot, will be named ``sky_container``, and the user will be
``root``.
