Docker: Cluster Setup
=====================

.. raw:: html

    <p><a href="https://colab.research.google.com/github/run-house/notebooks/blob/stable/docs/docker-setup.ipynb">
    <img height="20px" width="117px" src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a></p>


Runhouse integrates with
`SkyPilot <https://skypilot.readthedocs.io/en/latest/docs/index.html>`__
to enable automatic setup of an existing Docker container when you
launch your `on-demand
cluster <https://www.run.house/docs/api/python/cluster#ondemandcluster-class>`__.
When you specify a Docker image for an on-demand cluster, the container
is automatically built and set up remotely on the cluster. The Runhouse
server will start directly inside the remote container.

**NOTE:** This guide details the setup and usage for on-demand clusters
only. It is not yet supported for static clusters.

Cluster & Docker Setup
----------------------

Public Docker Image
~~~~~~~~~~~~~~~~~~~

One can specify a Docker Image through the Runhouse Image class, which is
passed into the cluster factory. Call ``.from_docker(image_id)`` on the image,
passing in the Docker container in the format ``<registry>/<image>:<tag>``.


.. code:: ipython3

    base_image = rh.Image("base_image").from_docker("nvcr.io/nvidia/pytorch:23.10-py3")

    docker_cluster = rh.ondemand_cluster(
        name="pytorch_cluster",
        image=base_image,
        instance_type="CPU:2+",
        provider="aws",
    )

Private Docker Image
~~~~~~~~~~~~~~~~~~~~

To use a Docker image hosted on a private registry, such as ECR, you
need to additionally provide the ``user``, ``password``, and ``registry server``
values, as used in ``docker login -u <user> -p <password> <registry server>``.

These values are propagated to SkyPilot at launch time, which will be used
for setting up the base container on the cluster.

There are two approaches to providing this information:

1. Creating a runhouse Secret as follows, and pass it to the Image along with the
   Docker image above.

.. code:: ipython3

    values = {
        "username": <user>,
        "password": <password>,
        "server": <server>,
    }
    docker_secret = rh.provider_secret("docker", values=values)

.. code::

    base_image = rh.Image("base_image").from_docker(
        "pytorch-training:2.2.0-cpu-py310-ubuntu20.04-ec2", docker_secret=docker_secret
    )

2. Directly set your local environment variables, as expected by and extracted by
   Skypilot during launch time. In this case, you do not need to specify the secret
   during the runhouse Image construction.

   * ``SKYPILOT_DOCKER_USERNAME``: ``<user>``

   * ``SKYPILOT_DOCKER_PASSWORD``: ``<password>``

   * ``SKYPILOT_DOCKER_SERVER``: ``<registry server>``

For instance, to use the PyTorch2.2 ECR Framework provided
`here <https://github.com/aws/deep-learning-containers/blob/master/available_images.md#ec2-framework-containers-tested-on-ec2-ecs-and-eks-only>`__,
you can set your environment variables as so:

.. code::

   $ export SKYPILOT_DOCKER_USERNAME=AWS
   $ export SKYPILOT_DOCKER_PASSWORD=$(aws ecr get-login-password --region us-east-1)
   $ export SKYPILOT_DOCKER_SERVER=763104351884.dkr.ecr.us-east-1.amazonaws.com

.. code::

   base_image = rh.Image("base_image").from_docker("pytorch-training:2.2.0-cpu-py310-ubuntu20.04-ec2")

In either case, we can then construct the cluster with the image as follows:

.. code::

    ecr_cluster = rh.ondemand_cluster(
        name="ecr_pytorch_cluster",
        image_id=base_image,
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

SSH
~~~

To SSH directly onto the container, where the Runhouse server is
started, you can use ``runhouse cluster ssh <cluster_name>``.

User and Container Name
~~~~~~~~~~~~~~~~~~~~~~~

By default, the remote Docker container, which is set up through
Skypilot, will be named ``sky_container``, and the user will be
``root``.
