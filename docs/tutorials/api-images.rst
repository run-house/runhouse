Images
======

.. raw:: html

    <p><a href="https://colab.research.google.com/github/run-house/notebooks/blob/stable/docs/api-images.ipynb">
    <img height="20px" width="117px" src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a></p>

Runhouse clusters expose various functions that allow you to set up
state, dependencies, and whatnot on all nodes of your cluster. These
include:

-  ``cluster.install_packages(...)``
-  ``cluster.rsync(...)``
-  ``cluster.set_env_vars(...)``
-  ``cluster.run_bash(...)``

A Runhouse “Image” is simply an abstraction that allows you to run
several setup steps *before* we install ``runhouse`` and bring up the
Runhouse daemon and initial set up on your cluster’s nodes. You can also
specify a Docker ``image_id`` as the “base image” of your Runhouse
image.

Here’s a simple example of using the Runhouse Image abstraction in your
cluster setup:

.. code:: ipython3

    import runhouse as rh

    image = (
        rh.Image(name="sample_image")
        .from_docker("python:3.12.8-bookworm")
        .install_packages(["numpy", "pandas"])
        .sync_secrets(["huggingface"])
        .set_env_vars({"RH_LOG_LEVEL": "debug"})
    )

    cluster = rh.cluster(name="ml_ready_cluster", image=image, instance_type="CPU:2+", provider="aws").up_if_not()


.. parsed-literal::
    :class: code-output

    I 12-17 12:04:55 provisioner.py:560] [32mSuccessfully provisioned cluster: ml_ready_cluster[0m
    I 12-17 12:04:57 cloud_vm_ray_backend.py:3402] Run commands not specified or empty.
    Clusters
    [2mAWS: Fetching availability zones mapping...[0mNAME              LAUNCHED        RESOURCES                                                                  STATUS  AUTOSTOP  COMMAND
    ml_ready_cluster  a few secs ago  1x AWS(m6i.large, image_id={'us-east-1': 'docker:python:3.12.8-bookwor...  UP      (down)    /Users/rohinbhasin/minico...

    [?25h

The growing listing of setup steps available for Runhouse images is
available in the API reference.
