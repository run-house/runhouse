Docker: Dev and Prod Workflows with Runhouse
============================================

.. raw:: html

    <p><a href="https://colab.research.google.com/github/run-house/notebooks/blob/stable/docs/docker-workflows.ipynb">
    <img height="20px" width="117px" src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a></p>

This guide demonstrates how to use the same Docker image with your
Runhouse cluster, for both:

-  **Production**: running functions and code that is pre-installed on
   the Docker image
-  **Local development**: making local edits to your repo, and
   propagating over those local changes to the cluster for
   experimentation

Afterwards, we provide a script that shows how to easily set up and
toggle between these two settings, using the same cluster setup.

In this example, we are going to be using the `DJLServing 0.27.0 with
DeepSpeed
0.12.6 <https://github.com/aws/deep-learning-containers/blob/master/available_images.md#large-model-inference-containers>`__
Container, which includes HuggingFace Tranformers (4.39.0), Diffusers
(0.16.0), and Accelerate (0.28.0). We will use the container version of
these packages to demonstrate the pre-packaged production workflow, as
well as local editable versions to showcase the local experimentation
use cases.

Docker Cluster Setup
--------------------

Because we are pulling the Docker image from AWS ECR, we need to provide
the corresponding credentials in order to properly pull and setup the
image on the cluster. This can be done through a Runhouse Docker secret,
or by setting environment variables. Please refer to <Guide: Docker
Cluster Setup> for more details.

.. code:: ipython3

    import subprocess
    import runhouse as rh

    docker_ecr_creds = {
        "username": "AWS",
        "password": subprocess.run("aws ecr get-login-password --region us-west-1", shell=True, capture_output=True).stdout.strip().decode("utf-8"),
        "server": "763104351884.dkr.ecr.us-west-1.amazonaws.com",
    }
    docker_secret = rh.provider_secret("docker", values=docker_ecr_creds)

Next, construct a Runhouse image, passing in the docker image ID and
secret. Feed this image into the OnDemand cluster factory, and up the
cluster.

.. code:: ipython3

    base_image = rh.Image("docker_image").from_docker(
        "djl-inference:0.27.0-deepspeed0.12.6-cu121", docker_secret=docker_secret
    )

    cluster = rh.ondemand_cluster(
        name="diffusers_docker",
        image=base_image,
        instance_type="g5.8xlarge",
        provider="aws",
    )
    cluster.up_if_not()



.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #008000; text-decoration-color: #008000">⠴</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">Preparing SkyPilot runtime (3/3 - runtime)</span>  [2mView logs at: ~/sky_logs/sky-2024-12-23-13-56-48-619803/provision.log[0m
    </pre>



.. parsed-literal::
    :class: code-output

    [0m[32m✓ Cluster launched: diffusers_docker.[0m  [2mView logs at: ~/sky_logs/sky-2024-12-23-13-56-48-619803/provision.log[0m


.. parsed-literal::
    :class: code-output

    INFO | 2024-12-23 14:03:39 | runhouse.resources.hardware.launcher_utils:391 | Starting Runhouse server on cluster
    INFO | 2024-12-23 14:03:39 | runhouse.resources.hardware.cluster:1247 | Restarting Runhouse API server on diffusers_docker.

    INFO:     Started server process [2929]
    INFO:     Waiting for application startup.
    INFO:     Application startup complete.
    INFO:     Uvicorn running on http://0.0.0.0:32300 (Press CTRL+C to quit)





.. parsed-literal::
    :class: code-output

    <runhouse.resources.hardware.on_demand_cluster.OnDemandCluster at 0x127ea7730>



Sample Function
---------------

The function we’ll be using in our demo is ``is_transformers_available``
from ``diffusers.utils``. We’ll first show how to use the base version
of this function, which was installed on the box through the cluster
setup (e.g. a production setting). Then, we’ll show how to propogate up
local changes and run them on the cluster, if your local version differs
from the one in the Docker container (e.g. different package version, or
locally edited).

.. code:: ipython3

    from diffusers.utils import is_transformers_available

Production Workflow
-------------------

The core of the production workflow is that the Docker image already
contains the exact packages and versions we want, probably published
into the registry in CI/CD. We don’t want to perform any installs or
code changes within the image throughout execution so we can preserve
exact reproducibility.

**NOTE**: By default, Ray and Runhouse are installed on the ondemand
cluster during setup time (generally attempting to match the versions
you have locally), unless we detect that they’re already present. To
make sure that no installs occur in production, please make sure that
you have Runhouse and Ray installed in your docker image.

Defining the Function
~~~~~~~~~~~~~~~~~~~~~

The function is the ``is_transformers_available`` function imported
above. When creating the function to run remotely on the production
Runhouse env, we pass in the flag ``sync_local=False`` to indicate that
we want to use the function on the cluster, without re-syncing over
anything.

.. code:: ipython3

    prod_fn = rh.function(is_transformers_available).to(cluster, sync_local=False)


.. parsed-literal::
    :class: code-output

    INFO | 2024-12-23 14:04:57 | runhouse.resources.hardware.ssh_tunnel:91 | Running forwarding command: ssh -T -L 32300:localhost:32300 -i ~/.ssh/sky-key -o Port=10022 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o IdentitiesOnly=yes -o ExitOnForwardFailure=yes -o ServerAliveInterval=5 -o ServerAliveCountMax=3 -o ConnectTimeout=30s -o ForwardAgent=yes -o ProxyCommand='ssh -i ~/.ssh/sky-key -o Port=22 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o IdentitiesOnly=yes -o ExitOnForwardFailure=yes -o ServerAliveInterval=5 -o ServerAliveCountMax=3 -o ConnectTimeout=30s -o ForwardAgent=yes -W %h:%p ubuntu@52.24.239.151' root@localhost
    INFO | 2024-12-23 14:05:00 | runhouse.resources.module:511 | Sending module is_transformers_available of type <class 'runhouse.resources.functions.function.Function'> to diffusers_docker


Calling the Function
~~~~~~~~~~~~~~~~~~~~

Now, simply call the function, and it will detect the corresponding
function on the cluster to run. In this case, it returns whether or not
transformers is available on the cluster, which it is, as it was part of
the Docker image.

.. code:: ipython3

    prod_fn()


.. parsed-literal::
    :class: code-output

    INFO | 2024-12-23 14:05:01 | runhouse.servers.http.http_client:439 | Calling is_transformers_available.call
    INFO | 2024-12-23 14:05:06 | runhouse.servers.http.http_client:504 | Time to call is_transformers_available.call: 4.86 seconds




.. parsed-literal::
    :class: code-output

    True



For even more specifics on any setup for running your function, you can
also directly use cluster functionality (e.g. setting additional env
vars, installing packages/running commands), or construct isolated
processes (see Process API guide) with specific compute to run the
function on.

Local Development
-----------------

Now for the local development and experimentation case. Let’s say we
have the HuggingFace ``diffusers`` repository cloned and installed as a
local editable package, and are making changes to it that we want
reflected when we run it on the cluster. We also have a different
version of the transformers package installed.

Local Changes
~~~~~~~~~~~~~

Let’s continue using the ``is_transformers_available`` function, except
this time we’ll change the function to return the version number of the
transformers package if it exists, instead of ``True``. This shows that
we have ``transformers==4.44.2`` installed locally.

In my local diffusers/src/diffusers/utils/import_utils.py file:

::

   def is_transformers_available():
       try:
           import transformers
           return transformers.__version__
       except ImportError:
           return False

.. code:: ipython3

    from diffusers.utils import is_transformers_available

    is_transformers_available()




.. parsed-literal::
    :class: code-output

    '4.44.2'



Installing local version
~~~~~~~~~~~~~~~~~~~~~~~~

When Runhoue installs packages on the remote cluster, it will check if
you have a version of the package locally, as well as whether a version
of the package already exists on this cluster. If it already exists
remotely, by default the remote package will not be overriden, but you
can force the local version by passing in the paramteter
``force_sync_local==True`` to ``cluster.install_packages``.

.. code:: ipython3

    cluster.install_packages(["transformers", "diffusers"], force_sync_local=True)

Defining the Function
~~~~~~~~~~~~~~~~~~~~~

Now construct a Runhouse function normally and send it to the cluster.
Here, we can leave out the ``sync_local`` flag, which defaults to True -
the local function will be synced onto the cluster.

.. code:: ipython3

    dev_fn = rh.function(is_transformers_available).to(cluster)


.. parsed-literal::
    :class: code-output

    INFO | 2024-12-23 14:11:05 | runhouse.resources.module:511 | Sending module is_transformers_available of type <class 'runhouse.resources.functions.function.Function'> to diffusers_docker


Calling the Function
~~~~~~~~~~~~~~~~~~~~

Now, when we call the function, it returns the version of the
transformers library installed, rather than a True/False. It also
correctly returns the same version as the locally installed version,
showing that both local diffusers and transformers packages were
properly synced and installed on the cluster.

.. code:: ipython3

    dev_fn()


.. parsed-literal::
    :class: code-output

    INFO | 2024-12-23 14:11:19 | runhouse.servers.http.http_client:439 | Calling is_transformers_available.call
    INFO | 2024-12-23 14:11:21 | runhouse.servers.http.http_client:504 | Time to call is_transformers_available.call: 2.48 seconds




.. parsed-literal::
    :class: code-output

    '4.44.2'



Summary - Setting Up Your Code
------------------------------

Here, we implement the above as a script to demonstrate the difference
between dev and prod. The script can easily be adapted and shared
between teammates developing and working with the same repos, with a
flag or variable flip to differentiate between experimentation and
production branches.

::

   from diffusers.utils import is_transformers_available

   if __name__ == "__main__":
       cluster = rh.ondemand_cluster(...)
       cluster.up_if_not()

       if prod:
           remote_fn = rh.function(is_transformers_available).to(cluster, sync_local=False)
       else:
           cluster.install_packages(["transformers", "diffusers"], )
           remote_fn = rh.function(is_transformers_available).to(cluster)

       remote_fn()
       cluster.teardown()
