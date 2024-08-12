Docker: Dev and Prod Workflows
==============================

.. raw:: html

    <p><a href="https://colab.research.google.com/github/run-house/notebooks/blob/stable/docs/docker-workflows.ipynb">
    <img height="20px" width="117px" src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a></p>

This guide demonstrates how to use the same Docker image with your
Runhouse cluster, for both:

-  **Production**: running functions and code that is pre-installed on
   the Docker image
-  **Local development**: making local edits to your repo, and having
   local changes propagated over to the cluster for experimentation

Afterwards, we provide a script that shows how to easily set up and
toggle between these two settings, using the same cluster setup.

In this example, we are going to be using the `DJLServing 0.27.0 with
DeepSpeed
0.12.6 <https://github.com/aws/deep-learning-containers/blob/master/available_images.md#large-model-inference-containers>`__
Container, which includes HuggingFace Tranformers (4.39.0), Diffusers
(0.16.0), and Accelerate (0.28.0). We will use both the container
version of these packages, as well as local editable versions to
showcase both production ready and local experimentation use cases for
using the same Docker image.

Setup
-----

Runhouse uses SkyPilot under the hood to set up the Docker image on the
cluster. Because we are pulling the Docker image from AWS ECR, we first
set some environment variables necessary to pull the Docker image.

For more specific details on getting your Docker image set up with
Runhouse, please take a look at the `Docker Setup
Guide <https://www.run.house/docs/docker-setup>`__.

.. code:: ipython3

    ! export SKYPILOT_DOCKER_USERNAME=AWS
    ! export SKYPILOT_DOCKER_PASSWORD=$(aws ecr get-login-password --region us-west-1)
    ! export SKYPILOT_DOCKER_SERVER=763104351884.dkr.ecr.us-west-1.amazonaws.com

Once these variables are set, we can import runhouse and construct an
ondemand cluster, specifying the container image id as follows, and call
``cluster.up_if_not()`` to launch the cluster with the Docker image
loaded on it.

.. code:: ipython3

    import runhouse as rh


.. parsed-literal::
    :class: code-output

    INFO | 2024-08-01 02:18:48.921683 | Loaded Runhouse config from /Users/caroline/.rh/config.yaml


.. code:: ipython3

    cluster = rh.ondemand_cluster(
        name="diffusers_docker",
        image_id="docker:djl-inference:0.27.0-deepspeed0.12.6-cu121",
        instance_type="g5.8xlarge",
        provider="aws",
    )
    cluster.up_if_not()

The function we’ll be using in our demo is ``is_transformers_available``
from ``diffusers.utils``. We’ll first show what using this function
directly on the box (e.g. a production setting) looks like. After, we’ll
show the case if we had local versions of the repositories, that we’d
modified, and wanted to test out our changes on the cluster.

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

Defining the Env
~~~~~~~~~~~~~~~~

Here, we construct a Runhouse env containing anything you need for
running your code, that doesn’t already live on the cluster. For
instance, any environment variables or additional packages that you
might need installed. Do **NOT** include the packages already installed
on the container that you want pinned to the specific version, in this
case diffusers and transformers.

Then send and create the env on the cluster by directly calling
``env.to(cluster)``.

.. code:: ipython3

    prod_env = rh.env(name="prod_env", env_vars={"HF_TOKEN": "****"})
    prod_env.to(cluster)


.. parsed-literal::
    :class: code-output

    INFO | 2024-08-01 02:19:13.168591 | Port 32300 is already in use. Trying next port.
    INFO | 2024-08-01 02:19:13.172968 | Running forwarding command: ssh -T -L 32301:localhost:32300 -i ~/.ssh/sky-key -o Port=10022 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o IdentitiesOnly=yes -o ExitOnForwardFailure=yes -o ServerAliveInterval=5 -o ServerAliveCountMax=3 -o ConnectTimeout=30s -o ForwardAgent=yes -o ProxyCommand='ssh -T -L 32301:localhost:32300 -i ~/.ssh/sky-key -o Port=22 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o IdentitiesOnly=yes -o ExitOnForwardFailure=yes -o ServerAliveInterval=5 -o ServerAliveCountMax=3 -o ConnectTimeout=30s -o ForwardAgent=yes -W %h:%p ubuntu@3.142.171.243' root@localhost
    INFO | 2024-08-01 02:19:16.685047 | Calling prod_env._set_env_vars


.. parsed-literal::
    :class: code-output

    ----------------
    [36mdiffusers_docker[0m
    ----------------
    [36mprod_env env: Calling method _set_env_vars on module prod_env
    [0m

.. parsed-literal::
    :class: code-output

    INFO | 2024-08-01 02:19:17.273890 | Time to call prod_env._set_env_vars: 0.59 seconds
    INFO | 2024-08-01 02:19:17.350932 | Calling prod_env.install


.. parsed-literal::
    :class: code-output

    [36mprod_env env: Calling method install on module prod_env
    [0m

.. parsed-literal::
    :class: code-output

    INFO | 2024-08-01 02:19:17.929387 | Time to call prod_env.install: 0.58 seconds




.. parsed-literal::
    :class: code-output

    <runhouse.resources.envs.env.Env at 0x133a6eb60>



Defining the Function
~~~~~~~~~~~~~~~~~~~~~

The function is the ``is_transformers_available`` function imported
above. When creating the function to run remotely on the production
Runhouse env, we pass in the **name** of the Runhouse env. By passing in
the env name, rather than the object, it simply signals that we want to
use the env that already lives on the cluster, without re-syncing over
anything.

.. code:: ipython3

    prod_fn = rh.function(is_transformers_available).to(cluster, env=prod_env.name)


.. parsed-literal::
    :class: code-output

    INFO | 2024-08-01 02:19:22.140840 | Sending module is_transformers_available of type <class 'runhouse.resources.functions.function.Function'> to diffusers_docker


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

    INFO | 2024-08-01 02:19:27.817880 | Calling is_transformers_available.call


.. parsed-literal::
    :class: code-output

    [36mprod_env env: Calling method call on module is_transformers_available
    [0m

.. parsed-literal::
    :class: code-output

    INFO | 2024-08-01 02:19:31.554237 | Time to call is_transformers_available.call: 3.74 seconds




.. parsed-literal::
    :class: code-output

    True



Local Development
-----------------

Now for the local development and experimentation case. Let’s say we
have the HuggingFace diffusers and transformers repositories cloned and
installed as a local editable package, and are making changes to it that
we want reflected when we run it on the cluster.

Local Changes
~~~~~~~~~~~~~

Let’s continue using the ``is_transformers_available`` function, except
this time we’ll change the function to return the version number of the
transformers package if it exists, instead of True.

In my local diffusers/src/diffusers/utils/import_utils.py file:

::

   def is_transformers_available:
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

    '4.44.0.dev0'



Defining the Env
~~~~~~~~~~~~~~~~

In this case, because we want to use our local diffusers package, as
well as our local transformers package and version, we include these as
requirements inside our Runhouse env. There is no need to preemptively
send over the env, as now we can directly pass in the env object when we
define the function, to sync over the local changes.

.. code:: ipython3

    dev_env = rh.env(name="dev_env", env_vars={"HF_TOKEN": "****"}, reqs=["diffusers", "transformers"])

Defining the Function
~~~~~~~~~~~~~~~~~~~~~

Define a Runhouse function normally, passing in the function, and
sending it to the cluster. Here, we simply pass in the ``dev_env``
object into the env argument. This will ensure that the folder that this
function is locally found in, along with any requirements in the env
requirements is synced over to the cluster properly. Even though the
container already contains its own version of these packages,
requirements that can be found locally, such as our local modified
diffusers and transformers (v 4.44.0.dev0) repositories will be synced
to the cluster.

.. code:: ipython3

    dev_fn = rh.function(is_transformers_available).to(cluster, env=dev_env)


.. parsed-literal::
    :class: code-output

    INFO | 2024-08-01 02:34:20.997084 | Copying package from file:///Users/caroline/Documents/diffusers to: diffusers_docker
    INFO | 2024-08-01 02:34:24.924803 | Copying package from file:///Users/caroline/Documents/transformers to: diffusers_docker
    INFO | 2024-08-01 02:34:31.626250 | Calling dev_env._set_env_vars


.. parsed-literal::
    :class: code-output

    [36mdev_env env: Calling method _set_env_vars on module dev_env
    [0m

.. parsed-literal::
    :class: code-output

    INFO | 2024-08-01 02:34:32.324740 | Time to call dev_env._set_env_vars: 0.7 seconds
    INFO | 2024-08-01 02:34:32.444053 | Calling dev_env.install


.. parsed-literal::
    :class: code-output

    [36mdev_env env: Calling method install on module dev_env
    [0m[36mInstalling Package: diffusers with method pip.
    [0m[36mRunning via install_method pip: python3 -m pip install /root/diffusers
    [0m[36mInstalling Package: transformers with method pip.
    [0m[36mRunning via install_method pip: python3 -m pip install /root/transformers
    [0m

.. parsed-literal::
    :class: code-output

    INFO | 2024-08-01 02:34:56.084695 | Time to call dev_env.install: 23.64 seconds
    INFO | 2024-08-01 02:34:56.239915 | Sending module is_transformers_available of type <class 'runhouse.resources.functions.function.Function'> to diffusers_docker


Calling the Function
~~~~~~~~~~~~~~~~~~~~

Now, we call the function

.. code:: ipython3

    dev_fn()


.. parsed-literal::
    :class: code-output

    INFO | 2024-08-01 02:35:01.303550 | Calling is_transformers_available.call


.. parsed-literal::
    :class: code-output

    [36mdev_env env: Calling method call on module is_transformers_available
    [0m

.. parsed-literal::
    :class: code-output

    INFO | 2024-08-01 02:35:02.946712 | Time to call is_transformers_available.call: 1.64 seconds




.. parsed-literal::
    :class: code-output

    '4.44.0.dev0'



Summary - Setting Up Your Code
------------------------------

Here, we implement the above as a script that can be used to toggle
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
           env = rh.env(name="prod_env_name", env_vars={...}, ...)
           env.to(cluster)
           remote_fn = rh.function(is_transformers_available).to(cluster, env=env.name)
       else:
           env = rh.env(name="dev_env_name", reqs=["diffusers", "trasnformers"], ...)
           remote_fn = rh.function(is_transformers_available).to(cluster, env=env)

       remote_fn()

To summarize the core differences between local experimentation and
production workflow:

**Local Development**: Include local packages to sync in the ``reqs``
field of the ``env`` that the function is associated with.

**Production Workflow**: Do not include production packages that are
part of the Docker image in the ``reqs`` field of the ``env``. Send the
``env`` to the cluster prior to defining the function, and then pass in
the env name rather than the env object for the function. Also, include
Runhouse and Ray on the image to pin those for production as well.
