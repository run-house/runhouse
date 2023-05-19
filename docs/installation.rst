Installation and Setup Guide
============================

Installation
~~~~~~~~~~~~

Runhouse can be installed with:

.. code-block:: console

    $ pip install runhouse

Depending on which cloud providers you plan to use, you can also install the following
additional dependencies (to install the right versions of tools like boto, gsutil, etc.):

.. code-block:: console

    $ pip install "runhouse[aws]"
    $ pip install "runhouse[gcp]"
    $ pip install "runhouse[azure]"
    # Or
    $ pip install "runhouse[all]"

Cluster Setup
~~~~~~~~~~~~~
Runhouse is not managed compute; everything runs inside your own compute and storage, using your credentials.

If you are using an existing cluster (BYO-cluster), no additional setup is needed -- just have your cluster IP
address and path to SSH credentials ready.

For clusters through cloud accounts (AWS, Azure, GCP, LambdaLabs), Runhouse supports autoscaled, on-demand clusters,
where we spin up and down cloud instances (in your own cloud account) for you.

We use `SkyPilot <https://skypilot.readthedocs.io/en/latest/>`_ for much of the heavy lifting
with launching and terminating cloud instances, and will use their APIs to check for proper setup.
To check which cloud providers are setup, as well as detailed instructions for setting up other
cloud providers, run the following on CLI, or check out SkyPilot's
`cloud account setup <https://skypilot.readthedocs.io/en/latest/getting-started/installation.html#cloud-account-setup>`_
for instructions.

.. code-block:: console

    $ sky check

SkyPilot also provides an excellent suite of CLI commands for basic instance management operations.
There are a few that you'll be reaching for frequently when using Runhouse with autoscaling that you
should familiarize yourself with, :ref:`here <Cluster>`.

Secrets and Portability
~~~~~~~~~~~~~~~~~~~~~~~

Using Runhouse with only the OSS Python package is perfectly fine.
However, you can unlock some unique portability features by creating an (always free) `account <https://api.run.house/>`_
and saving your secrets and/or resource metadata there.

For example, you can open a Google Colab, call :code:`runhouse login`, and all of your secrets or resources
will be ready to use there with no additional setup. Think of the OSS-package-only experience as
akin to Microsoft Office, while creating an account will make your cloud resources sharable and
accessible from anywhere like Google Docs. You can see examples of this portability
in the `Runhouse Tutorials <https://github.com/run-house/tutorials/>`_.

To create an account, visit our `dashboard <https://api.run.house/>`_, or simply call
:code:`runhouse login` from the command line (or :code:`rh.login()` from Python).

.. note::
    These portability features only ever store light metadata about your resources
    (e.g. folder name, cloud provider, storage bucket, path) on our API servers.
    *All the actual data and compute stays inside your own cloud account and never hits our servers*.

    Secrets are stored in `Hashicorp Vault <https://www.vaultproject.io/>`_ (an industry standard for secrets management),
    never on our API servers, and our APIs simply call into Vault's APIs.
    We plan to add support for BYO secrets management shortly. Let us know if you need it and which system you use.
