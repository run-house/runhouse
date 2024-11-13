Den Quick Start
===============

.. raw:: html

    <p><a href="https://colab.research.google.com/github/run-house/notebooks/blob/stable/docs/quick-start-den.ipynb">
    <img height="20px" width="117px" src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a></p>

With `Runhouse Den <https://www.run.house/dashboard>`__, you can manage
all of your compute and make it available through ephemeral clusters for
both research and production workflows. Whether you have a ML platforms
team or just yourself, Runhouse is the easiest way to ensure your
compute is efficiently utilized and oversubscribed without slowing down
development velocity.

- Centrally manage multiple cloud accounts and Kubernetes clusters in
  the same place.
- Both developers and the platforms team can monitor resource usage in
  detail and access persisted logs in the web UI.
- Save and reload services and resources (clusters, functions, classes,
  etc) - for instance, share an embeddings endpoint to be used by
  multiple teams with perfect reproducibility.
- Set limits and quotas for teams and individuals, and queue requests
  for compute.

A few of the ways Runhouse Den has helped teams include:

- Moving research teams from locally or ad hoc hosted notebooks, towards
  standardized, reproducible compute and training workflows that are
  ready to be executed in production.
- Improving Platform team observability over day-to-day activities as
  the ML team grew, and getting much better disambiguation between waste
  (clusters up too long, clusters underutilizing their resources) and
  genuinely valuable workflows that just happen to be long-lived.
- Reducing research-to-production time by 80% since resources launched
  and saved to Den can be scheduled nearly as-is by orchestrators.
- Improving key / credentials management
- Adopting a second cloud due to better quotas and cloud credits (while
  being mostly opaque to the end user)

If you haven’t already, check-out
`Quickstart <https://www.run.house/docs/tutorials/quick-start-cloud>`__
or the `Runhouse usage
guide <https://www.run.house/docs/main/en/how-to-use-runhouse>`__ to see
how you can get started with the Runhouse client as well.

Account Creation & Login
------------------------

You can create an account on the `run.house <https://www.run.house>`__
website or by calling the login command in Python or CLI.

To login, call ``runhouse login --sync-secrets`` in CLI. This will ask
you a series of questions on whether to sync local secrets to Runhouse -
e.g. your AWS / GCP / Azure secrets; once synced, you can launch compute
via Runhouse anywhere you are authenticated with Runhouse.

For more information on Secrets management, refer to the `Secrets
Tutorial <https://www.run.house/docs/tutorials/api-secrets>`__. Secrets
are always securely stored in `Vault <https://www.vaultproject.io/>`__.
If you have any questions about Runhouse’s information security
policies, please reach out at `hello@run.house <hello@run.house>`__.

Launching and Saving Clusters
-----------------------------

Let’s start by constructing some Runhouse resources that we’d like to
save.

.. code:: ipython3

    import runhouse as rh
    cluster = rh.ondemand_cluster(
        name="rh-cluster",
        instance_type="CPU:2+",
        provider="aws",
        launcher_type="den"
    ).up_if_not()

.. code:: ipython3

    def get_platform(a = 0):
            import platform
            return platform.platform()

    remote_get_platform = rh.function(get_platform).to(cluster)


.. parsed-literal::
    :class: code-output

    INFO | 2024-05-16 03:51:58.483032 | Because this function is defined in a notebook, writing it out to /Users/donny/code/notebooks/docs/get_platform_fn.py to make it importable. Please make sure the function does not rely on any local variables, including imports (which should be moved inside the function body). This restriction does not apply to functions defined in normal Python files.
    INFO | 2024-05-16 03:51:58.493093 | Port 32300 is already in use. Trying next port.
    INFO | 2024-05-16 03:51:58.494347 | Forwarding port 32301 to port 32300 on localhost.
    INFO | 2024-05-16 03:51:59.587613 | Server rh-cluster is up.
    INFO | 2024-05-16 03:51:59.595752 | Copying package from file:///Users/donny/code/notebooks to: rh-cluster
    INFO | 2024-05-16 03:52:00.716693 | Calling _cluster_default_env.install
    INFO | 2024-05-16 03:52:01.235732 | Time to call _cluster_default_env.install: 0.52 seconds
    INFO | 2024-05-16 03:52:01.252665 | Sending module get_platform of type <class 'runhouse.resources.functions.function.Function'> to rh-cluster


You can save the resources we created above to your Den account with the
``save`` method:

.. code:: ipython3

    cluster.save()
    remote_get_platform.save()

Reloading
---------

Once saved, resources can be reloaded from any environment in which you
are logged into. For instance, if you are running this in a Colab
notebook, you can jump into your terminal, call ``runhouse login``, and
then reconstruct and run the function on the cluster with the following
Python script:

.. code:: ipython3

   import runhouse as rh

   if __name__ == "__main__":
       reloaded_fn = rh.function(name="get_platform")
       print(reloaded_fn())

The ``name`` used to reload the function is the method name by default.
You can customize a function name using the following syntax:

.. code:: ipython3

   remote_get_platform = rh.function(fn=get_platform, name="my_function").to(cluster)

Sharing
-------

You can also share your resource with collaborators, and choose which
level of access to give. Once shared, they will be able to see the
resource in their dashboard as well, and be able to load and use the
shared resource. They’ll need to load the resource using its full name,
which includes your username (``/your_username/get_platform``).

.. code:: ipython3

    remote_get_platform.share(
        users=["teammate1@email.com"],
        access_level="write",
    )

Web UI
------

After saving your resources, you can log in and see them on your `Den
dashboard <https://www.run.house/dashboard>`__, labeled as
``/<username>/rh-cluster`` and ``/<username>/get_platform``.

Clicking into the resource provides information about your resource. You
can view the resource metadata, previous versions, and activity, or add
a description to the resource.

Dive Deeper
-----------

Check on more in-depth tutorials on:

- Resource Management https://www.run.house/docs/tutorials/api-resources
- Secrets Management https://www.run.house/docs/tutorials/api-secrets
