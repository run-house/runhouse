Den Quick Start
===============

`Runhouse Den <https://www.run.house/dashboard>`__ lets you manage and
track your infra, services, and resources (clusters, functions, secrets,
etc). These resources can be easily reloaded from any environment, are
ready to be used without additional setup, and can even be shared with
another user or teammate. Then, in the Den Web UI, you can access,
visualize, and manage your resources along with version history.

Installing Runhouse
-------------------

To use Runhouse to launch on-demand clusters, run the following
installation command. This includes
`SkyPilot <https://github.com/skypilot-org/skypilot>`__, which is used
for launching fresh VMs through various cloud providers.

.. code:: ipython3

    !pip install "runhouse[sky]"

Account Creation & Login
------------------------

You can create an account on the `run.house <https://www.run.house>`__
website or by calling the login command in Python or CLI.

To login on your dev environment, call ``rh.login()`` in Python or
``runhouse login`` in CLI.

.. code:: ipython3

    import runhouse as rh

.. code:: ipython3

    rh.login()

As you’ll see in the login prompts, Runhouse also optionally offers
secrets management, where it can automatically detect local AI provider
secrets (e.g. clouds, Hugging Face, OpenAI, etc.), and gives you the
option to upload them securely into your account to use on remote
clusters or in other environments. For more information on Secrets
management, refer to the `Secrets
Tutorial <https://www.run.house/docs/tutorials/api-secrets>`__.

Saving
------

Let’s start by constructing some runhouse resources that we’d like to
save down. These resources were first defined in our `Cloud Quick Start
Tutorial <https://www.run.house/docs/tutorials/quick-start-cloud>`__. As
a reminder, you may need to confirm that your cloud credentials are
properly configured by running ``sky check``.

.. code:: ipython3

    cluster = rh.ondemand_cluster(
        name="rh-cluster",
        instance_type="CPU:2+",
        provider="aws"
    )

.. code:: ipython3

    def get_platform(a = 0):
            import platform
            return platform.platform()

    remote_get_platform = rh.function(get_platform).to(cluster)


.. parsed-literal::

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

.. code:: python

   import runhouse as rh

   if __name__ == "__main__":
       reloaded_fn = rh.function(name="get_platform")
       print(reloaded_fn())

The ``name`` used to reload the function is the method name by default.
You can customize a function name using the following syntax:

.. code:: python

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

-  Resource Management
   https://www.run.house/docs/tutorials/api-resources
-  Secrets Management https://www.run.house/docs/tutorials/api-secrets
