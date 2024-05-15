Den Quick Start
===============

.. raw:: html

    <p><a href="https://colab.research.google.com/github/run-house/notebooks/blob/stable/docs/quick-start-den.ipynb">
    <img height="20px" width="117px" src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a></p>

`Runhouse Den <https://www.run.house/dashboard>`__ lets you manage and
track your infra, services, and resources (clusters, functions, secrets,
etc). These resources can be easily reloaded from any environment, and
ready to be used without additional setup, or even shared with another
user or teammate. Then, in the Web UI, access, visualize, and manage
your resources, along with version history.

Account Creation & Login
------------------------

You can create an account on the `run.house <https://www.run.house>`__
website, or by calling the login command in Python or CLI.

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
Tutorial <https://www.run.house/docs/tutorials/quick-start-cloud>`__.

.. code:: ipython3

    cluster = rh.ondemand_cluster(
        name="rh-cluster",
        instance_type="CPU:2+",
        provider="aws"
    )



.. parsed-literal::
    :class: code-output

    Output()



.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"></pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">
    </pre>



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


You can save the resources we created above with:

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

::

   """
   import runhouse as rh

   if __name__ == "__main__":
       reloaded_fn = rh.function(name="get_platform")
       print(reloaded_fn())
   """

Sharing
-------

You can also share your resource with collaborators, and choose which
level of access to give. Once shared, they will be able to see the
resource in their dashboard as well, and be able to load and use the
shared resource. They’ll need to load the resource using its full name,
which includes your username (``/your_username/get_platform``).

.. code:: ipython3

    remote_fn.share(
        users=["teammate1@email.com"],
        access_level="write",
    )

Web UI
------

After saving your resources, you can log in and see them on your `Den
dashboard <https://www.run.house/dashboard>`__, labeled as
``<username>/rh-cluster`` and ``<username>/get_platform``.

Clicking into the resource provides information about your resource. You
can view the resource metadata, previous versions, and activity, or add
a description to the resource.

Dive Deeper
-----------

Check on more in-depth tutorials on:

-  Resource Management
   https://www.run.house/docs/tutorials/api-resources
-  Secrets Management https://www.run.house/docs/tutorials/api-secrets
