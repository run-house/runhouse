Den Quick Start
===============

.. raw:: html

    <p><a href="https://colab.research.google.com/github/run-house/notebooks/blob/stable/docs/getting_started/den_quick_start.ipynb">
    <img height="20px" width="117px" src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a></p>

`Runhouse Den <https://www.run.house/dashboard>`__ let’s you save and
keep track of your Runhouse resources (cluster, function, data, etc).
These resources can be easily reloaded from any environment, and ready
to be used without additional setup, or even shared with another user or
teammate. Then, in the Web UI, access, visualize, and manage your
resources, along with version history.

Account Creation & Login
------------------------

You can create an account on the `run.house <https://run.house>`__
website, or by calling the login command in Python or CLI.

To login on your dev environment, call ``rh.login()`` in Python or
``runhouse login`` in CLI.

.. code:: ipython3

    rh.login()

As part of logging in, Runhouse also optionally offers secrets
management, where it can automatically detect locally set up provider
secrets, and gives you the option to upload them securely into your
account. For more information on Secrets management, refer to the
`Secrets
Tutorial <https://www.run.house/docs/main/en/tutorials/api_basics/secrets>`__.

Saving
------

Let’s start by constructing some runhouse resources that we’d like to
save down. These resources are taking from our `Cloud Quick Start
Tutorial <run.house/docs/tutorials/quick_start_cloud>`__.

.. code:: ipython3

    import runhouse as rh

.. code:: ipython3

    cluster = rh.ondemand_cluster(
        name="rh-cluster",
        instance_type="CPU:2+",
        provider="aws"
    )

.. code:: ipython3

    def get_pid(a = 0):
        import os
        return os.getpid()

    remote_fn = rh.function(get_pid).to(cluster)

You can save the resources we created above with:

.. code:: ipython3

    cluster.save()
    remote_fn.save()

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
       reloaded_fn = rh.function(name="get_pid")
       print(reloaded_fn())
   """

Sharing
-------

You can also share your resource with another user, and choose which
level of access to give. Once shared, they will be able to see the
resource in their dashboard as well, and be able to load and use the
shared resource.

.. code:: ipython3

    remote_fn.share(
        users=["teammate1@email.com"],
        access_level="write",
    )

Web UI
------

After saving your resources, you can log in and see them on your `Den
dashboard <https://www.run.house/dashboard>`__, labeled as
``<username>/rh-cluster`` and ``<username>/get_pid``.

Clicking into the resource provides information about your resource. You
can view the resource metadata, previous versions, and activity, or add
a description to the resource.

Dive Deeper
-----------

Check on more in-depth tutorials on:

-  Resource Management
   https://www.run.house/docs/tutorials/api_resources
-  Secrets Management
   https://www.run.house/docs/tutorials/api_secrets
