Accessibility
=======================================

Resource Name System (RNS)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Cloud resources are already inherently portable, so making them accessible across environments and users in
Google-Docs-like manner only requires a bit of metadata and snappy resource APIs. For example, if you wanted all of
your collaborators to share a "data space" where you could reference files in blob storage by name
(instead of passing around lots of urls), you could stand up a key-value store mapping name to URL and an API
to resolve the names.

Now imagine you wanted to do this for tabular data, folders, and code packages, compute
instances, and services too, so you came up with a way of putting them into the KV store too. And now for each of
the above, you and your collaborators might have a number of providers underneath the resource (e.g. Parquet in S3,
DataBricks, Snowflake, etc.), and perhaps a number of variants (e.g. Pandas, Hugging Face, Dask, RAPIDS, etc.),
so you create a unified front-end into like resources and a dispatch system to make sure resources load properly based
on the various metadata morphologies.

Finally, you have lots of collaborators and resources and don't just want a
single massive global list of name strings, so you allow folder hierarchies.

There you go, you've built the Runhouse RNS. 🔥


We support saving resource metadata to the :code:`/rh` directory of the working git package or a remote metadata
service we call the Runhouse RNS API. Both have their advantages:

1. **Local RNS** - The git-based approach allows you to publish the exact resource metadata in the same version tree as your code, so you can be sure that the code and resources are always 1-for-1 compatible. It also is a highly visible way to distribute the resources to OSS users, who can see it right in the repo, rather than having to be aware that it exists behind an API. Imagine you publish some research, and the exact cloud configurations and data artifacts you used were published with it so consumers of the work don't need to reverse engineer your compute and data rig.
2. **Runhouse RNS** - The RNS API allows your resources to be accessible anywhere with an internet connection and Python interpreter, so it's way more portable. It also allows you to quickly share resources with collaborators without needing to check them into git and ask them to fetch and change their branch. The web-based approach also allows for a global source of truth for a resource (e.g. a single BERT preprocessing service shared by a team, or a most up-to-date model checkpoint), which will be updated with zero downtime by all consumers when you push a new version. Lastly, the RNS API is backed by a management API to view and manage all resources.

.. note::
    Not every resource in Runhouse is named. You can use the Runhouse APIs if you like the ergonomics without ever
    naming anything. Anonymous resources are simply never written to a metadata store.

.. note::
    By default, the only top-level folders in the Runhouse RNS you have permission to write to are your
    username and any organizations you are in.

We're still early in uncovering the patterns and antipatterns for a global shared environment for compute and data resources (shocker),
but for now we generally encourage OSS projects to publish resources in the local RNS of their package, and individuals and teams to largely rely on Runhouse RNS.


Secrets and Login / Logout
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Using Runhouse across environments, such as reusing a service from inside a Colab or loading secrets or configs
into a remote environment, is much easier if you create a Runhouse account. You don't need to do this if you only plan
to use Runhouse's APIs in a single environment, and don't plan to share resources with others.

.. note::
    Logging in simply saves your token to :code:`~/.rh/config.yaml`, and offers to download or upload your secrets or
    defaults (e.g. :code:`default_provider`, :code:`autostop`, etc.).


**Logging In:**

Run this wherever your cloud credentials are already saved, such as your laptop.
Follow the prompts to log in. If this is your first time logging in, you should probably upload
your secrets, and none of the other prompts will have any real effect (you probably haven't set any defaults yet):

.. code-block:: console

    $ runhouse login

or in Python (e.g. in a notebook)

.. code-block:: python

    rh.login(interactive=True)


**Logging Out:**

Run this wherever your cloud credentials are already saved.

.. code-block:: console

    $ runhouse logout

or in Python

.. code-block:: python

    rh.logout(interactive=True)


.. tip::
    See our :ref:`Secrets API <Secrets Management>` and :ref:`usage examples <Secrets in Vault>` to see how Runhouse
    allows you to make your secrets available across different environments.

Setting Config Options
~~~~~~~~~~~~~~~~~~~~~~

Runhouse stores user configs both locally in :code:`~/.rh/config.yaml` and remotely in the Runhouse database.
This allows you to preserve your same config across environments. Some important configs to consider setting:

Whether to use spot instances, which are cheaper but can be reclaimed at any time.
This is :code:`False` by default because you'll need to request spot quota from the cloud providers to use spot
instances.

.. code-block:: python

    rh.configs.set('use_spot', True)


Default autostop time for the Cluster, to dynamically stop the cluster after inactivity to save money.
The cluster will stay up for the specified amount of time (in minutes) after inactivity,
or indefinitely if `-1` is provided. Calling a Function on the cluster after the cluster terminates will
automatically restart the cluster. You can also call :code:`cluster.keep_warm(autostop=-1)` to control
this for an existing cluster:

.. code-block:: python

    rh.configs.set('default_autostop', 30)

Default Cloud provider, if you have multiple Cloud accounts set up locally.
Setting it to :code:`cheapest` will use the cheapest provider (through SkyPilot) for your desired hardware,
(including spot pricing, if enabled). Other options are :code:`aws`, :code:`gcp`, :code:`azure`, or :code:`lambda`

.. code-block:: python

    rh.configs.set('default_provider', 'cheapest')


To save updated configs to Runhouse to access them elsewhere:

.. code-block:: python

    rh.configs.upload_defaults()


Viewing RPC Logs
~~~~~~~~~~~~~~~~
If you didn't run your function with :code:`stream_logs=True` and otherwise need to see the logs for Runhouse
on a particular cluster, you can ssh into the cluster with :code:`ssh <cluster name>` and :code:`screen -r` (and use control A+D to exit.
If you control-C you will stop the server). The server runs inside that screen instance, so logs are written to there.

Restarting the RPC Server
~~~~~~~~~~~~~~~~~~~~~~~~~
Sometimes the RPC server will crash, or you'll update a package that the server has already imported.
In those cases, you can try to restart just the server (~20 seconds) to save yourself the trouble of nuking and
reallocating the hardware itself (minutes). You can do this by running:

.. code-block:: python

    my_cluster.restart_grpc_server()


Notebooks
~~~~~~~~~

If you prefer to work or debug in notebooks, you can call the following to tunnel a JupyterLab server into your local
browser from your Runhouse cluster or function:

.. code-block:: console

    $ runhouse notebook my_cluster

or in Python:

.. code-block:: python

    my_cluster.notebook()

If you'd like to use a hosted notebook service like Colab, you'll benefit a lot from creating a
Runhouse account to store your secrets and loading them into Colab with :code:`rh.login()`.
This is not required, and you can still drop them into the Colab VM manually.


Notes on Notebooks
------------------
Notebooks are funny beasts. The code and variable inside them are not designed to be reused to shuttled around. As such:

1. If you want to :code:`rh.function` a function defined inside the notebook, it cannot contain variables or imports from outside the function, and you should assign a :code:`name` to the function. We will write the function out to a separate :code:`.py` file and import it from there, and the filename will be set to the :code:`function.name`.
2. If you really want to use local variables or avoid writing out the function, you can set :code:`serialize_notebook_fn=True` in :code:`rh.function()`. This will cloudpickle the function before sending it, but we do not support saving and reloading these kinds of functions (cloudpickle does not support this kind of reuse and it will create issues).
3. It is nearly always better to try to write your code in a :code:`.py` file somewhere and import it into the notebook, rather than define important functions in the notebook itself. You can also use the :code:`%%writefile` magic to write your code into a file, and then import it back into the notebook.

If you want to sync down your code or data to local from the cluster afterwards:

.. code-block:: python

    rh.folder(path='remote_directory', system=rh.cluster('my_cluster').to('here', path='local_directory'))
