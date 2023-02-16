Accessibility, Portability, and Sharing
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

1. **Local RNS** - The git-based approach allows you to publish the exact resource metadata in the same version tree as your code, so you can be sure that the code and resources are always 1-for-1 compatible. It also is a highly visible way to distribute the resources to interested OSS users, who can see it right in the repo, rather than having to be aware that it exists behind an API. Imagine you publish some research, and the exact cloud configurations and data artifacts you used were published with it so consumers of the work don't need to reverse engineer your compute and data rig.
2. **Runhouse RNS** - The RNS API allows your resources to be accessible anywhere with an internet connection and Python interpreter, so it's obviously way more portable. It also allows you to quickly share resources with collaborators without needing to check them into git and ask them to fetch and change their branch. The web-based approach also allows you to keep a global source of truth for a resource (e.g. a single BERT preprocessing service shared by a team, or a most up-to-date model checkpoint), which will be updated with zero downtime by all consumers when you push a new version. Lastly, the RNS API is backed by a management API to view and manage all resources.

.. note::
    Not every resource in Runhouse is named. You can use the Runhouse APIs if you like the ergonomics without ever
    naming anything. Anonymous resources are simply never written to a metadata store.


Every named resource has a name and "full name" at :code:`resource.rns_address`, which is organized into
hierarchical folders. When you create a resource, you can name= it with just a name (we will resolve it as being in
the :code:`rh.current_folder()`) or the full address. Resources in the local RNS begin with the ~ folder.
Resources built-into the Runhouse Python package begin with ^ (like a house). All other addresses are in the
Runhouse RNS. By default, the only top-level folders in the Runhouse RNS you have permission to write to are your
username and any organizations you are in. The @ alises to your username - for example:

.. code-block:: python

   my_resource.save(name='@/myresource')

To persist a resource, call:

.. code-block:: python

    resource.save()
    resource.save(name='new_name')  # Saves to rh.current_folder()
    resource.save(name='@/my_full/new_name')  # Saves to Runhouse RNS
    resource.save(name='~/my_full/new_name')  # Saves to Local RNS



To load a resource, call :code:`rh.load('my_name')`, or just call the resource factory constructor with
only the name, e.g.

.. code-block:: python

    rh.function(name='my_function')
    rh.cluster(name='~/my_name')
    rh.table(name='@/my_datasets/my_table')

You may need to pass the full rns_address if the resource is not in rh.current_folder(). To check if a resource exists, you can call:

.. code-block:: python

    rh.exists(name='my_function')
    rh.exists(name='~/local_resource')
    rh.exists(name='@/my/rns_path/to/my_table')

We're still early in uncovering the patterns and antipatterns for a global shared environment for compute and data resources (shocker), but for now we generally encourage OSS projects to publish resources in the local RNS of their package, and individuals and teams to largely rely on Runhouse RNS.


Secrets and Logging In & Out
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Using Runhouse across environments, such as reusing a service from inside a Colab or loading secrets or configs
into a remote environment, is much easier if you create a Runhouse account. You don't need to do this if you only plan
to use Runhouse's APIs in a single environment, and don't plan to share resources with others.

.. tip::
    Logging in simply saves your token to :code:`~/.rh/config.yaml`, and offers to download or upload your secrets or
    defaults (e.g. default provider, autostop, etc.).


**Logging In:**

.. code-block:: console

    $ runhouse login

Run this wherever your cloud credentials are already saved, such as your laptop.
Follow the prompts to log in. If this is your first time logging in, you should probably upload
your secrets, and none of the other prompts will have any real effect (you probably haven't set any defaults yet):

or in Python (e.g. in a notebook)

.. code-block:: python

    rh.login(interactive=True)


**Logging Out:**

.. code-block:: console

    $ runhouse logout

Run this wherever your cloud credentials are already saved.

or in Python (e.g. in a notebook)

.. code-block:: python

    rh.logout(interactive=True)


Setting Config Options
~~~~~~~~~~~~~~~~~~~~~~

Runhouse stores user configs both locally in :code:`~/.rh/config.yaml` and remotely in the Runhouse database.
This allows you to preserve your same config across environments. Some important configs to consider setting:

Whether to use spot instances (cheaper but can be reclaimed at any time) by default.
Note that this is :code:`False` by default because you'll need to request spot quota from the cloud providers to use spot
instances. You can override this setting in the cluster factory constructor:

.. code-block:: python

    rh.configs.set('use_spot', False)


Clusters can start and stop dynamically to save money. If you set :code:`autostop = 10`, the cluster will terminate after
10 minutes of inactivity. If you set :code:`autostop = -1`, the cluster will stay up indefinitely.
After the cluster terminates, if you call a Function which is on that cluster, the Function will automatically start the
cluster again. You can also call :code:`cluster.keep_warm(autostop=-1)` to control this for an existing cluster:

.. code-block:: python

    rh.configs.set('default_autostop', 30)

You can set your default Cloud provider if you have multiple Cloud accounts set up locally.
If you set it to :code:`cheapest`, SkyPilot will select the cheapest provider for your desired hardware
(including spot pricing, if enabled). You can set this to :code:`aws`, :code:`gcp`, or :code:`azure` too:

.. code-block:: python

    rh.configs.set('default_provider', 'cheapest')


Now that you've changed some configs, you probably want to save them to Runhouse to access them elsewhere:

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
