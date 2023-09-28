Security and Authentication
===========================
By default, Runhouse collects metadata from provisioned clusters and data relating to performance and error monitoring.
This data will only be used by Runhouse to improve the product.
No Personal Identifiable Information (PII) is collected and we will not sell or buy data about you.

Cluster Metadata Collection
---------------------------
We collect non-sensitive data on the cluster that helps us understand how Runhouse is being used. This data includes:

- Python version
- Resources (cpus, gpus, memory)
- Cloud provider
- Region
- Instance type


Performance & Error Monitoring
------------------------------------
We collect data from normal package usage that helps us understand the performance and errors that are raised.
This data is collected and sent to `Sentry <https://sentry.io/>`_, a third-party error tracking service.

Removing Collected Data
------------------------------------
If you would like us to remove your collected data, please contact us.

Disabling Data Collection
-----------------------------------
To disable data collection and error tracking collection, set :code:`disable_data_collection` to :code:`true` in your
local Runhouse config (:code:`~/.rh/config.yaml`), or in Python:

.. code-block:: python

    import runhouse as rh
    rh.configs.disable_data_collection()

Cluster Authentication & Verification
-------------------------------------
Runhouse provides a couple of options to manage the connection to the Runhouse API server running on a cluster.


API Server Connection
~~~~~~~~~~~~~~~~~~~~~

The below options can be specified with the ``server_connection_type`` parameter
when :ref:`initializing a cluster <Cluster Factory Method>`:

- ``ssh``: Connects to the cluster via port forwarding. The API server will be started with HTTP.
- ``tls``: Connects to the cluster via port forwarding and enforces verification via TLS certificates. The API server
  will be started with HTTPS. Only users with a valid cert will be able to make requests to the API server.
- ``none``: Does not use any port forwarding or enforce any authentication. The API server will be started
  with HTTP.

.. note::

    The ``tls`` connection type is the most secure and is recommended for production use if you are not running inside
    of a VPC.

API Server Auth
~~~~~~~~~~~~~~~

Runhouse allows you to authenticate users via their Runhouse token (generated when
:ref:`logging in <Login/Logout>`) and saved to local Runhouse configs in path: :code:`~/.rh/config.yaml`.

When :ref:`initializing a cluster <Cluster Factory Method>`, you can set the :code:`den_auth` parameter to :code:`True`
to enable token authentication. Runhouse will handle adding the token to each subsequent request as an auth header with
format: :code:`{"Authorization": "Bearer <token>"}`

Enabling TLS and Den Auth for the API server makes it incredibly fast and easy to stand up a microservice with
standard token authentication, allowing you to easily share Runhouse resources with collaborators, teams,
customers, etc.

Let's illustrate this with a simple example:

.. code-block:: python

    import runhouse as rh

    def np_array(num_list: list):
        import numpy as np

        return np.array(num_list)

    # Launch a cluster with TLS and Den Auth enabled
    cpu = rh.ondemand_cluster(instance_type="m5.xlarge",
                              provider="aws",
                              name="rh-cluster",
                              den_auth=True,
                              server_connection_type="tls").up_if_not()

    # Remote function stub which lives on the cluster
    remote_func = rh.function(np_array).to(cpu, env=["numpy"])
    my_list = [1,2,3]

    # Run function on cluster
    # Note: only users with a Runhouse token and access to this cluster can call this function
    res = remote_func(my_list)


.. note::

    For more examples on using clusters and functions see
    the :ref:`Compute Guide <Compute: Clusters, Functions, Packages, & Envs>`.
