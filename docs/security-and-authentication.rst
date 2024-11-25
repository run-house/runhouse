Security and Authentication
===========================
By default, Runhouse collects metadata from provisioned clusters and data relating to performance and error monitoring.
This data will only be used by Runhouse to improve the product.

Cluster Metadata Collection
---------------------------
We collect non-sensitive data on the cluster that helps us understand how Runhouse is being used. This data includes:

- Python version
- Resources (cpus, gpus, memory)
- Cloud provider
- Region
- Instance type


Disabling Data Collection
-----------------------------------
To disable data collection and error tracking collection, set the environment variable :code:`DISABLE_DATA_COLLECTION`
to :code:`True`. Alternatively, set :code:`disable_data_collection` to :code:`true` in your
local Runhouse config (:code:`~/.rh/config.yaml`), or in Python:

.. code-block:: python

    import runhouse as rh
    rh.configs.disable_data_collection()


Cluster Observability
---------------------------------------
Runhouse collects various telemetry data by default on clusters. This data will be used to provide better observability
into logs, traces, and metrics associated with clusters. We will not sell data or buy any observability data collected.

To disable observability globally for all clusters, set the environment variable :code:`disable_observability`
to :code:`True`. Alternatively, set :code:`disable_observability` to :code:`true` in your
local Runhouse config (:code:`~/.rh/config.yaml`), or in Python:

.. code-block:: python

    import runhouse as rh
    rh.configs.disable_observability()
