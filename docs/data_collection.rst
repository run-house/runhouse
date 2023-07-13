Security and Metadata Collection
================================
By default, Runhouse collects metadata from provisioned clusters and data relating to performance and error monitoring.
This data will only be used by Runhouse to improve the product.
No Personal Identifiable Information (PII) is collected and we will not sell or buy data about you.

Cluster Metadata
------------------------------------
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
