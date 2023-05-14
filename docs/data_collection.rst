Security and Metadata Collection
================================
By default, Runhouse collects metadata from provisioned clusters.
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

If you would like us to remove your collected data, please contact us.

Disabling Data Collection
-----------------------------------
To disable data collection, set :code:`disable_data_collection` to :code:`true` in your local Runhouse config
(:code:`~/.rh/config.yaml`), or in Python:

.. code-block:: python

    import runhouse as rh
    rh.configs.disable_data_collection()
