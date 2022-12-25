Table
====================================

A table is a Runhouse primitive used for abstracting a particular tabular data storage configuration.

It contains the following attributes:

- :code:`data`: Data to be stored in the table.
- :code:`name`: Name the table to re-use later on.
- :code:`data_url`: Full path to the data file (e.g. :code:`my-test-bucket/my_table.parquet`)
- :code:`data_source`: FSSpec protocol (e.g. :code:`s3` or :code:`gcs`),
- :code:`data_config`: Metadata to store for the table.
- :code:`create`: For the creation of a table for the first time should be :code:`True`.
- :code:`partition_cols`: We support partitioning tables by columns.

.. tip::
    Run :code:`fsspec.available_protocols()` for a list of available data sources.


.. note::
    For a tutorial on creating a table see the :ref:`Data Layer`.


.. autoclass:: runhouse.rns.tables.table.Table
   :members:
   :exclude-members:

    .. automethod:: __init__


Table Factory Method
~~~~~~~~~~~~~~~~~~~~

.. autofunction:: runhouse.rns.tables.table.table