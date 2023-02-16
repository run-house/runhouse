Table
====================================
A table is a Runhouse primitive used for abstracting a particular tabular data storage configuration.


Table Factory Method
~~~~~~~~~~~~~~~~~~~~

.. autofunction:: runhouse.table

Table
~~~~~

.. tip::
    Run :code:`fsspec.available_protocols()` for a list of available data sources. Runhouse provides
    explicit support for :code:`file`, :code:`github`, :code:`sftp`, :code:`ssh`, :code:`s3`, and
    :code:`gs`.

.. note::
    Check out this `tutorial <https://github.com/run-house/tutorials/tree/main/t05_BERT_pipeline/>`_
    for more details on creating and using a table.

.. autoclass:: runhouse.Table
   :members:
   :exclude-members:

    .. automethod:: __init__
