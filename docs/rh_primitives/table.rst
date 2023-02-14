Tables
====================================

Table
~~~~~

A table is a Runhouse primitive used for abstracting a particular tabular data storage configuration.

.. tip::
    Run :code:`fsspec.available_protocols()` for a list of available data sources. Runhouse provides
    explicit support for :code:`file`, :code:`github`, :code:`sftp`, :code:`ssh`, :code:`s3`, and
    :code:`gs`.

.. note::
    Check out this `tutorial <https://github.com/run-house/tutorials/tree/main/t05_BERT_pipeline/>`_
    for more details on creating and using a table.

.. autoclass:: runhouse.rns.tables.table.Table
   :members:
   :exclude-members:

    .. automethod:: __init__


Dask Table
~~~~~~~~~~

.. autoclass:: runhouse.rns.tables.dask_table.DaskTable
   :members:
   :exclude-members:

    .. automethod:: __init__


HuggingFace Table
~~~~~~~~~~~~~~~~~

.. autoclass:: runhouse.rns.tables.huggingface_table.HuggingFaceTable
   :members:
   :exclude-members:

    .. automethod:: __init__

Pandas Table
~~~~~~~~~~~~

.. autoclass:: runhouse.rns.tables.pandas_table.PandasTable
   :members:
   :exclude-members:

    .. automethod:: __init__

Ray Table
~~~~~~~~~

.. autoclass:: runhouse.rns.tables.ray_table.RayTable
   :members:
   :exclude-members:

    .. automethod:: __init__


Table Factory Method
~~~~~~~~~~~~~~~~~~~~

.. autofunction:: runhouse.rns.tables.table.table