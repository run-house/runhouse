Table
====================================
A Table is a Runhouse primitive used for abstracting a particular tabular data storage configuration.


Table Factory Method
~~~~~~~~~~~~~~~~~~~~

.. autofunction:: runhouse.table

Table Class
~~~~~~~~~~~

.. autoclass:: runhouse.Table
   :members:
   :exclude-members:

    .. automethod:: __init__


API Usage
~~~~~~~~~
To initialize a Runhouse Table, use the factory method ``rh.table``

.. code:: python

   my_table = rh.table(data=data,
                     name="~/my_pandas_table",
                     path="table/my_pandas_table.parquet",
                     system="file",
                    mkdir=True)

To load an existing table by name:

.. code:: python

   reloaded_table = rh.table(name="~/my_pandas_table", dryrun=True)

This ``reloaded_table`` holds a reference to the Table's path, and can be used as follows:

.. code:: python

   batches = reloaded_table.stream(batch_size=100)
      for batch in batches:
         ....
