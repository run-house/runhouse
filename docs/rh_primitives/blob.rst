Blob
====================================
A Blob is a Runhouse primitive that represents an entity for storing data and lives inside of a :ref:`Folder`.


Blob Factory Method
~~~~~~~~~~~~~~~~~~~

.. autofunction:: runhouse.blob

Blob Class
~~~~~~~~~~

.. autoclass:: runhouse.Blob
   :members:
   :exclude-members:

    .. automethod:: __init__


API Usage
~~~~~~~~~

To initalize a Blob, use ``rh.blob``:

.. code:: python

   data = json.dumps(list(range(50)))

   # Remote blob with name and no path (saved to bucket called runhouse/blobs/my-blob)
   rh.blob(name="@/my-blob", data=data, data_source='s3', dryrun=False)

   # Remote blob with name and path
   rh.blob(name='@/my-blob', path='/runhouse-tests/my_blob.pickle', data=data, system='s3', dryrun=False)

   # Local blob with name and path, save to local filesystem
   rh.blob(name=name, data=data, path=str(Path.cwd() / "my_blob.pickle"), dryrun=False)

   # Local blob with name and no path (saved to ~/.cache/blobs/my-blob)
   rh.blob(name="~/my-blob", data=data, dryrun=False)

To load an existing blob by name:

.. code:: python

   my_local_blob = rh.blob(name="~/my_blob")
   my_s3_blob = rh.blob(name="@/my_blob")

To get the contets from the blob:

.. code:: python

   raw_data = my_local_blob.fetch()
   result = pickle.loads(raw_data)  # deserialization
