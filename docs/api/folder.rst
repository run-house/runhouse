Folder
====================================
A Folder represents a specified location for organizing and storing other Runhouse primitives
across various systems.


Folder Factory Method
~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: runhouse.folder


Folder Class
~~~~~~~~~~~~

.. autoclass:: runhouse.Folder
   :members:
   :exclude-members:

    .. automethod:: __init__

API Usage
~~~~~~~~~

To initalize a Folder, use the ``rh.folder`` factory method.

.. code:: python

   rh.folder(name='folder_name', path='remote_directory/path_to_folder', system='s3')

To copy a folder from one system to another (in this case, from local to a cluster or s3):

.. code:: python

   local_cluster = rh.cluster(...).up_if_not()  # Instantiate cluster and check that it is up
   local_folder = rh.folder(Path.cwd(), name='my_local_folder')

   # Send the folder to my_cluster
   cluster_folder = local_folder.to(system=local_cluster)

   # Send the folder to s3
   s3_folder = rh.folder(name='my_local_folder').to(system="s3")
