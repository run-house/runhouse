Resource Basics
===============

Resources are the Runhouse abstraction for objects that can be saved, shared, and reused.

Every named resource has a name ``my_resource.name`` and "full name" ``my_resource.rns_address``, and
is organized into heirarchical folders. By default, providing a ``name`` resolves it as being in
``rh.current_folder()`` or the full address. Resources in the local RNS begin with the ``~`` folder,
and Resources built-into the Runhouse Python package begin with ``^`` (like a house). ``@`` aliases to
your username.

.. code-block:: python

   my_resource.save(name='@/myresource')

To persist a resource, call:

.. code-block:: python

    resource.save()
    resource.save(name='new_name')  # Saves to rh.current_folder()
    resource.save(name='@/my_full/new_name')  # Saves to Runhouse RNS
    resource.save(name='~/my_full/new_name')  # Saves to Local RNS

To load a resource, you can call :code:`rh.load('my_name')`, or use the resource factory constructor with
just the name.

.. code-block:: python

    rh.function(name='my_function')
    rh.cluster(name='~/my_name')
    rh.table(name='@/my_datasets/my_table')

You may need to pass the full rns_address if the resource is not in :code:`rh.current_folder()`. To check
if a resource exists, you can call:

.. code-block:: python

    rh.exists(name='my_function')
    rh.exists(name='~/local_resource')
    rh.exists(name='@/my/rns_path/to/my_table')
