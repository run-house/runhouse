Command Line Interface
------------------------------------
Runhouse provides CLI commands for logging in/out, and for basic interaction
with the cluster.

The commands can be run like follows:

.. code-block:: console

   $ runhouse login
   $ runhouse ssh cluster_name

.. automodule:: runhouse.main
   :members: login, logout, ssh, notebook, start, restart, status
   :undoc-members:
   :show-inheritance:
