Command Line Interface
------------------------------------
Runhouse provides CLI commands for logging in/out, and for basic interaction
with the cluster.

The commands can be run like follows:

.. code-block:: console

   $ runhouse login
   $ runhouse cluster ssh cluster_name

.. automodule:: runhouse.main
   :members: login, logout, cluster_ssh, server_start, server_restart, server_stop, server_status, cluster_status, cluster_list, cluster_keep_warm, cluster_up, cluster_down, cluster_logs
   :undoc-members:
   :show-inheritance:
