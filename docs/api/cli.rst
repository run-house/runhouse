Command Line Interface
------------------------------------
Runhouse provides CLI commands for the following use cases:

* logging in and out (``runhouse login/logout``)
* interacting with or retrieving information about clusters (``runhouse cluster <cmd>``)
* interacting with the Runhouse server (``runhouse server <cmd>``)

The commands can be run using either ``runhouse`` or the ``rh``` alias

.. automodule:: runhouse.main
   :members: login, logout, cluster_ssh, server_start, server_restart, server_stop, server_status, cluster_status, cluster_list, cluster_keep_warm, cluster_up, cluster_down, cluster_logs
   :undoc-members:
   :show-inheritance:
