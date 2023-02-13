Cluster
====================================
A cluster is a Runhouse primitive used for abstracting a particular hardware configuration.
This can be either an on-demand cluster (requires valid cloud credentials) or a BYO
(bring-your-own) cluster (requires IP address and ssh creds).

A cluster is assigned a name, through which it can be accessed and reused later on.

Runhouse also provides a number of convenient builtin default instance types:

.. list-table::
   :widths: 25 25 25 25
   :header-rows: 1

   * - Name
     - GPUs
     - CPUs
     - RAM (GB)
   * - `rh-4-gpu`
     - 4
     - 16
     - 64
   * - `rh-4-v100`
     - 4
     - 16
     - 64
   * - `rh-8-cpu`
     - 0
     - 8
     - 32
   * - `rh-8-gpu`
     - 8
     - 32
     - 128
   * - `rh-32-cpu`
     - 0
     - 32
     - 128
   * - `rh-cpu`
     - 0
     - 1
     - 4
   * - `rh-gpu`
     - 1
     - 4
     - 16
   * - `rh-v100`
     - 1
     - 4
     - 16

.. tip::
    When using one of our builtin instances, you can reference it by using the name of the instance type
    preceded by :code:`^`. For example, :code:`^rh-4-gpu`.

.. autoclass:: runhouse.rns.hardware.cluster.Cluster
  :members:
  :exclude-members:

    .. automethod:: __init__

SkyCluster
~~~~~~~~~~
A SkyCluster is a cluster that uses SkyPilot functionality underneath to handle
various cluster properties.

.. autoclass:: runhouse.rns.hardware.skycluster.SkyCluster
   :members:
   :exclude-members:

    .. automethod:: __init__

Cluster Factory Method
~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: runhouse.rns.hardware.cluster_factory.cluster
