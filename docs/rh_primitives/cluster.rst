Cluster
====================================
A cluster is a Runhouse primitive used for abstracting a particular hardware configuration.

It contains the following attributes:

- :code:`name`: Name the cluster to re-use later on.
- :code:`instance_type`: See below for default builtins supported. You may also create your own.
- :code:`num_instances`: Number of instances to create.
- :code:`create`: For the creation of a cluster for the first time should be :code:`True`.
- :code:`autostop_mins`: Automatically down the cluster after specified number of minutes.
- :code:`use_spot`: Use spot instance.
- :code:`sky_data`: Config data for the cluster. Runhouse will generally provide this for you under the hood.

Runhouse provides a number of convenient builtin default instance types:

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

.. autoclass:: runhouse.rns.hardware.skycluster.Cluster
   :members:
   :exclude-members:

    .. automethod:: __init__


Cluster Factory Method
~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: runhouse.rns.hardware.skycluster.cluster