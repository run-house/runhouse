Python API
====================================
`Runhouse offers a programmatic API in Python to manage your account and resources.`


Resources
------------------------------------
Resources are the Runhouse abstraction for objects that can be saved, shared, and reused.
This includes both compute abstractions (clusters, functions, packages) and data abstractions
(blobs, folders, tables).

.. toctree::
   :maxdepth: 1

   rh_primitives/resource


Compute Abstractions
------------------------------------
The Function, Cluster, and Package APIs allow a seamless flow of code and execution across local and remote compute.
They blur the line between program execution and deployment, providing both a path of least resistence for running
a sub-routine on specific hardware, while unceremoniously turning that sub-routine into a reusable service.
They also provide convenient dependency isolation and management, provider-agnostic provisioning and termination,
and rich debugging and accessibility interfaces built-in.

.. toctree::
   :maxdepth: 1

   rh_primitives/cluster


.. toctree::
   :maxdepth: 1

   rh_primitives/function


.. toctree::
   :maxdepth: 1

   rh_primitives/package


Data Abstractions
------------------------------------
The Folder, Table, and Blob APIs provide a simple interface for storing, recalling, and moving data between
the user's laptop, remote compute, cloud storage, and specialized storage (e.g. data warehouses).
They provide least-common-denominator APIs across providers, allowing users to easily specify the actions
they want to take on the data without needed to dig into provider-specific APIs. We'd like to extend this
to other data concepts in the future, like kv-stores, time-series, vector and graph databases, etc.

.. toctree::
   :maxdepth: 1

   rh_primitives/blob


.. toctree::
   :maxdepth: 1

   rh_primitives/folder


.. toctree::
   :maxdepth: 1

   rh_primitives/table
