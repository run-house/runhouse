Python API
====================================
Runhouse offers a programmatic API in Python to manage your account and resources.


Resources
------------------------------------
Resources are the Runhouse abstraction for objects that can be saved, shared, and reused.
This includes both compute abstractions (clusters, functions, packages, environments) and
data abstractions (folders).

.. toctree::
   :maxdepth: 1

   python/resource


Compute Abstractions
------------------------------------
The Function, Cluster, Env, Package, and Module APIs allow a seamless flow of code and execution across local and remote compute.
They blur the line between program execution and deployment, providing both a path of least resistence for running
a sub-routine on specific hardware, while unceremoniously turning that sub-routine into a reusable service.
They also provide convenient dependency isolation and management, provider-agnostic provisioning and termination,
and rich debugging and accessibility interfaces built-in.

.. toctree::
   :maxdepth: 1

   python/function

.. toctree::
   :maxdepth: 1

   python/cluster

.. toctree::
   :maxdepth: 1

   python/image

.. toctree::
   :maxdepth: 1

   python/package

.. toctree::
    :maxdepth: 1

    python/module


Data Abstractions
------------------------------------
The Folder APIs provide a simple interface for storing, recalling, and moving data between
the user's laptop, remote compute, and cloud storage (currently we support `S3` and `GCS`). They provide
least-common-denominator APIs across providers, allowing users to easily specify the actions
they want to take on the data without needed to dig into provider-specific APIs.

.. toctree::
   :maxdepth: 1

   python/folder



Secrets
------------------------------------
Runhouse provides a convenient interface for managing your secrets in a secure manner.
Secrets are stored in `Vault <https://www.vaultproject.io/>`__, an industry standard for
secrets management, and never touches Runhouse servers. Please see
:ref:`Security and Authentication` for more information on security.

.. toctree::
   :maxdepth: 1

   python/secrets

.. toctree::
   :maxdepth: 1

   python/login
