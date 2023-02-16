High Level Overview
====================================

Runhouse has four top-level objectives:

1. Allowing users to natively program across compute resources
2. Allowing users to command data between storage and compute
3. Making resources accessible across environments and users
4. Allowing resources to be shared among teams as living assets

It achieves the above by providing four pillar features:

Compute
~~~~~~~
The :ref:`Functions`, :ref:`Clusters`, and :ref:`Package` APIs allow a seamless flow of code and execution across local and remote compute.
They blur the line between program execution and deployment, providing both a path of least resistence for running a
sub-routine on specific hardware, while unceremoniously turning that sub-routine into a reusable service.

They also provide convenient dependency isolation and management, provider-agnostic provisioning and termination,
and rich debugging and accessibility interfaces built-in.


Data
~~~~
The :ref:`Folder`, :ref:`Table`, and :ref:`Blob` APIs provide a simple interface for storing, recalling, and moving data between the
user's laptop, remote compute, cloud storage, and specialized storage (e.g. data warehouses).
They provide least-common-denominator APIs across providers, allowing users to easily specify the actions they want
to take on the data without needed to dig into provider-specific APIs.

We'd like to extend this to other data concepts in the future, like kv-stores, time-series, vector and graph databases, etc.

Accessibility
~~~~~~~~~~~~~
Runhouse strives to provide a Google-Docs-like experience for portability and sharing of resources across users and environments.

The :ref:`Resource Name System (RNS)` allows resources to be named, persisted, and recalled across environments.
It consists of a lightweight metadata standard for each resource type which captures the information needed to load
it in a new environment (e.g. Folder -> provider, bucket, path, etc.), and a mechanism for saving and loading from
either the working git repo or a remote Runhouse key-value metadata store.

The metadata store allows resources to be shared across users and environments, while the git approach allows for
local persistence and versioning or sharing across OSS projects.

The :ref:`Secrets Management` API provides a simple interface for storing and retrieving secrets to a allow a more seamless
experience when accessing resources across environments. It provides a simple interface for storing and retrieving
secrets from a variety of providers (e.g. AWS, Azure, GCP, Hugging Face, Github, etc.) as well as SSH Keys and
custom secrets, and stores them in Hashicorp Vault.

Management
~~~~~~~~~~
Runhouse provides tools for visibility and management of resources as long-living assets shared by teams or projects.
Both resources and users can be organized into arbitrarily-nested groups to apply access permissions,
default behaviors (e.g. default storage locations, compute providers, instance autotermination, etc.),
project delineation, or staging (e.g. dev vs. prod).

The `management UI <https://api.run.house/>`_ provides an individual or admin view of all resources, secrets, groups,
and sharing (this is only an MVP, and will be overhauled soon). Resource metadata is automatically versioned in RNS,
allowing teams to maintain single-sources of truth for assets with zero downtime to update or roll back, and trace
exact lineage for any resource (assuming the underlying the resources are not being deleted).
We provide basic logging out of the box today, and are working on providing comprehensive logging, monitoring, alerting.
