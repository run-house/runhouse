Architecture Overview
=====================



Runhouse Resources
~~~~~~~~~~~~~~~~~~

Resources are the Runhouse primitive for objects that can be saved, shared, and reused. This can be split
into compute resources (clusters, functions, environments, and runs) and data resources (folder, table, blob, etc).

Compute
-------

The Compute APIs allow a seamless flow of code and execution across local and remote compute. They blur the line
between program execution and deployment, providing both a path of least resistence for running a sub-routine on
specific hardware, while unceremoniously turning that sub-routine into a reusable service. They also provide
convenient dependency isolation and management, provider-agnostic provisioning and termination, and rich
debugging and accessibility interfaces built-in.

* **Cluster**: A set of machines which can be sent code or data. Generally, they are Ray clusters under the hood.

* **Environment**: A set of packages to be installed via HTTP on remote clusters.

* **Functions**: Functions are associated with clusters and environments, and are executed using an HTTP endpoint.

Data
-------

The Data APIs provide a simple interface for storing, recalling, and moving data between the user's laptop,
remote compute, cloud storage, and specialized storage (e.g. data warehouses). They provide least-common-denominator
APIs across providers, allowing users to easily specify the actions they want to take on the data without needed to
dig into provider-specific APIs.

* **Folder**: Represents a specified location (could be local, remote, or file storage), for managing where various
  Runhouse resources live.

* **Table**: Provides convenient APIs for writing, partitioning, fetch, and stream various data types.

* **Blob**: Represents a single serialized file stored in a particular system.

Accessibility and Management
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Runhouse enables the portability and sharing of resources across users and environments, and provides
tools for visibility and management of these resources as long-living assets shared by teams or projects.

Accessibility
-------------

The Runhouse RNS (Resource Naming System) provides a convenient way to name, persist, and recall resources
acoss environments. Meanwhile, the Secrets APIs provide a simple interface for storing and retrieving secrets
from your secure Runhouse account.

* **Resource Naming System (RNS)**: Consists of lightweight metadata for each resource type to captures the
  information needed to load it in a new environment, and a mechanism for saving and loading from either the working
  git repo or a remote Runhouse key-value metadata store. The git-based approach (Local RNS) allows for local
  persistence and versioning or sharing across OSS projects. The metadata store (Runhouse RNS) is even more portable;
  it allows resource sharing across users and environments, anywhere there is an Internet connection and Python
  interpreter. The RNS is backed by a management API (see below) to view and manage all resources.

* **Secrets API**: Provides a simple interface for storing and retrieving secrets to a allow a more seamless
  experience when accessing resources across environments. It provides a simple interface for storing and retrieving
  secrets from a variety of providers (e.g. AWS, Azure, GCP, Hugging Face, Github, etc.) as well as SSH Keys and
  custom secrets, and stores them in Hashicorp Vault.

* **Configs**: Set and preserve default configs across environments.

Management
----------

Runhouse provides tools for visibility and management of resources as long-living assets shared by teams or projects.

* **Organization structure**: Both resources and users can be organized into arbitrarily-nested groups to apply access
  permissions, default behaviors (e.g. default storage locations, compute providers, instance autotermination, etc.),
  project delineation, or staging (e.g. dev vs. prod).

* `Management UI <https://www.run.house/dashboard>`__: provides an individual or admin view of all resources, secrets,
  groups, and sharing. Resource metadata is automatically versioned in RNS, allowing teams to maintain single-sources
  of truth for assets with zero downtime to update or roll back, and trace exact lineage for any existing resource.
