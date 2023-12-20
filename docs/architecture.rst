Architecture Overview
=====================



Runhouse Resources
~~~~~~~~~~~~~~~~~~

Resources are the Runhouse primitive for objects that can be saved, shared, and reused. This can be split
into compute resources (clusters, functions, modules, environments, and runs) and data resources
(folder, table, blob, etc).

Compute
-------

The Compute APIs allow a seamless flow of code and execution across local and remote compute. They blur the line
between program execution and deployment, providing both a path of least resistence for running a sub-routine on
specific hardware, while unceremoniously turning that sub-routine into a reusable service. They also provide
convenient dependency isolation and management, provider-agnostic provisioning and termination, and rich
debugging and accessibility interfaces built-in.

* **Cluster**: A set of machines which can be sent code or data. Generally, they are Ray clusters under the hood.

* **Environment**: An environment respresents a compute environment, consisting of packages and environment variables.
  Each remote environment on a cluster is associated with a Ray Actor servlet, which handles all activities within the
  environement (calling functions, installing packages, getting/putting objects).

* **Function**: Functions are associated with clusters and environments, and are executed using an HTTP endpoint.

* **Module**: Modules represent classes that can be sent to and used on remote clusters and environments. Modules
  can live on remote hardware and its class methods called remotely.

Data
-------

The Data APIs provide a simple interface for storing, recalling, and moving data between the user's laptop,
remote compute, cloud storage, and specialized storage (e.g. data warehouses). They provide least-common-denominator
APIs across providers, allowing users to easily specify the actions they want to take on the data without needed to
dig into provider-specific APIs.

* **Folder**: Represents a specified location (could be local, remote, or file storage), for managing where various
  Runhouse resources live.

* **Table**: Provides convenient APIs for writing, partitioning, fetch, and stream various data types.

* **Blob**: Represents a data object stored in a particular system.

* **File**: Represents a file object stored in a particular system.

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

* `Runhouse Den <https://www.run.house/dashboard>`__: provides an individual or admin view of all resources, secrets,
  groups, and sharing. Resource metadata is automatically versioned in RNS, allowing teams to maintain single-sources
  of truth for assets with zero downtime to update or roll back, and trace exact lineage for any existing resource.

Resource Access Levels
----------------------

Runhouse lets you control the access levels you provide to the users of your resources:

- :code:`read` (default): Users cannot modify the config, but can access the resource directly.
- :code:`write`: Users have full control over the resource, including modifying its config.


Resource Visibility
-------------------

In addition to providing control over the access levels, Runhouse lets you control the visibility you provide
to your users.

Similar to Google Drive, you can choose how visible you would like each resource to be:

- :code:`private` (default): Only users who have been granted access explicitly to the resource can view and access it.
- :code:`unlisted`: Users can access and search for the resource by its name only (will not be vieweable by default).
- :code:`public`: Resource will be available publicly to all users.

You can specify these options when sharing any Runhouse resource.
Here are some examples for how that could look when sharing a function:

.. code-block:: python

    # Share a function with a particular user, giving them write access to the resource and private visibility
    # The function will be visible to the user in their Den dashboard
    my_func.share("user_a@gmail.com", access_level="write", visibility="private")

    # Make this function available to all users, who will be given read access to that resource
    my_func.share(visibility="public")

    # Share the function with a list of users, giving them read only access to the resource and unlisted visibility,
    # The function will not be visible by default in these users' Den dashboards, and will only appear
    # if searched by name
    my_func.share(["user_a@gmail.com", "rh_username_2"],
                  visibility="unlisted")
