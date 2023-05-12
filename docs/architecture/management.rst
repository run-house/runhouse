Management
=======================================

Runhouse provides tools for visibility and management of resources as long-living assets shared by teams or projects.
Both resources and users can be organized into arbitrarily-nested groups to apply access permissions,
default behaviors (e.g. default storage locations, compute providers, instance autotermination, etc.),
project delineation, or staging (e.g. dev vs. prod).

The `Management UI <https://api.run.house/>`_ provides an individual or admin view of all resources, secrets, groups,
and sharing (this is only an MVP, and will be overhauled soon). Resource metadata is automatically versioned in RNS,
allowing teams to maintain single-sources of truth for assets with zero downtime to update or roll back, and trace
exact lineage for any resource (assuming the underlying the resources are not being deleted).
We provide basic logging out of the box today, and are working on providing comprehensive logging, monitoring, alerting.


.. tip::
    In addition to managing existing resources, you'll benefit a lot from creating a
    Runhouse account to store your secrets and load them into different environments (ex: Colab).
