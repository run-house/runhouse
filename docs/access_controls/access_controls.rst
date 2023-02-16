Access Controls
====================================

We currently provide three types of access controls across all Runhouse resources:

- :code:`Write`: Full control over the resource, including modifying the config for the resource.
- :code:`Read`: Cannot modify the config, but can access the resource directly (e.g. ssh level for a Function).
- :code:`Proxy`: Only have http access to the resource, unable to access the resource directly.

.. tip::

    Runhouse allows you to manage access to all resources via a single **access control plane**.
    You can share any resource with individual Runhouse accounts, your team, company, or even the general public.
