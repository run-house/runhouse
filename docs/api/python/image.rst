Image
=====
A Runhouse image allows you to easily encapsulate various setup steps to take across each node on the cluster before
it is launched. See the :ref:`Images` section for a more in-depth explanation.

Image Class
~~~~~~~~~~~

.. autoclass:: runhouse.Image
   :members:
   :exclude-members:

    .. automethod:: __init__

ImageSteupStepType
~~~~~~~~~~~~~~~~~~

.. autoclass:: runhouse.resources.images.ImageSetupStepType

    .. autoattribute:: PACKAGES
    .. autoattribute:: CMD_RUN
    .. autoattribute:: SETUP_CONDA_ENV
    .. autoattribute:: RSYNC
    .. autoattribute:: SYNC_SECRETS
    .. autoattribute:: SET_ENV_VARS

ImageSetupStep
~~~~~~~~~~~~~~

.. autoclass:: runhouse.resources.images.ImageSetupStep
    :members:
    :exclude-members:

    .. automethod:: __init__
