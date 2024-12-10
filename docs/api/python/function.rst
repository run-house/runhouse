Function
====================================

A Function is a portable code block that can be sent to remote hardware to run as a subroutine or service.
It is comprised of the entrypoint, system (:ref:`Cluster`), and requirements necessary to run it.


Function Factory Methods
~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: runhouse.function

Function Class
~~~~~~~~~~~~~~

.. autoclass:: runhouse.Function
   :members:
   :exclude-members: map, starmap, get_or_call

    .. automethod:: __init__
