Send
====================================

A Send is a contraction for `"Serverless-Endpoint"`.
Quite literally you are taking your code and `"sending"` it to the cloud to be run as a serverless endpoint, or microservice.
It comprises of the entrypoint, hardware/cluster, and requirements necessary to run a service.

.. autoclass:: runhouse.rns.send.Send
   :members:
   :exclude-members:

    .. automethod:: __init__

Send Factory Method
~~~~~~~~~~~~~~~~~~~

.. autofunction:: runhouse.rns.send.send
