Send
====================================

A Send is a contraction for `"Serverless-Endpoint"`.
Quite literally you are taking your code and `"sending"` it to the cloud to be run as a serverless endpoint, or microservice.

It contains the following attributes:

- :code:`name`: Name the send to re-use later on.
- :code:`fn`: The function which will execute on the remote cluster when this send is called.
- :code:`hardware`: Hardware to use for the Send, either a string name of a Cluster object, or a Cluster object.
- :code:`package`: Package to send to the remote cluster, either a string name of a Package, package url, or a Package object.
- :code:`reqs`: List of requirements to install on the remote cluster, or path to a requirements.txt file. If a list of pypi packages is provided, including 'requirements.txt' in the list will install the requirements in `package`. By default, if reqs is left as None, we'll set it to ['requirements.txt'], which installs just the requirements of package. If an empty list is provided, no requirements will be installed.
- :code:`image`: Docker image id to use on the remote cluster, or path to Dockerfile.
- :code:`dryrun`: Whether to create the Send if it doesn't exist, or load the Send object as a dryrun.


.. note::
    For a tutorial on creating and sharing a send see the :ref:`Compute Layer` guide.


.. autoclass:: runhouse.rns.send.Send
   :members:
   :exclude-members:

    .. automethod:: __init__

Send Factory Method
~~~~~~~~~~~~~~~~~~~

.. autofunction:: runhouse.rns.send.send
