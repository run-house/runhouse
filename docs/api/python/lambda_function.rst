Lambda Function
====================================

An AWS Lambda Function is a serverless provided by AWS. It allows to run applications and backend services without
provisioning or managing servers. Runhouse will allow you to maintain and share your lambda functions and code.
It is comprised of the entry point, configuration, and dependencies necessary to run the service.

.. note::

    Lambda Function support is an alpha and under active development.
    Please report any bugs or let us know of any feature requests.

Lambda Function Factory Method
~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: runhouse.lambda_function

Lambda Function Class
~~~~~~~~~~~~~~

.. autoclass:: runhouse.LambdaFunction
   :members:
   :exclude-members:

    .. automethod:: __init__
