Lambda Function
====================================

.. note::

    Lambda Function support is an alpha and under active development.
    Please report any bugs or let us know of any feature requests.

| A `Lambda Function <https://aws.amazon.com/lambda/>`_ is a serverless compute service provided by AWS. It allows  you
 to run applications and backend services without provisioning or managing servers. Runhouse will allow you to
 maintain, invoke and share your lambda functions and code. It is comprised of the entry point, configuration, and
 dependencies necessary to run the service.


| There are two core options to create an AWS Lambda using Runhouse:
#. Pass a callable Python function to the factory method.
#. Follow a typical Lambda creation flow (as if you are using AWS APIs). That includes passing path(s) to Python file(s)
   and providing a handler function name to the constructor. Arguments such as runtime, Lambda name, timeout
   and memory size are accepted by the factory method as well, but they are not mandatory and have default values.
#. Create rh.function instance, and than send it over to AWS Lambdas. (e.g. :code: `rh.function(summer).to(system=aws_lambda)`

More specific information about acceptable arguments can be found in the the Lambda function factory method
documentation below.




Lambda Function Factory Method
~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: runhouse.lambda_function

Lambda Function Class
~~~~~~~~~~~~~~

.. autoclass:: runhouse.LambdaFunction
   :members:
   :exclude-members: to

    .. automethod:: __init__

Lambda Hardware Setup
~~~~~~~~~~~~~~

In order to create an AWS Lambda, you must grant the necessary permissions to do so. It can be done using an IAM
role, which should be attached to a certain AWS profile. This setup is made in the :code:`~/.aws/config` file.

For example, your local :code:`~/.aws/config` contains:

.. code-block:: ini

    [profile lambda]
    role_arn = arn:aws:iam::123456789:role/lambda-creation-role
    region = us-east-1
    source_profile = default

There are several ways to provide the necessary credentials which enables Lambda creation:

#. Specify the profile name in your code editor:

    #. Install the `AWS Toolkit` extension.
    #. Choose the relevant profile (e.g. `profile lambda`) and region (e.g `us-east-1`).

#. Environment Variable: setting :code:`AWS_PROFILE` to :code:`"lambda"`

.. note::
    If no specific profile is provided, Runhouse will try using the default profile. Note if this default AWS identity
    will not have the relevant IAM permissions for creating a Lambda attached to it, you will not be able to create a
    Lambda.
