Function
====================================

A Function is a portable code block that can be sent to remote hardware to run as a subroutine or service.
It is comprised of the entrypoint, system (:ref:`Cluster`), and requirements necessary to run it.


Function Factory Methods
~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: runhouse.function

.. autofunction:: runhouse.aws_lambda_fn

Function Class
~~~~~~~~~~~~~~

.. autoclass:: runhouse.Function
   :members:
   :exclude-members: map, starmap, get_or_call, send_secrets

    .. automethod:: __init__

Lambda Function Class
~~~~~~~~~~~~~~~~~~~~~

.. note::

    Lambda Function support is an alpha and under active development.
    Please report any bugs and let us know of any feature requests.

| A `Lambda Function <https://aws.amazon.com/lambda/>`__ is a serverless compute service provided by AWS. It allows  you
 to run applications and backend services without provisioning or managing servers. Runhouse will allow you to
 maintain, invoke and share your Lambda functions and their code. It is comprised of the entry point, configuration, and
 dependencies necessary to run the service.


| There are few core options to create an AWS Lambda using Runhouse:

#. Pass a callable Python function to the factory method.
#. Follow a typical Lambda creation flow (as if you are using AWS APIs). That includes passing path(s) to Python files(s)
   (containing the code) and a handler function name to the from_handler_file() method. Arguments such as runtime,
   Lambda name, timeout and memory size are accepted by the from_handler_file() method as well, but they are not
   mandatory and have default values.
#. Create rh.function instance, and than send it over to AWS Lambdas. For example:
   :code:`rh.function(summer).to(system="aws_lambda")`

.. autoclass:: runhouse.LambdaFunction
   :members:
   :exclude-members: to

    .. automethod:: __init__


Lambda Hardware Setup
---------------------

To create an AWS Lambda, you must grant the necessary permissions to do so. They are provided by an IAM
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
    will not have the relevant IAM permissions for creating a Lambda, you will not be able to create a
    Lambda.
