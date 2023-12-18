Secrets
=======
Runhouse provides a convenient interface for managing your secrets in a secure manner.
Secrets are stored in `Vault <https://www.vaultproject.io/>`_, and never on Runhouse servers.

See the Accessibility API tutorial for more details on using the Secrets API.

Secrets Factory Methods
~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: runhouse.secret

.. autofunction:: runhouse.provider_secret


Secret Class
~~~~~~~~~~~~

.. autoclass:: runhouse.Secret
   :members:
   :exclude-members:

    .. automethod:: __init__

ProviderSecret Class
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: runhouse.ProviderSecret
   :members:
   :exclude-members:

    .. automethod:: __init__


AWSSecret Class
---------------
.. autoclass:: runhouse.resources.secrets.provider_secrets.aws_secret.AWSSecret
   :show-inheritance:

   .. autoattribute:: _PROVIDER
   .. autoattribute:: _DEFAULT_CREDENTIALS_PATH
   .. autoattribute:: _DEFAULT_ENV_VARS

AzureSecret Class
---------------
.. autoclass:: runhouse.resources.secrets.provider_secrets.azure_secret.AzureSecret
   :show-inheritance:

   .. autoattribute:: _PROVIDER
   .. autoattribute:: _DEFAULT_CREDENTIALS_PATH
   .. autoattribute:: _DEFAULT_ENV_VARS

GCPSecret Class
---------------
.. autoclass:: runhouse.resources.secrets.provider_secrets.gcp_secret.GCPSecret
   :show-inheritance:

   .. autoattribute:: _PROVIDER
   .. autoattribute:: _DEFAULT_CREDENTIALS_PATH
   .. autoattribute:: _DEFAULT_ENV_VARS

GitHubSecret Class
---------------
.. autoclass:: runhouse.resources.secrets.provider_secrets.github_secret.GitHubSecret
   :show-inheritance:

   .. autoattribute:: _PROVIDER
   .. autoattribute:: _DEFAULT_CREDENTIALS_PATH

HuggingFaceSecret Class
---------------
.. autoclass:: runhouse.resources.secrets.provider_secrets.huggingface_secret.HuggingFaceSecret
   :show-inheritance:

   .. autoattribute:: _PROVIDER
   .. autoattribute:: _DEFAULT_CREDENTIALS_PATH


LambdaSecret Class
---------------
.. autoclass:: runhouse.resources.secrets.provider_secrets.lambda_secret.LambdaSecret
   :show-inheritance:

   .. autoattribute:: _PROVIDER
   .. autoattribute:: _DEFAULT_CREDENTIALS_PATH


SSHSecret Class
---------------
.. autoclass:: runhouse.resources.secrets.provider_secrets.ssh_secret.SSHSecret
   :show-inheritance:

   .. autoattribute:: _PROVIDER
   .. autoattribute:: _DEFAULT_CREDENTIALS_PATH
   .. autoattribute:: _DEFAULT_KEY


SkySecret Class
---------------
.. autoclass:: runhouse.resources.secrets.provider_secrets.sky_secret.SkySecret
   :show-inheritance:

   .. autoattribute:: _PROVIDER
   .. autoattribute:: _DEFAULT_CREDENTIALS_PATH
   .. autoattribute:: _DEFAULT_KEY
