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
