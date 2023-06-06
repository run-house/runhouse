Resource and Secrets Management
===============================

Saving Secrets
~~~~~~~~~~~~~~
There are a few ways to save secrets to Runhouse to make them available conveniently across environments.

If your secrets are saved into your local environment (e.g. :code:`~/.aws/...`), the fastest way to save them is to run
:code:`runhouse login` in your command line (or :code:`runhouse.login()` in a Python interpreter), which will prompt
you for your Runhouse token and ask if you'd like to upload secrets. It will then extract secrets from your environment
and upload them to Vault. Alternatively, you can run:


.. code-block:: python

    import runhouse as rh
    rh.Secrets.extract_and_upload()

To add locally stored secrets for a specific provider (AWS, Azure, GCP):

.. code-block:: python

    rh.Secrets.put(provider="aws")


To add secrets for a custom provider or those not stored in local config files, use:

.. code-block:: python

    rh.Secrets.put(provider="snowflake", secret={"private_key": "************"})



These will not be automatically loaded into new environments via :code:`runhouse login`, but can be accessed in code via
:code:`rh.Secrets.get(provider="snowflake")`.


Loading Secrets
~~~~~~~~~~~~~~~

To load secrets into your environment, you can run :code:`runhouse login` or :code:`rh.login()` in your command line or Python
interpreter. This will prompt you for your Runhouse token and download secrets from Vault. Alternatively, you can run:

.. code-block:: python

    import runhouse as rh
    rh.Secrets.download_into_env()
    # or
    rh.Secrets.download_into_env(providers=["aws", "azure"])


To get secrets for a specific provider:

.. code-block:: python

    my_creds = rh.Secrets.get(provider="aws")


Deleting Secrets
~~~~~~~~~~~~~~~~
To delete secrets from Vault for a list of providers:

.. code-block:: python

    rh.Secrets.delete(providers=["huggingface"])


You can also delete your secrets stored in Vault directly from the
Management UI `Account & Secrets page <https://api.run.house/dashboard/?option=account/>`_.
