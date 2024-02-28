Secrets
=======

.. raw:: html

    <p><a href="https://colab.research.google.com/github/run-house/notebooks/blob/stable/docs/api_secrets.ipynb">
    <img height="20px" width="117px" src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a></p>

Secrets are a Runhouse resource that provides a simple interface for
handling your secrets, such as provider credentials and API keys, across
environments. With Runhouse APIs, easily

-  construct a secret object, based on local variables, files, or
   environment variables
-  save new secrets and reload existing ones
-  set secrets across clusters, environments, and functions

For a more detailed API documentation, you can refer to our `Runhouse
docs <https://www.run.house/docs/api/python/secrets>`__.

Constructing Secrets
--------------------

Secrets are constructed using the ``rh.secret`` or
``rh.provider_secret`` factory functions.

.. code:: ipython3

    import runhouse as rh

Base Secret
~~~~~~~~~~~

Base secrets are constructed with ``rh.secret``, and consist of a values
dictionary mapping secret keys to values.

.. code:: ipython3

    my_secret = rh.secret(name="my_secret", values={"key": "secret_val"})
    my_secret.save()
    del my_secret


.. parsed-literal::
    :class: code-output

    INFO | 2023-12-20 05:59:57.445868 | Saving config for ~/my_secret to: /Users/caroline/.rh/secrets/my_secret.json


.. code:: ipython3

    my_secret = rh.secret("my_secret")
    my_secret.values


.. parsed-literal::
    :class: code-output

    INFO | 2023-12-20 05:59:57.772553 | Loading config from local file /Users/caroline/.rh/secrets/my_secret.json




.. parsed-literal::
    :class: code-output

    {'key': 'secret_val'}



Provider Secret
~~~~~~~~~~~~~~~

Provider secrets are constructed with ``rh.provider_secret`` and are
associated with a provider type. These can be constructed by passing in
a ``values`` key-pair mapping, or by providing the local file, or local
environment variables associated with the keys, and as a result, have
additional functionality such as being able to write to a file or
environment variables. There are various supported builtin provider
types, such as cluster providers (aws, azure, …), api key based
providers (openai, anthropic, …), and ssh keys. These secret classes
have default locations (file path or environment variables) that
Runhouse will use to extract the secret values from out-of-the-box, if
the ``values`` are not explicitly provided.

.. code:: ipython3

    rh.Secret.builtin_providers(as_str=True)




.. parsed-literal::
    :class: code-output

    ['aws',
     'azure',
     'gcp',
     'github',
     'huggingface',
     'lambda',
     'ssh',
     'sky',
     'anthropic',
     'cohere',
     'langchain',
     'openai',
     'pinecone',
     'wandb']



Compute Providers
^^^^^^^^^^^^^^^^^

Here, we construct a default AWS provider secret. We locally have dummy
variables stored in the default path ~/.aws/credentials, and we see that
this is automatically set.

.. code:: ipython3

    !cat ~/.aws/credentials


.. parsed-literal::
    :class: code-output

    [default]
    aws_access_key_id = ABCD_KEY
    aws_secret_access_key = 1234_KEY


.. code:: ipython3

    # default provider secret for AWS. Will pull in values from expected default configuration when used.
    aws_secret = rh.provider_secret("aws")

    print(f"extracted path: {aws_secret.path}")
    print(f"extracted values: {aws_secret.values}")


.. parsed-literal::
    :class: code-output

    extracted path: ~/.aws/credentials
    extracted values: {'access_key': 'ABCD_KEY', 'secret_key': '1234_KEY'}


You can also instantiate secrets by directly passing in their secret
values (if it isn’t locally set up yet), and optionally save it down
locally.

.. code:: ipython3

    # provider secret constructed from values dictionary, for LambdaLabs.
    lambda_secret = rh.provider_secret("lambda", values={"api_key": "lambda_key"})

    print(f"values: {lambda_secret.values}")

    lambda_secret = lambda_secret.write()
    print(f"path: {lambda_secret.path}")


.. parsed-literal::
    :class: code-output

    INFO | 2023-12-20 17:37:57.775584 | Secrets already exist in ~/.lambda_cloud/lambda_keys.


.. parsed-literal::
    :class: code-output

    values: {'api_key': 'lambda_key'}
    path: ~/.lambda_cloud/lambda_keys


Or, you can construct a secret with a non-default path, and Runhouse
will extract out the values.

.. code:: ipython3

    !cat ~/.aws/credentials_custom


.. parsed-literal::
    :class: code-output

    [default]
    aws_access_key_id = ABCD_KEY_CUSTOM
    aws_secret_access_key = 1234_KEY_CUSTOM


.. code:: ipython3

    aws_secret_custom = rh.provider_secret("aws", path="~/.aws/credentials_custom")

    print(f"path: {aws_secret_custom.path}")
    print(f"values: {aws_secret_custom.values}")


.. parsed-literal::
    :class: code-output

    path: ~/.aws/credentials_custom
    values: {'access_key': 'ABCD_KEY_CUSTOM', 'secret_key': '1234_KEY_CUSTOM'}


API Keys
^^^^^^^^

These provider secrets consist of a single API key, associated with a
default environment variable key, often ``PROVIDER_API_KEY``. They can
be constructed by passing in a values dict mapping ``api_key`` to the
value, or the value will be inferred from the environment variables.
Calling ``.write()`` will set the environment variable in the current
process.

Secrets from inferred env value:

.. code:: ipython3

    import os
    os.environ["OPENAI_API_KEY"] = "openai_key"

.. code:: ipython3

    openai_secret = rh.provider_secret("openai")
    openai_secret.values




.. parsed-literal::
    :class: code-output

    {'api_key': 'openai_key'}



Passing in value to the constructor:

.. code:: ipython3

    anthropic_secret = rh.provider_secret("anthropic", values={"api_key": "ahthropic_key"})
    anthropic_secret.write()

    os.environ["ANTHROPIC_API_KEY"]




.. parsed-literal::
    :class: code-output

    'ahthropic_key'



SSH Keys
^^^^^^^^

SSH public and private key pairs are another type of supported builtin
provider type. Simply pass in ``provider="ssh"`` and ``name=<key>``.

.. code:: ipython3

    !cat ~/.ssh/example


.. parsed-literal::
    :class: code-output

    **private key values**


.. code:: ipython3

    !cat ~/.ssh/example.pub


.. parsed-literal::
    :class: code-output

    **public key values**


.. code:: ipython3

    ssh_secret = rh.provider_secret(provider="ssh", name="example")
    ssh_secret.values




.. parsed-literal::
    :class: code-output

    {'public_key': '**public key values**\n',
     'private_key': '**private key values**\n'}



Sending Secrets
---------------

You can directly send secrets to a cluster using the ``secret.to()``
API, bulk sync secrets using ``cluster.sync_secrets()``, or by including
them as part of a ``rh.env()``.

.. code:: ipython3

    cluster = rh.ondemand_cluster("example-cluster")

``secret.to(system, env)``
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    lambda_secret.path




.. parsed-literal::
    :class: code-output

    '~/.lambda_cloud/lambda_keys'



.. code:: ipython3

    # path secret
    lambda_secret.to(cluster)
    rh.file(path=lambda_secret.path, system=cluster).fetch(mode="r", deserialize=False)


.. parsed-literal::
    :class: code-output

    INFO | 2023-12-20 17:43:02.929930 | Connected (version 2.0, client OpenSSH_8.2p1)
    INFO | 2023-12-20 17:43:03.168593 | Authentication (publickey) successful!
    INFO | 2023-12-20 17:43:03.171218 | Connecting to server via SSH, port forwarding via port 32300.
    INFO | 2023-12-20 17:43:03.172050 | Checking server example-cluster
    INFO | 2023-12-20 17:43:03.677092 | Server example-cluster is up.
    INFO | 2023-12-20 17:43:03.820783 | Calling lambda._write_to_file


.. parsed-literal::
    :class: code-output

    base servlet: Calling method _write_to_file on module lambda


.. parsed-literal::
    :class: code-output

    INFO | 2023-12-20 17:43:04.110755 | Time to call lambda._write_to_file: 0.29 seconds
    INFO | 2023-12-20 17:43:04.523570 | Getting file_20231220_124304
    INFO | 2023-12-20 17:43:04.633602 | Time to get file_20231220_124304: 0.11 seconds



.. parsed-literal::
    :class: code-output

    'api_key = lambda_key\n'



.. code:: ipython3

    env = rh.env()
    openai_secret.to(cluster, env=env)


.. parsed-literal::
    :class: code-output

    INFO | 2023-12-20 17:43:11.602308 | Getting base_env
    INFO | 2023-12-20 17:43:13.070980 | Calling base_env._set_env_vars


.. parsed-literal::
    :class: code-output

    base_env servlet: Calling method _set_env_vars on module base_env


.. parsed-literal::
    :class: code-output

    INFO | 2023-12-20 17:43:13.432078 | Time to call base_env._set_env_vars: 0.36 seconds




.. parsed-literal::
    :class: code-output

    <runhouse.resources.secrets.provider_secrets.openai_secret.OpenAISecret at 0x17fe7f1f0>



.. code:: ipython3

    def _get_env_var(var):
        import os
        return os.environ[var]

    get_env_var = rh.function(_get_env_var, system=cluster, env=env)
    get_env_var("OPENAI_API_KEY")


.. parsed-literal::
    :class: code-output

    INFO | 2023-12-20 17:43:16.529605 | Writing out function to /Users/caroline/Documents/runhouse/runhouse/docs/notebooks/api/_get_env_var_fn.py. Please make sure the function does not rely on any local variables, including imports (which should be moved inside the function body).
    INFO | 2023-12-20 17:43:16.540215 | Setting up Function on cluster.
    INFO | 2023-12-20 17:43:16.543037 | Copying package from file:///Users/caroline/Documents/runhouse/runhouse to: example-cluster
    INFO | 2023-12-20 17:43:16.544655 | Running command: ssh -T -i ~/.ssh/sky-key -o Port=22 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o IdentitiesOnly=yes -o ExitOnForwardFailure=yes -o ServerAliveInterval=5 -o ServerAliveCountMax=3 -o ConnectTimeout=30s -o ForwardAgent=yes -o ControlMaster=auto -o ControlPath=/tmp/skypilot_ssh_caroline/41014bb4d3/%C -o ControlPersist=300s ubuntu@44.201.245.202 'bash --login -c -i '"'"'true && source ~/.bashrc && export OMP_NUM_THREADS=1 PYTHONWARNINGS=ignore && (mkdir -p ~/runhouse/)'"'"' 2>&1'
    INFO | 2023-12-20 17:43:22.149570 | Calling base_env.install


.. parsed-literal::
    :class: code-output

    base_env servlet: Calling method install on module base_env
    Installing package: Package: runhouse
    Installing Package: runhouse with method reqs.
    reqs path: runhouse/requirements.txt
    pip installing requirements from runhouse/requirements.txt with: -r runhouse/requirements.txt
    Running: /opt/conda/bin/python3.10 -m pip install -r runhouse/requirements.txt


.. parsed-literal::
    :class: code-output

    INFO | 2023-12-20 17:43:26.164394 | Time to call base_env.install: 4.01 seconds
    INFO | 2023-12-20 17:43:26.420370 | Function setup complete.
    INFO | 2023-12-20 17:43:26.427886 | Calling _get_env_var.call


.. parsed-literal::
    :class: code-output

    base_env servlet: Calling method call on module _get_env_var


.. parsed-literal::
    :class: code-output

    INFO | 2023-12-20 17:43:26.686193 | Time to call _get_env_var.call: 0.26 seconds




.. parsed-literal::
    :class: code-output

    'openai_key'



cluster.sync_secrets()
^^^^^^^^^^^^^^^^^^^^^^

You can pass in a list of secrets along with an env into
``cluster.sync_secrets`` to be synced over from local to a cluster. The
list can consist of secrets resources or the string corresponding to the
provider/name.

.. code:: ipython3

    cluster.sync_secrets(["aws", "gcp", openai_secret])


.. parsed-literal::
    :class: code-output

    INFO | 2023-12-20 17:43:32.041330 | Calling aws._write_to_file


.. parsed-literal::
    :class: code-output

    base servlet: Calling method _write_to_file on module aws
    Secrets already exist in ~/.aws/credentials.


.. parsed-literal::
    :class: code-output

    INFO | 2023-12-20 17:43:32.293934 | Time to call aws._write_to_file: 0.25 seconds
    INFO | 2023-12-20 17:43:32.676000 | Calling gcp._write_to_file


.. parsed-literal::
    :class: code-output

    base servlet: Calling method _write_to_file on module gcp
    Secrets already exist in ~/.config/gcloud/application_default_credentials.json.


.. parsed-literal::
    :class: code-output

    INFO | 2023-12-20 17:43:32.936027 | Time to call gcp._write_to_file: 0.26 seconds
    INFO | 2023-12-20 17:43:33.315656 | Getting None
    INFO | 2023-12-20 17:43:33.551578 | Calling env_20231220_174256._set_env_vars


.. parsed-literal::
    :class: code-output

    base servlet: Calling method _set_env_vars on module env_20231220_174256


.. parsed-literal::
    :class: code-output

    INFO | 2023-12-20 17:43:33.790339 | Time to call env_20231220_174256._set_env_vars: 0.24 seconds


rh.env
^^^^^^

You can also include a list of secrets in a Runhouse env object. When
the env is then sent to a cluster, as part of a function or directly,
the secrets will be synced onto the environment as well, and accessible
from function and system calls running in the environment.

.. code:: ipython3

    secrets_env = rh.env(secrets=["aws", openai_secret])

    get_env_var = rh.function(_get_env_var, system=cluster, env=secrets_env)
    get_env_var("OPENAI_API_KEY")


.. parsed-literal::
    :class: code-output

    INFO | 2023-12-20 17:48:29.631094 | Writing out function to /Users/caroline/Documents/runhouse/runhouse/docs/notebooks/api/_get_env_var_fn.py. Please make sure the function does not rely on any local variables, including imports (which should be moved inside the function body).
    INFO | 2023-12-20 17:48:29.662722 | Setting up Function on cluster.
    INFO | 2023-12-20 17:48:29.664560 | Copying package from file:///Users/caroline/Documents/runhouse/runhouse to: example-cluster
    INFO | 2023-12-20 17:48:29.665912 | Running command: ssh -T -i ~/.ssh/sky-key -o Port=22 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o IdentitiesOnly=yes -o ExitOnForwardFailure=yes -o ServerAliveInterval=5 -o ServerAliveCountMax=3 -o ConnectTimeout=30s -o ForwardAgent=yes -o ControlMaster=auto -o ControlPath=/tmp/skypilot_ssh_caroline/41014bb4d3/%C -o ControlPersist=300s ubuntu@44.201.245.202 'bash --login -c -i '"'"'true && source ~/.bashrc && export OMP_NUM_THREADS=1 PYTHONWARNINGS=ignore && (mkdir -p ~/runhouse/)'"'"' 2>&1'
    INFO | 2023-12-20 17:48:31.562203 | Calling aws._write_to_file


.. parsed-literal::
    :class: code-output

    base servlet: Calling method _write_to_file on module aws
    Secrets already exist in ~/.aws/credentials.


.. parsed-literal::
    :class: code-output

    INFO | 2023-12-20 17:48:31.783056 | Time to call aws._write_to_file: 0.22 seconds
    INFO | 2023-12-20 17:48:32.119017 | Getting base_env
    INFO | 2023-12-20 17:48:32.227256 | Time to get base_env: 0.11 seconds
    INFO | 2023-12-20 17:48:32.229612 | Calling base_env._set_env_vars


.. parsed-literal::
    :class: code-output

    base_env servlet: Calling method _set_env_vars on module base_env


.. parsed-literal::
    :class: code-output

    INFO | 2023-12-20 17:48:32.474858 | Time to call base_env._set_env_vars: 0.25 seconds
    INFO | 2023-12-20 17:48:32.678774 | Calling base_env.install


.. parsed-literal::
    :class: code-output

    base_env servlet: Calling method install on module base_env
    Env already installed, skipping


.. parsed-literal::
    :class: code-output

    INFO | 2023-12-20 17:48:32.917056 | Time to call base_env.install: 0.24 seconds
    INFO | 2023-12-20 17:48:33.114583 | Function setup complete.
    INFO | 2023-12-20 17:48:33.122322 | Calling _get_env_var.call


.. parsed-literal::
    :class: code-output

    base_env servlet: Calling method call on module _get_env_var


.. parsed-literal::
    :class: code-output

    INFO | 2023-12-20 17:48:33.352872 | Time to call _get_env_var.call: 0.23 seconds




.. parsed-literal::
    :class: code-output

    'openai_key'



Saving and Loading Secrets
--------------------------

You can save a secret using the ``.save()`` API, and reload a saved
secret by calling ``rh.secret(<name>)``.

If you are not logged in to your Runhouse account, the secret config
will be saved locally.

If you have a Runhouse account, which you can create
`here <run.house/login>`__ or by calling either the ``runhouse login``
CLI command or ``rh.login()`` Python command, calling ``.save()`` will
save the resource metadata on Runhouse servers, and the secret values to
Hashicorp Vault.

Local Secret
~~~~~~~~~~~~

.. code:: ipython3

    local_secret = rh.provider_secret(provider="lambda", name="lambda_key")
    local_secret.save()
    del local_secret


.. parsed-literal::
    :class: code-output

    INFO | 2023-12-20 06:03:31.257864 | Saving config for ~/lambda_key to: /Users/caroline/.rh/secrets/lambda_key.json


.. code:: ipython3

    reloaded_secret = rh.provider_secret("lambda_key")
    reloaded_secret.values


.. parsed-literal::
    :class: code-output

    INFO | 2023-12-20 06:03:46.371170 | Loading config from local file /Users/caroline/.rh/secrets/lambda_key.json




.. parsed-literal::
    :class: code-output

    {'api_key': 'lambda_key'}



Den Secret
~~~~~~~~~~

If you have a Runhouse account, which you can create
`here <run.house/login>`__ or by calling either the ``runhouse login``
CLI command or ``rh.login()`` Python command, you can save secret to
your dashboard. The metadata for the Secret resource, such as the
provider, any path or env vars, etc, will be saved into Runhouse Den
servers, while the secrets values themselves will be stored securely in
Hashicorp Vault.

.. code:: ipython3

    runhouse login()

.. code:: ipython3

    rh.provider_secret("gcp", name="gcp_secret").save()


.. parsed-literal::
    :class: code-output

    INFO | 2023-12-20 06:09:17.712653 | Saving config for gcp_secret to Den
    INFO | 2023-12-20 06:09:18.184111 | Saving secrets for gcp_secret to Vault




.. parsed-literal::
    :class: code-output

    <runhouse.resources.secrets.provider_secrets.gcp_secret.GCPSecret at 0x1650c7fd0>



.. code:: ipython3

    rh.provider_secret("gcp_secret")




.. parsed-literal::
    :class: code-output

    <runhouse.resources.secrets.provider_secrets.provider_secret.ProviderSecret at 0x10545efd0>



Login and Logout
~~~~~~~~~~~~~~~~

The login flow gives you the option to upload locally detected builtin
provider secrets, or load down saved-down Vault secrets into your local
environment. If loading down new secrets, the location (file or env var)
of the new secrets will be logged in your runhouse config yaml at
``~/.rh/config.yaml`` as well. There are some useful APIs as well for
seeing which secrets you have locally configured or stored in Vault.

.. code:: ipython3

    runhouse.login()

.. code:: ipython3

    # list of my locally configured secrets
    locally_configued_secrets = rh.Secret.extract_provider_secrets()
    locally_configued_secrets




.. parsed-literal::
    :class: code-output

    {'aws': <runhouse.resources.secrets.provider_secrets.aws_secret.AWSSecret at 0x1631ccd90>,
     'gcp': <runhouse.resources.secrets.provider_secrets.gcp_secret.GCPSecret at 0x105f16fd0>,
     'github': <runhouse.resources.secrets.provider_secrets.github_secret.GitHubSecret at 0x1631c3d60>,
     'lambda': <runhouse.resources.secrets.provider_secrets.lambda_secret.LambdaSecret at 0x1631c3ac0>,
     'sky': <runhouse.resources.secrets.provider_secrets.sky_secret.SkySecret at 0x1631c3850>,
     'ssh-sagemaker-ssh-gw': <runhouse.resources.secrets.provider_secrets.ssh_secret.SSHSecret at 0x105f495e0>,
     'ssh-id_rsa': <runhouse.resources.secrets.provider_secrets.ssh_secret.SSHSecret at 0x105f49190>,
     'ssh-id_rsa_tmp': <runhouse.resources.secrets.provider_secrets.ssh_secret.SSHSecret at 0x105f492e0>}



.. code:: ipython3

    # if previously logged in and saved secrets to vault, can load down the secrets
    vault_secrets = rh.Secret.vault_secrets()
    vault_secrets




.. parsed-literal::
    :class: code-output

    ['aws', 'gcp', 'github', 'huggingface', 'lambda', 'ssh-id_rsa']



To save a secret to Vault, simply call ``.save()`` on the resource. This
will save both the values themselves, and relevant metadata such as the
path where it is locally stored.

You can manually construct and save a resource, or iterate through one
of the lists above.

.. code:: ipython3

    aws_secret_custom.save()
    locally_configued_secrets["gcp"].save()


.. parsed-literal::
    :class: code-output

    INFO | 2023-12-11 17:50:58.715913 | Saving config for aws to Den
    INFO | 2023-12-11 17:50:58.748314 | Saving secrets for aws to Vault
    INFO | 2023-12-11 17:50:59.565812 | Saving config for gcp to Den
    INFO | 2023-12-11 17:50:59.597261 | Saving secrets for gcp to Vault


Logout will prompt you one by one the secrets that have been saved
locally, whether or not you’d like to remove the associated file or env
vars.

.. code:: ipython3

    rh.logout()


.. parsed-literal::
    :class: code-output

    Delete credentials in ['ANTHROPIC_API_KEY'] for anthropic? [y/N]:


.. parsed-literal::
    :class: code-output

    Delete your local Runhouse config file? [y/N]:


.. parsed-literal::
    :class: code-output

    INFO | 2023-12-20 17:56:54.337915 | Successfully logged out of Runhouse.
