Cluster
=======
A Cluster is a Runhouse primitive used for abstracting a particular hardware configuration.
This can be either an :ref:`on-demand cluster <OnDemandCluster Class>` (requires valid cloud credentials or a
local Kube config if launching on Kubernetes), or a
:ref:`BYO (bring-your-own) cluster <Cluster Class>` (requires IP address and ssh creds).

A cluster is assigned a name, through which it can be accessed and reused later on.

Cluster Factory Methods
~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: runhouse.cluster

.. autofunction:: runhouse.ondemand_cluster

Cluster Class
~~~~~~~~~~~~~

.. autoclass:: runhouse.Cluster
  :members:
  :exclude-members:

    .. automethod:: __init__


Cluster Hardware Setup
----------------------

No additional setup is required. You will just need to have the IP address for the cluster
and the path to SSH credentials ready to be used for the cluster initialization.


OnDemandCluster Class
~~~~~~~~~~~~~~~~~~~~~
A OnDemandCluster is a cluster that uses SkyPilot functionality underneath to handle
various cluster properties.

.. autoclass:: runhouse.OnDemandCluster
   :members:
   :exclude-members:

    .. automethod:: __init__

OnDemandCluster Hardware Setup
------------------------------

On-Demand clusters use SkyPilot to automatically spin up and down clusters on the cloud. You will
need to first set up cloud access on your local machine:

Run ``sky check`` to see which cloud providers are enabled, and how to set up cloud credentials for each of the
providers.

.. code-block::

    $ sky check

For a more in depth tutorial on setting up individual cloud credentials, you can refer to
`SkyPilot setup docs <https://skypilot.readthedocs.io/en/latest/getting-started/installation.html#cloud-account-setup>`__.


Specifying a VPC
^^^^^^^^^^^^^^^^
If you would like to launch an on-demand cluster within a specific VPC, you can specify its name in your local
``~/.sky/config.yaml`` in the following format:

.. code-block:: yaml

    <cloud-provider>:
      vpc: <vpc-name>

See the `SkyPilot docs <https://skypilot.readthedocs.io/en/latest/cloud-setup/cloud-permissions/aws.html#using-a-specific-vpc>`__
for more details on configuring a VPC.


Cluster Authentication & Verification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Runhouse provides a couple of options to manage the connection to the Runhouse API server running on a cluster.

Server Connection
-----------------

The below options can be specified with the ``server_connection_type`` parameter
when :ref:`initializing a cluster <Cluster Factory Methods>`. By default the Runhouse API server will
be started on the cluster on port :code:`32300`.

- ``ssh``: Connects to the cluster via an SSH tunnel, by default on port :code:`32300`.
- ``tls``: Connects to the cluster via HTTPS (by default on port :code:`443`) using either a provided certificate, or
  creating a new self-signed certificate just for this cluster. You must open the needed ports in the firewall, such
  as via the open_ports argument in the OnDemandCluster, or manually in the compute itself or cloud console.
- ``none``: Does not use any port forwarding or enforce any authentication. Connects to the cluster with HTTP by
  default on port :code:`80`. This is useful when connecting to a cluster within a VPC, or creating a tunnel manually
  on the side with custom settings.


.. note::

    The ``tls`` connection type is the most secure and is recommended for production use if you are not running inside
    of a VPC. However, be mindful that you must secure the cluster with authentication (see below) if you open it
    to the public internet.

Server Authentication
---------------------

If desired, Runhouse provides out-of-the-box authentication via users' Runhouse token (generated when
:ref:`logging in <Login/Logout>`) and set locally at: :code:`~/.rh/config.yaml`). This is crucial if the cluster
has ports open to the public internet, as would usually be the case when using the ``tls`` connection type. You may
also set up your own authentication manually inside of your own code, but you should likely still enable Runhouse
authentication to ensure that even your non-user-facing endpoints into the server are secured.

When :ref:`initializing a cluster <Cluster Factory Methods>`, you can set the :code:`den_auth` parameter to :code:`True`
to enable token authentication. Calls to the cluster server can then be made using an auth header with the
format: :code:`{"Authorization": "Bearer <cluster-token>"}`. The Runhouse Python library adds this header to its calls
automatically, so your users do not need to worry about it after logging into Runhouse.


.. note::

   Runhouse never uses your default Runhouse token for anything other than requests made to
   `Runhouse Den <https://www.run.house/dashboard>`__. Your token will never be exposed or shared with anyone else.


TLS Certificates
^^^^^^^^^^^^^^^^
Enabling TLS and `Runhouse Den Dashboard <https://www.run.house/dashboard>`__ Auth for the API server makes it incredibly
fast and easy to stand up a microservice with standard token authentication, allowing you to easily share Runhouse
resources with collaborators, teams, customers, etc.

Let's illustrate this with a simple example:

.. code-block:: python

    import runhouse as rh

    def concat(a: str, b: str):
        return a + b

    # Launch a cluster with TLS and Den Auth enabled
    cpu = rh.ondemand_cluster(instance_type="m5.xlarge",
                              provider="aws",
                              name="rh-cluster",
                              den_auth=True,
                              open_ports=[443],
                              server_connection_type="tls").up_if_not()

    # Remote function stub which lives on the cluster
    remote_func = rh.function(concat).to(cpu)

    # Save to Runhouse Den
    remote_func.save()

    # Give read access to the function to another user - this will allow them to call this service remotely
    # and view the function metadata in Runhouse Den
    remote_func.share("user1@gmail.com", access_level="read")

    # This other user (user1) can then call the function remotely from any python environment
    res = remote_func("run", "house")
    >> print(res)
    >> "runhouse"


We can also call the function via an HTTP request, making it easy for other users to call the function with
a Runhouse cluster token (Note: this assumes the user has been granted access to the function or
write access to the cluster):

.. code-block::

    $ curl -X GET "https://<DOMAIN>/concat/call?a=run&b=house"
    -H "Content-Type: application/json" -H "Authorization: Bearer <CLUSTER-TOKEN>"

Caddy
^^^^^
Runhouse gives you the option of using `Caddy <https://caddyserver.com/>`__ as a reverse proxy for the Runhouse API
server, which is a FastAPI app launched with `Uvicorn <https://www.uvicorn.org/>`__. Using Caddy provides you with a
safer and more conventional approach running the FastAPI app on a higher, non-privileged port (such as 32300, the
default Runhouse port) and then use Caddy as a reverse proxy to forward requests from the HTTP port (default: 80) or
the HTTPS port (default: 443).

Caddy also enables generating and auto-renewing self-signed certificates, making it easy to secure your cluster with
HTTPS right out of the box.

.. note::

    Caddy is enabled by default when you launch a cluster with the :code:`server_port` set to either 80 or 443.


**Generating Certs**

Runhouse offers two options for enabling TLS/SSL on a cluster with Caddy:

1. *Using existing certs*: provide the path to the cert and key files with the :code:`ssl_certfile` and
   :code:`ssl_keyfile` arguments. These certs will be used by Caddy as specified in the Caddyfile on the cluster.
   If no cert paths are provided and no domain is specified, Runhouse will issue
   self-signed certificates to use for the cluster. These certs will not be verified by a CA.
2. *Using Caddy to generate CA verified certs*: Provide the :code:`domain` argument. Caddy will then obtain
   certificates from Let's Encrypt on-demand when a client connects for the first time.


Using a Custom Domain
~~~~~~~~~~~~~~~~~~~~~
Runhouse supports using custom domains for deploying your apps and services. You can provide the domain ahead of time
before launching the cluster by specifying the :code:`domain` argument:

.. code-block:: python

     cluster = rh.cluster(name="rh-serving-cpu",
                          domain="<your domain>",
                          instance_type="m5.xlarge",
                          server_connection_type="tls",
                          open_ports=[443]).up_if_not()

.. note::

    After the cluster is launched, make sure to add the relevant A record to your domain's DNS settings to point this
    domain to the cluster's public IP address.

    You'll need to also ensure the relevant ports are open (ex: 443) in the security group settings of the cluster.
    Runhouse will also automatically set up a TLS certificate for the domain via
    `Caddy <https://caddyserver.com/docs/automatic-https>`__.

If you have an existing cluster, you can also configure a domain by including the IP and domain when
initializing the Runhouse cluster object:

.. code-block:: python

     cluster = rh.cluster(name="rh-serving-cpu",
                          ips=["<public IP>"],
                          domain="<your domain>",
                          server_connection_type="tls",
                          open_ports=[443]).up_if_not()

Now we can send modules or functions to our cluster and seamlessly create endpoints which we can then share
and call from anywhere.

Let's take a look at an example of how to deploy a simple
`LangChain RAG app <https://www.run.house/examples/langchain-rag-app-aws-ec2>`__.

Once the app has been created and sent to the cluster, we can call it via HTTP directly:

.. code-block:: python

    import requests

    resp = requests.get("https://<domain>/basic_rag_app/invoke?user_prompt=<prompt>")
    print(resp.json())


Or via cURL:

.. code-block::

    $ curl "https://<domain>/basic_rag_app/invoke?user_prompt=<prompt>"
