Manual Setup and Troubleshooting
================================
Below we document steps for manually setting up SSH tunnel and syncing files/packages,
as well as some commonly encountered errors and solutions. Please create an issue on
`Github <https://github.com/run-house/runhouse/issues>`_ or message us on
`Discord <https://discord.gg/RnhB6589Hs>`_ for more support.

Manual SSH Tunneling
--------------------
If you are having trouble forming successful SSH connections with the Runhouse pip package,
want more control over SSH tunneling, or want to use latest unreleased features from main,
you may perform the following manual setup steps.

1. Install Runhouse on your local box with:

   .. code::

        pip install runhouse  # latest
        # or
        pip install git+https://github.com/run-house/runhouse.git  # main branch

2. SSH into your server with a tunnel. You can use a different local or remote port here as well;
   steps 5 and 6 contain additional info for setting up your port.

   .. code::

        ssh user@host -L 32300:localhost:32300

3. Install Runhouse on the remote box using the same command as in step 1:

   .. code::

        pip install runhouse  # latest
        # or
        pip install git+https://github.com/run-house/runhouse.git  # main branch

4. Start Ray on the remote box:

   .. code::

        ray start --head

   * If you receive an error ``“WARNING: The scripts ray, rllib, serve and tune are installed
     in '/home/user/.local/bin' which is not on PATH.”``, you may need to add the destination of the installed packages
     to the path, e.g. ``PATH=$PATH:/home/user/.local/bin``.

5. Start the Runhouse server on the remote box:

   .. code::

        runhouse start

   If you previously already had a version of Runhouse installed and started, you will need to run:

   .. code::

        runhouse restart

   * To exit out of the process and run it in the background instead, you can do ``Ctrl+C`` and then ``runhouse start --screen``.
     Please first make sure regular ``runhouse start`` does not error out prior to testing it out in a screen.

   * If you previously specified a custom port in step 2, add ``--port <my_port>`` to the runhouse start command.

6. Initialize the Runhouse cluster in Python as follows. Make sure to use ``localhost`` as the cluster address rather than the true IP,
   as you manually created the tunnel above.

   .. code::

        gpu = rh.cluster(name='rh-a10x', host=["localhost"], ssh_creds={"ssh_user": "user"})

   * To specify another port, you can pass in ``host=["localhost:<my_port>"]``

   * If your remote box already has a port open to http traffic, you can directly use it instead of forming a new tunnel.
     Pass in ``host=["<my ip>:<my port>"]``.


Manual File and Package Syncing
-------------------------------
If you are running into an rsync or module not found error, you can work around by manually syncing over local files or
installing any packages on your remote box.

1. To sync over the local folder (e.g. ``tutorials``) to remote. Not the trailing slash ``/`` at the end of the local directory
   to prevent an additional level of folder nesting on the remote cluster. Now is also a good time to install any packages on the remote box,
   including ``requirements.txt``.

   .. code::

      rsync -a tutorials/ user@216.153.62.74:tutorials

   To ensure Runhouse doesn't try to resync over your code again, pass in an empty env object, ``env=rh.Env()``,
   when sending functions or modules to the cluster.

   .. code::

      generate_gpu = rh.function(fn=sd_generate).to(gpu, env=rh.Env())


2. If you're encountering a Module not found error, you may need to pass a Runhouse package object pointing to the
   package in the remote filesystem into your Env so the server can find the module properly. You can do that like so:

   .. code::

        my_env = rh.Env(workdir=rh.Package(path="tutorials", system=gpu, install_method="reqs"))
        generate_gpu = rh.function(fn=sd_generate).to(gpu, env=my_env)

   * ``install_method`` can be ``local`` (to skip install), ``pip``, ``conda``, or ``reqs``
     (pip installs requirements.txt)

Common Errors
-------------

Q: I'm running into the following error: ``ImportError: Failed to import grpc on Apple Silicon.``

   Please run ``pip uninstall grpcio; conda install grpcio``, then rerun your code. This is due to the Ray/grpcio
   compatibility on Apple Silicon during setup.

Q: I'm running into an OpenSSL error.

   Please run ``pip install pip --upgrade; pip install pyopenssl --upgrade``.

Q: I'm running into an rsync error.

   Rsync is finicky and we are working to support more reliable file syncing. In the meantime, please refer to the
   Manual Package syncing instructions above. It is also worth noting that rsync will throw an error if ``.bashrc``
   outputs anything to the terminal. See `Issue <https://github.com/run-house/runhouse/issues/91>`_.
