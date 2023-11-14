import contextlib
import copy
import logging
import os
import pkgutil
import re
import subprocess
import sys
import threading
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from runhouse.servers.http.certs import TLSCertConfig

# Filter out DeprecationWarnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import requests.exceptions
import sshtunnel

from sshtunnel import HandlerSSHTunnelForwarderError, SSHTunnelForwarder

from runhouse.globals import obj_store, open_cluster_tunnels, rns_client
from runhouse.resources.envs.utils import _get_env_from
from runhouse.resources.hardware.utils import (
    _current_cluster,
    ServerConnectionType,
    SkySSHRunner,
    SshMode,
)
from runhouse.resources.resource import Resource

from runhouse.servers.http import HTTPClient

logger = logging.getLogger(__name__)


class Cluster(Resource):
    RESOURCE_TYPE = "cluster"
    REQUEST_TIMEOUT = 5  # seconds

    DEFAULT_SERVER_PORT = 32300
    DEFAULT_HTTP_PORT = 80
    DEFAULT_HTTPS_PORT = 443
    LOCALHOST = "127.0.0.1"
    LOCAL_HOSTS = ["localhost", LOCALHOST]

    SERVER_LOGFILE = os.path.expanduser("~/.rh/server.log")
    CLI_RESTART_CMD = "runhouse restart"
    SERVER_START_CMD = f"{sys.executable} -m runhouse.servers.http.http_server"
    SERVER_STOP_CMD = f'pkill -f "{SERVER_START_CMD}"'
    # 2>&1 redirects stderr to stdout
    START_SCREEN_CMD = (
        f"screen -dm bash -c \"{SERVER_START_CMD} |& tee -a '{SERVER_LOGFILE}' 2>&1\""
    )
    RAY_START_CMD = "ray start --head --port 6379"
    # RAY_BOOTSTRAP_FILE = "~/ray_bootstrap_config.yaml"
    # --autoscaling-config=~/ray_bootstrap_config.yaml
    # We need to use this instead of ray stop to make sure we don't stop the SkyPilot ray server,
    # which runs on other ports but is required to preserve autostop and correct cluster status.
    RAY_KILL_CMD = 'pkill -f ".*ray.*6379.*"'

    def __init__(
        self,
        name,
        ips: List[str] = None,
        ssh_creds: Dict = None,
        server_host: str = None,
        server_port: int = None,
        server_connection_type: str = None,
        ssl_keyfile: str = None,
        ssl_certfile: str = None,
        den_auth: bool = False,
        enable_local_span_collection: bool = False,
        dryrun=False,
        **kwargs,  # We have this here to ignore extra arguments when calling from from_config
    ):
        """
        The Runhouse cluster, or system. This is where you can run Functions or access/transfer data
        between. You can BYO (bring-your-own) cluster by providing cluster IP and ssh_creds, or
        this can be an on-demand cluster that is spun up/down through
        `SkyPilot <https://github.com/skypilot-org/skypilot>`_, using your cloud credentials.

        .. note::
            To build a cluster, please use the factory method :func:`cluster`.
        """

        super().__init__(name=name, dryrun=dryrun)

        self.address = ips[0] if isinstance(ips, List) else ips
        self._ssh_creds = ssh_creds
        self.ips = ips
        self._rpc_tunnel = None
        self._rh_version = None

        self.client = None
        self.den_auth = den_auth
        self.enable_local_span_collection = enable_local_span_collection
        self.cert_config = TLSCertConfig(
            cert_path=ssl_certfile, key_path=ssl_keyfile, dir_name=self.name
        )

        self.server_connection_type = server_connection_type
        self.server_port = server_port or self.DEFAULT_SERVER_PORT
        self.server_host = server_host

    def save_config_to_cluster(self):
        import json

        config = self.config_for_rns
        if "live_state" in config.keys():
            # a bunch of setup commands that mess up json dump
            del config["live_state"]
        json_config = f"{json.dumps(config)}"

        self.run(
            [
                f"mkdir -p ~/.rh; touch ~/.rh/cluster_config.yaml; echo '{json_config}' > ~/.rh/cluster_config.yaml"
            ]
        )

    @staticmethod
    def from_config(config: dict, dryrun=False):
        resource_subtype = config.get("resource_subtype")
        if resource_subtype == "Cluster":
            return Cluster(**config, dryrun=dryrun)
        elif resource_subtype == "OnDemandCluster":
            from .on_demand_cluster import OnDemandCluster

            return OnDemandCluster(**config, dryrun=dryrun)
        elif resource_subtype == "SageMakerCluster":
            from .sagemaker_cluster import SageMakerCluster

            return SageMakerCluster(**config, dryrun=dryrun)
        else:
            raise ValueError(f"Unknown cluster type {resource_subtype}")

    @property
    def config_for_rns(self):
        config = super().config_for_rns
        self.save_attrs_to_config(
            config,
            [
                "ips",
                "server_port",
                "server_host",
                "server_connection_type",
                "den_auth",
                "enable_local_span_collection",
            ],
        )
        if self.ips is not None:
            config["ssh_creds"] = self.ssh_creds()
        else:
            config["ips"] = [self.address]

        if self._use_custom_cert:
            config["ssl_certfile"] = self.cert_config.cert_path

        if self._use_custom_key:
            config["ssl_keyfile"] = self.cert_config.key_path

        return config

    @property
    def server_address(self):
        """Address to use in the requests made to the cluster. If creating an SSH tunnel with the cluster,
        ths will be set to localhost, otherwise will use the cluster's public IP address."""
        if self.server_host in [self.LOCALHOST, "localhost"]:
            return self.LOCALHOST
        return self.address

    def is_up(self) -> bool:
        """Check if the cluster is up.

        Example:
            >>> rh.cluster("rh-cpu").is_up()
        """
        return self.address is not None

    def up_if_not(self):
        """Bring up the cluster if it is not up. No-op if cluster is already up.
        This only applies to on-demand clusters, and has no effect on self-managed clusters.

        Example:
            >>> rh.cluster("rh-cpu").up_if_not()
        """
        if not self.is_up():
            if not hasattr(self, "up"):
                raise NotImplementedError(
                    f"Cluster <{self.name}> does not have an up method."
                )
            self.up()
        return self

    def up(self):
        return self

    def keep_warm(self):
        logger.info(
            f"cluster.keep_warm will have no effect on self-managed cluster {self.name}."
        )
        return self

    def _sync_runhouse_to_cluster(self, _install_url=None, env=None):
        if not self.address:
            raise ValueError(f"No address set for cluster <{self.name}>. Is it up?")

        local_rh_package_path = Path(pkgutil.get_loader("runhouse").path).parent

        # Check if runhouse is installed from source and has setup.py
        if (
            not _install_url
            and local_rh_package_path.parent.name == "runhouse"
            and (local_rh_package_path.parent / "setup.py").exists()
        ):
            # Package is installed in editable mode
            local_rh_package_path = local_rh_package_path.parent
            dest_path = f"~/{local_rh_package_path.name}"

            self._rsync(
                source=str(local_rh_package_path),
                dest=dest_path,
                up=True,
                contents=True,
                filter_options="- docs/",
            )
            rh_install_cmd = "python3 -m pip install ./runhouse"
        # elif local_rh_package_path.parent.name == 'site-packages':
        else:
            # Package is installed in site-packages
            # status_codes = self.run(['pip install runhouse-nightly==0.0.2.20221202'], stream_logs=True)
            # rh_package = 'runhouse_nightly-0.0.1.dev20221202-py3-none-any.whl'
            # rh_download_cmd = f'curl https://runhouse-package.s3.amazonaws.com/{rh_package} --output {rh_package}'
            if not _install_url:
                import runhouse

                _install_url = f"runhouse=={runhouse.__version__}"
            rh_install_cmd = f"python3 -m pip install {_install_url}"

        install_cmd = f"{env._run_cmd} {rh_install_cmd}" if env else rh_install_cmd

        status_codes = self.run([install_cmd], stream_logs=True)

        if status_codes[0][0] != 0:
            raise ValueError(f"Error installing runhouse on cluster <{self.name}>")

    def install_packages(
        self, reqs: List[Union["Package", str]], env: Union["Env", str] = None
    ):
        """Install the given packages on the cluster.

        Args:
            reqs (List[Package or str): List of packages to install on cluster and env
            env (Env or str): Environment to install package on. If left empty, defaults to base environment.
                (Default: ``None``)

        Example:
            >>> cluster.install_packages(reqs=["accelerate", "diffusers"])
            >>> cluster.install_packages(reqs=["accelerate", "diffusers"], env="my_conda_env")
        """
        from runhouse.resources.envs.env import Env

        self.check_server()
        env = _get_env_from(env) or Env(name=env or Env.DEFAULT_NAME)
        env.reqs = env._reqs + reqs
        env.to(self)

    def get(
        self, key: str, default: Any = None, remote=False, stream_logs: bool = False
    ):
        """Get the result for a given key from the cluster's object store. To raise an error if the key is not found,
        use `cluster.get(key, default=KeyError)`."""
        self.check_server()
        if self.on_this_cluster():
            return obj_store.get(key, default=default)
        try:
            res = self.client.call_module_method(
                key,
                None,
                remote=remote,
                stream_logs=stream_logs,
                system=self,
            )
        except KeyError as e:
            if default == KeyError:
                raise e
            return default
        return res

    # TODO deprecate
    def get_run(self, run_name: str, folder_path: str = None):
        self.check_server()
        return self.get(run_name, remote=True).provenance

    # TODO this doesn't need to be a dedicated rpc, it can just flow through Secrets.to and put_resource,
    #  like installing packages. Also, it should accept an env (for env var secrets and docker envs).
    def add_secrets(self, provider_secrets: dict):
        """Copy secrets from current environment onto the cluster"""
        self.check_server()
        return self.client.add_secrets(provider_secrets)

    def put(self, key: str, obj: Any, env=None):
        """Put the given object on the cluster's object store at the given key."""
        self.check_server()
        if self.on_this_cluster():
            return obj_store.put(key, obj, env=env)
        return self.client.put_object(key, obj, env=env)

    def put_resource(self, resource: Resource, state=None, dryrun=False):
        """Put the given resource on the cluster's object store. Returns the key (important if name is not set)."""
        self.check_server()
        env = (
            resource.env
            if hasattr(resource, "env")
            else resource.name or resource.env_name
            if resource.RESOURCE_TYPE == "env"
            else None
        )
        if self.on_this_cluster():
            return obj_store.put(key=resource.name, value=resource, env=env)
        return self.client.put_resource(
            resource, state=state or {}, env=env, dryrun=dryrun
        )

    def rename(self, old_key: str, new_key: str):
        """Rename a key in the cluster's object store."""
        self.check_server()
        if self.on_this_cluster():
            return obj_store.rename(old_key, new_key)
        return self.client.rename_object(old_key, new_key)

    def keys(self, env=None):
        """List all keys in the cluster's object store."""
        self.check_server()
        if self.on_this_cluster():
            return obj_store.keys()
        res = self.client.keys(env=env)
        return res

    def cancel(self, key: str, force=False):
        """Cancel a given run on cluster by its key."""
        self.check_server()
        if self.on_this_cluster():
            return obj_store.cancel(key, force=force)
        return self.client.cancel(key, force=force)

    def cancel_all(self, force=False):
        """Cancel all runs on cluster."""
        self.check_server()
        if self.on_this_cluster():
            return obj_store.cancel_all(force=force)
        return self.client.cancel("all", force=force)

    def delete(self, keys: Union[None, str, List[str]]):
        """Delete the given items from the cluster's object store. To delete all items, use `cluster.clear()`"""
        self.check_server()
        if isinstance(keys, str):
            keys = [keys]
        if self.on_this_cluster():
            return obj_store.delete(keys)
        return self.client.delete(keys)

    def clear(self):
        """Clear the cluster's object store."""
        self.check_server()
        if self.on_this_cluster():
            return obj_store.clear()
        return self.client.delete()

    def on_this_cluster(self):
        """Whether this function is being called on the same cluster."""
        return _current_cluster("name") == self.rns_address

    # ----------------- RPC Methods ----------------- #

    def connect_server_client(self, force_reconnect=False):
        if not self.address:
            raise ValueError(f"No address set for cluster <{self.name}>. Is it up?")

        if self._rpc_tunnel and force_reconnect:
            self._rpc_tunnel.close()

        tunnel_refcount = 0
        ssh_tunnel = None
        connected_port = self.server_port
        if self.address in open_cluster_tunnels:
            ssh_tunnel, connected_port, tunnel_refcount = open_cluster_tunnels[
                self.address
            ]
            if isinstance(ssh_tunnel, SSHTunnelForwarder):
                ssh_tunnel.check_tunnels()
                if ssh_tunnel.tunnel_is_up[ssh_tunnel.local_bind_address]:
                    self._rpc_tunnel = ssh_tunnel

        if "localhost" in self.address or ":" in self.address:
            if ":" in self.address:
                # Case 1: "localhost:23324" or <real_ip>:<custom port> (e.g. a port is already open to the server)
                self._rpc_tunnel, connected_port = self.address.split(":")
            else:
                # Case 2: "localhost"
                self._rpc_tunnel = self.address
        elif (
            self.server_connection_type
            not in [ServerConnectionType.NONE, ServerConnectionType.TLS]
            and ssh_tunnel is None
        ):
            # Case 3: server connection requires SSH tunnel, but we don't have one up yet
            self._rpc_tunnel, connected_port = self.ssh_tunnel(
                self.server_port,
                remote_port=self.server_port,
                num_ports_to_try=5,
            )

        open_cluster_tunnels[self.address] = (
            self._rpc_tunnel,
            connected_port,
            tunnel_refcount + 1,
        )

        if self._rpc_tunnel:
            logger.info(
                f"Connecting to server via SSH, port forwarding via port {connected_port}."
            )

        use_https = self._use_https
        cert_path = self.cert_config.cert_path if use_https else None

        # Connecting to localhost because it's tunneled into the server at the specified port.
        creds = self.ssh_creds()
        if self.server_connection_type in [
            ServerConnectionType.SSH,
            ServerConnectionType.AWS_SSM,
            ServerConnectionType.PARAMIKO,
        ]:
            ssh_user = creds.get("ssh_user")
            password = creds.get("password")
            auth = (ssh_user, password) if ssh_user and password else None
            self.client = HTTPClient(
                host=self.LOCALHOST,
                port=connected_port,
                auth=auth,
                cert_path=cert_path,
                use_https=use_https,
            )
        else:
            self.client = HTTPClient(
                host=self.server_address,
                port=connected_port,
                cert_path=cert_path,
                use_https=use_https,
            )

    def check_server(self, restart_server=True):
        if self.on_this_cluster():
            return

        # For OnDemandCluster, this initial check doesn't trigger a sky.status, which is slow.
        # If cluster simply doesn't have an address we likely need to up it.
        if not self.address and not self.is_up():
            if not hasattr(self, "up"):
                raise ValueError(
                    "Cluster must have a host address (i.e. be up) or have a reup_cluster method "
                    "(e.g. OnDemandCluster)."
                )
            # If this is a OnDemandCluster, before we up the cluster, run a sky.status to see if the cluster
            # is already up but doesn't have an address assigned yet.
            self.up_if_not()

        if not self.client:
            try:
                self.connect_server_client()
                cluster_config = self.config_for_rns
                if "live_state" in cluster_config.keys():
                    # a bunch of setup commands that mess up json dump
                    del cluster_config["live_state"]
                logger.info(f"Checking server {self.name}")
                self.client.check_server()
                logger.info(f"Server {self.name} is up.")
            except (
                requests.exceptions.ConnectionError,
                sshtunnel.BaseSSHTunnelForwarderError,
                requests.exceptions.ReadTimeout,
                requests.exceptions.ChunkedEncodingError,
            ):
                # It's possible that the cluster went down while we were trying to install packages.
                if not self.is_up():
                    logger.info(f"Server {self.name} is down.")
                    self.up_if_not()
                elif restart_server:
                    logger.info(
                        f"Server {self.name} is up, but the HTTP server may not be up."
                    )
                    self.restart_server()
                    for i in range(5):
                        logger.info(f"Checking server {self.name} again [{i + 1}/5].")
                        try:
                            self.client.check_server()
                            logger.info(f"Server {self.name} is up.")
                            return
                        except (
                            requests.exceptions.ConnectionError,
                            requests.exceptions.ReadTimeout,
                        ) as error:
                            if i == 5:
                                print(error)
                            time.sleep(5)
                raise ValueError(f"Could not connect to cluster <{self.name}>")

        import runhouse

        if self._rh_version is None:
            self._rh_version = self._get_rh_version()

        if not runhouse.__version__ == self._rh_version:
            logger.warning(
                f"Server was started with Runhouse version ({self._rh_version}), "
                f"but local Runhouse version is ({runhouse.__version__})"
            )

        return

    def ssh_tunnel(
        self, local_port, remote_port=None, num_ports_to_try: int = 0
    ) -> Tuple[SSHTunnelForwarder, int]:
        # Debugging cmds (mac):
        # netstat -vanp tcp | grep 32300
        # lsof -i :32300
        # kill -9 <pid>

        creds: dict = self.ssh_creds()
        connected = False
        ssh_tunnel = None
        while not connected:
            try:
                if local_port > local_port + num_ports_to_try:
                    raise Exception(
                        f"Failed to create SSH tunnel after {num_ports_to_try} attempts"
                    )

                if creds.get("ssh_proxy_command"):
                    # Start a tunnel using self.run in a thread, instead of ssh_tunnel
                    ssh_credentials = copy.copy(self.ssh_creds())
                    host = ssh_credentials.pop("ssh_host", self.address)
                    runner = SkySSHRunner(host, **ssh_credentials)
                    runner.tunnel(local_port, remote_port)
                    ssh_tunnel = runner  # Just to keep the object in memory
                else:
                    ssh_tunnel = SSHTunnelForwarder(
                        self.address,
                        ssh_username=creds.get("ssh_user"),
                        ssh_pkey=creds.get("ssh_private_key"),
                        ssh_password=creds.get("password"),
                        local_bind_address=("", local_port),
                        remote_bind_address=(
                            self.LOCALHOST,
                            remote_port or local_port,
                        ),
                        set_keepalive=1,
                        # mute_exceptions=True,
                    )
                    ssh_tunnel.start()
                connected = True
            except HandlerSSHTunnelForwarderError:
                # try connecting with a different port - most likely the issue is the port is already taken
                local_port += 1
                num_ports_to_try -= 1
                pass

        return ssh_tunnel, local_port

    @property
    def _use_https(self) -> bool:
        """Use HTTPS if server connection type is set to ``tls``"""
        connection_type = self.server_connection_type
        tls_conn = ServerConnectionType.TLS.value

        if isinstance(connection_type, str):
            return connection_type == tls_conn

        return connection_type.value == tls_conn

    @property
    def _use_nginx(self) -> bool:
        """Use Nginx if the server port is set to the default HTTP (80) or HTTPS (443) port.
        Note: Nginx will serve as a reverse proxy, forwarding traffic from the server port to the Runhouse API
        server running on port 32300."""
        return self.server_port in [self.DEFAULT_HTTP_PORT, self.DEFAULT_HTTPS_PORT]

    @property
    def _use_custom_cert(self):
        return Path(self.cert_config.cert_path).exists()

    @property
    def _use_custom_key(self):
        return Path(self.cert_config.key_path).exists()

    @staticmethod
    def _add_flags_to_commands(flags, start_screen_cmd, server_start_cmd):
        flags_str = "".join(flags)

        start_screen_cmd = start_screen_cmd.replace(
            server_start_cmd, server_start_cmd + flags_str
        )
        server_start_cmd += flags_str

        return start_screen_cmd, server_start_cmd

    @classmethod
    def _start_server_cmds(
        cls,
        restart,
        restart_ray,
        screen,
        create_logfile,
        host,
        port,
        use_https,
        den_auth,
        ssl_keyfile,
        ssl_certfile,
        force_reinstall,
        use_nginx,
        certs_address,
        enable_local_span_collection,
    ):
        cmds = []
        if restart:
            cmds.append(cls.SERVER_STOP_CMD)
        if restart_ray:
            cmds.append(cls.RAY_KILL_CMD)
            # TODO Add in BOOTSTRAP file if it exists?
            cmds.append(cls.RAY_START_CMD)

        server_start_cmd = cls.SERVER_START_CMD
        start_screen_cmd = cls.START_SCREEN_CMD

        flags = []

        den_auth_flag = " --use-den-auth" if den_auth else ""
        if den_auth_flag:
            logger.info("Starting server with Den auth.")
            flags.append(den_auth_flag)

        force_reinstall_flag = " --force-reinstall" if force_reinstall else ""
        if force_reinstall_flag:
            logger.info("Reinstalling Nginx and server configs.")
            flags.append(force_reinstall_flag)

        use_nginx_flag = " --use-nginx" if use_nginx else ""
        if use_nginx_flag:
            logger.info("Configuring Nginx on the cluster.")
            flags.append(use_nginx_flag)

        ssl_keyfile_flag = f" --ssl-keyfile {ssl_keyfile}" if ssl_keyfile else ""
        if ssl_keyfile_flag:
            logger.info(f"Using SSL keyfile in path: {ssl_keyfile}")
            flags.append(ssl_keyfile_flag)

        ssl_certfile_flag = f" --ssl-certfile {ssl_certfile}" if ssl_certfile else ""
        if ssl_certfile_flag:
            logger.info(f"Using SSL certfile in path: {ssl_certfile}")
            flags.append(ssl_certfile_flag)

        # Use HTTPS if explicitly specified or if SSL cert or keyfile path are provided
        https_flag = (
            " --use-https" if use_https or (ssl_keyfile or ssl_certfile) else ""
        )
        if https_flag:
            logger.info("Starting server with HTTPS.")
            flags.append(https_flag)

        host_flag = f" --host {host}" if host else ""
        if host_flag:
            logger.info(f"Using host: {host}.")
            flags.append(host_flag)

        port_flag = f" --port {port}" if port else ""
        if port_flag:
            logger.info(f"Using port: {port}.")
            flags.append(port_flag)

        address_flag = f" --certs-address {certs_address}" if certs_address else ""
        if address_flag:
            logger.info(f"Server public IP address: {certs_address}.")
            flags.append(address_flag)

        enable_local_span_collection_flag = (
            " --enable-local-span-collection" if enable_local_span_collection else ""
        )
        if enable_local_span_collection_flag:
            logger.info("Enabling local span telemetry collection on the cluster.")
            flags.append(enable_local_span_collection_flag)

        logger.info(
            f"Starting API server using the following command: {server_start_cmd}."
        )

        if flags:
            start_screen_cmd, server_start_cmd = cls._add_flags_to_commands(
                flags, start_screen_cmd, server_start_cmd
            )

        if screen:
            if create_logfile and not Path(cls.SERVER_LOGFILE).exists():
                Path(cls.SERVER_LOGFILE).parent.mkdir(parents=True, exist_ok=True)
                Path(cls.SERVER_LOGFILE).touch()
            cmds.append(start_screen_cmd)
        else:
            cmds.append(server_start_cmd)

        return cmds

    def restart_server(
        self,
        _rh_install_url: str = None,
        resync_rh: bool = True,
        restart_ray: bool = True,
        env_activate_cmd: str = None,
        restart_proxy: bool = False,
    ):
        """Restart the RPC server.

        Args:
            resync_rh (bool): Whether to resync runhouse. (Default: ``True``)
            restart_ray (bool): Whether to restart Ray. (Default: ``True``)
            env_activate_cmd (str, optional): Command to activate the environment on the server. (Default: ``None``)
            restart_proxy (bool): Whether to restart nginx on the cluster, if configured. (Default: ``False``)
        Example:
            >>> rh.cluster("rh-cpu").restart_server()
        """
        logger.info(f"Restarting Runhouse API server on {self.name}.")

        if resync_rh:
            self._sync_runhouse_to_cluster(_install_url=_rh_install_url)

        # Update the cluster config on the cluster
        self.save_config_to_cluster()

        use_custom_cert = self._use_custom_cert
        if use_custom_cert:
            # Copy the provided cert onto the cluster
            from runhouse import folder

            cert_dir = self.cert_config.DEFAULT_CERT_DIR
            folder(path=self.cert_config.cert_dir).to(self, path=cert_dir)

            # Path to cert file stored on the cluster
            cluster_cert_path = f"{cert_dir}/{self.cert_config.CERT_NAME}"
            logger.info(
                f"Copied TLS cert onto the cluster in path: {cluster_cert_path}"
            )

        use_custom_key = self._use_custom_key
        if use_custom_key:
            # Copy the provided key onto the cluster
            from runhouse import folder

            keyfile_dir = self.cert_config.DEFAULT_PRIVATE_KEY_DIR
            folder(path=self.cert_config.key_dir).to(self, path=keyfile_dir)

            # Path to key file stored on the cluster
            cluster_key_path = f"{keyfile_dir}/{self.cert_config.PRIVATE_KEY_NAME}"

            logger.info(
                f"Copied TLS keyfile onto the cluster in path: {cluster_key_path}"
            )

        https_flag = self._use_https
        nginx_flag = self._use_nginx
        cmd = (
            self.CLI_RESTART_CMD
            + (" --no-restart-ray" if not restart_ray else "")
            + (" --use-https" if https_flag else "")
            + (" --use-nginx" if nginx_flag else "")
            + (" --restart-proxy" if restart_proxy and nginx_flag else "")
            + (f" --ssl-certfile {cluster_cert_path}" if use_custom_cert else "")
            + (f" --ssl-keyfile {cluster_key_path}" if use_custom_key else "")
        )

        cmd = f"{env_activate_cmd} && {cmd}" if env_activate_cmd else cmd

        status_codes = self.run(commands=[cmd])
        if not status_codes[0][0] == 0:
            raise ValueError(f"Failed to restart server {self.name}.")

        if https_flag:
            rns_address = self.rns_address
            if not rns_address:
                raise ValueError("Cluster must have a name in order to enable HTTPS.")

            if not self.client:
                logger.info("Reconnecting server client. Server restarted with HTTPS.")
                self.connect_server_client()

            # Refresh the client params to use HTTPS
            self.client.use_https = https_flag
            self.client.cert_path = self.cert_config.cert_path

        self._rh_version = self._get_rh_version()

        return status_codes

    @contextlib.contextmanager
    def pause_autostop(self):
        """Context manager to temporarily pause autostop. Mainly for OnDemand clusters, for BYO cluster
        there is no autostop."""
        pass

    def call(
        self,
        module_name,
        method_name,
        *args,
        stream_logs=True,
        run_name=None,
        remote=False,
        run_async=False,
        save=False,
        **kwargs,
    ):
        """Call a method on a module that is in the cluster's object store.

        Args:
            module_name (str): Name of the module saved on system.
            method_name (str): Name of the method.
            stream_logs (bool): Whether to stream logs from the method call.
            run_name (str): Name for the run.
            remote (bool): Return a remote object from the function, rather than the result proper.
            run_async (bool): Run the method asynchronously and return a run_key to retreive results and logs later.
            *args: Positional arguments to pass to the method.
            **kwargs: Keyword arguments to pass to the method.

        Example:
            >>> cluster.call("my_module", "my_method", arg1, arg2, kwarg1=kwarg1)
        """
        self.check_server()
        # Note: might be single value, might be a generator!
        if self.on_this_cluster():
            # TODO
            pass
        return self.client.call_module_method(
            module_name,
            method_name,
            stream_logs=stream_logs,
            run_name=run_name,
            remote=remote,
            run_async=run_async,
            save=save,
            args=args,
            kwargs=kwargs,
            system=self,
        )

    def is_connected(self):
        """Whether the RPC tunnel is up.

        Example:
            >>> connected = cluster.is_connected()
        """
        return self.client is not None

    def disconnect(self):
        """Disconnect the RPC tunnel.

        Example:
            >>> cluster.disconnect()
        """
        if self._rpc_tunnel:
            self._rpc_tunnel.stop()
        # if self.client:
        #     self.client.shutdown()

    def __getstate__(self):
        """Delete non-serializable elements (e.g. thread locks) before pickling."""
        state = self.__dict__.copy()
        state["client"] = None
        state["_rpc_tunnel"] = None
        return state

    # ----------------- SSH Methods ----------------- #

    def ssh_creds(self):
        """Retrieve SSH credentials."""
        return self._ssh_creds

    def _rsync(
        self,
        source: str,
        dest: str,
        up: bool,
        contents: bool = False,
        filter_options: str = None,
        stream_logs: bool = False,
    ):
        """
        Sync the contents of the source directory into the destination.

        .. note:
            Ending `source` with a slash will copy the contents of the directory into dest,
            while omitting it will copy the directory itself (adding a directory layer).
        """
        # FYI, could be useful: https://github.com/gchamon/sysrsync
        if contents:
            source = source + "/" if not source.endswith("/") else source
            dest = dest + "/" if not dest.endswith("/") else dest

        ssh_credentials = copy.copy(self.ssh_creds())
        ssh_credentials.pop("ssh_host", self.address)
        pwd = ssh_credentials.pop("password", None)

        runner = SkySSHRunner(self.address, **ssh_credentials)
        if not pwd:
            if up:
                runner.run(["mkdir", "-p", dest], stream_logs=False)
            else:
                Path(dest).expanduser().parent.mkdir(parents=True, exist_ok=True)
            runner.rsync(
                source,
                dest,
                up=up,
                filter_options=filter_options,
                stream_logs=stream_logs,
            )
        else:
            import pexpect

            if up:
                ssh_command = runner.run(
                    ["mkdir", "-p", dest],
                    stream_logs=False,
                    return_cmd=True,
                    ssh_mode=SshMode.INTERACTIVE,
                )
                ssh = pexpect.spawn(ssh_command, encoding="utf-8")
                ssh.logfile_read = sys.stdout
                # If CommandRunner uses the control path, the password may not be requested
                next_line = ssh.expect(["assword:", pexpect.EOF])
                if next_line == 0:
                    ssh.sendline(pwd)
                    ssh.expect(pexpect.EOF)
                ssh.close()
            else:
                Path(dest).expanduser().parent.mkdir(parents=True, exist_ok=True)

            rsync_cmd = runner.rsync(
                source,
                dest,
                up=up,
                filter_options=filter_options,
                stream_logs=stream_logs,
                return_cmd=True,
            )
            ssh = pexpect.spawn(rsync_cmd, encoding="utf-8")
            if stream_logs:
                # FYI This will print a ton of of stuff to stdout
                ssh.logfile_read = sys.stdout

            # If CommandRunner uses the control path, the password may not be requested
            next_line = ssh.expect(["assword:", pexpect.EOF])
            if next_line == 0:
                ssh.sendline(pwd)
                ssh.expect(pexpect.EOF)
            ssh.close()

    def ssh(self):
        """SSH into the cluster

        Example:
            >>> rh.cluster("rh-cpu").ssh()
        """
        creds = self.ssh_creds()

        if creds.get("ssh_private_key"):
            cmd = (
                f"ssh {creds['ssh_user']}:{self.address} -i {creds['ssh_private_key']}"
            )
        else:
            # if needs a password, will prompt for manual password
            cmd = f"ssh {creds['ssh_user']}@{self.address}"

        subprocess.run(cmd.split(" "))

    def _ping(self, timeout=5):
        ssh_call = threading.Thread(target=lambda: self.run(['echo "hello"']))
        ssh_call.start()
        ssh_call.join(timeout=timeout)
        if ssh_call.is_alive():
            raise TimeoutError("SSH call timed out")
        return True

    def _get_rh_version(self):
        return self.run_python(["import runhouse", "print(runhouse.__version__)"])[0][
            1
        ].strip()

    def run(
        self,
        commands: List[str],
        env: Union["Env", str] = None,
        stream_logs: bool = True,
        port_forward: Union[None, int, Tuple[int, int]] = None,
        require_outputs: bool = True,
        run_name: Optional[str] = None,
    ) -> list:
        """Run a list of shell commands on the cluster. If `run_name` is provided, the commands will be
        sent over to the cluster before being executed and a Run object will be created.

        Example:
            >>> cpu.run(["pip install numpy"])
            >>> cpu.run(["pip install numpy", env="my_conda_env"])
            >>> cpu.run(["python script.py"], run_name="my_exp")
        """
        # TODO [DG] suspend autostop while running
        from runhouse.resources.provenance import run

        cmd_prefix = ""
        if env:
            if isinstance(env, str):
                from runhouse.resources.envs import Env

                env = Env.from_name(env)
            cmd_prefix = env._run_cmd

        if not run_name:
            # If not creating a Run then just run the commands via SSH and return
            return self._run_commands_with_ssh(
                commands, cmd_prefix, stream_logs, port_forward, require_outputs
            )

        # Create and save the Run locally
        with run(name=run_name, cmds=commands, overwrite=True) as r:
            return_codes = self._run_commands_with_ssh(
                commands, cmd_prefix, stream_logs, port_forward, require_outputs
            )

        # Register the completed Run
        r._register_cmd_run_completion(return_codes)
        logger.info(f"Saved Run to path: {r.folder.path}")
        return return_codes

    def _run_commands_with_ssh(
        self,
        commands: list,
        cmd_prefix: str,
        stream_logs: bool,
        port_forward: int = None,
        require_outputs: bool = True,
    ):
        return_codes = []

        ssh_credentials = copy.copy(self.ssh_creds())
        host = ssh_credentials.pop("ssh_host", self.address)
        pwd = ssh_credentials.pop("password", None)

        runner = SkySSHRunner(host, **ssh_credentials)

        if not pwd:
            for command in commands:
                command = f"{cmd_prefix} {command}" if cmd_prefix else command
                logger.info(f"Running command on {self.name}: {command}")
                ret_code = runner.run(
                    command,
                    require_outputs=require_outputs,
                    stream_logs=stream_logs,
                    port_forward=port_forward,
                )
                return_codes.append(ret_code)
        else:
            import pexpect

            for command in commands:
                command = f"{cmd_prefix} {command}" if cmd_prefix else command
                logger.info(f"Running command on {self.name}: {command}")
                # We need to quiet the SSH output here or it will print
                # "Shared connection to ____ closed." at the end, which messes with the output.
                ssh_command = runner.run(
                    command,
                    require_outputs=require_outputs,
                    stream_logs=stream_logs,
                    port_forward=port_forward,
                    return_cmd=True,
                    ssh_mode=SshMode.INTERACTIVE,
                    quiet_ssh=True,
                )
                ssh = pexpect.spawn(ssh_command, encoding="utf-8")
                if stream_logs:
                    ssh.logfile_read = sys.stdout
                next_line = ssh.expect(["assword:", pexpect.EOF])
                if next_line == 0:
                    ssh.sendline(pwd)
                    ssh.expect(pexpect.EOF)
                ssh.close()
                # Filter color characters from ssh.before, as otherwise sometimes random color characters
                # will be printed to the console.
                ssh.before = re.sub(r"\x1b\[[0-9;]*m", "", ssh.before)
                return_codes.append(
                    [ssh.exitstatus, ssh.before.strip(), ssh.signalstatus]
                )

        return return_codes

    def run_python(
        self,
        commands: List[str],
        env: Union["Env", str] = None,
        stream_logs: bool = True,
        port_forward: Optional[int] = None,
        run_name: Optional[str] = None,
    ):
        """Run a list of python commands on the cluster.

        Example:
            >>> cpu.run_python(['import numpy', 'print(numpy.__version__)'])
            >>> cpu.run_python(["print('hello')"])

        Note:
            Running Python commands with nested quotes can be finicky. If using nested quotes,
            try to wrap the outer quote with double quotes (") and the inner quotes with a single quote (').
        """
        cmd_prefix = "python3 -c"
        if env:
            if isinstance(env, str):
                from runhouse.resources.envs import Env

                env = Env.from_name(env)
            cmd_prefix = f"{env._run_cmd} {cmd_prefix}"
        command_str = "; ".join(commands)
        command_str_repr = (
            repr(repr(command_str))[2:-2]
            if self.ssh_creds().get("password")
            else command_str
        )
        formatted_command = f'{cmd_prefix} "{command_str_repr}"'

        # If invoking a run as part of the python commands also return the Run object
        return_codes = self.run(
            [formatted_command],
            stream_logs=stream_logs,
            port_forward=port_forward,
            run_name=run_name,
        )

        return return_codes

    def sync_secrets(self, providers: Optional[List[str]] = None):
        """Send secrets for the given providers.

        Args:
            providers(List[str] or None): List of providers to send secrets for.
                If `None`, all providers configured in the environment will by sent.

        Example:
            >>> cpu.sync_secrets(providers=["aws", "lambda"])
        """
        self.check_server()

        from runhouse import Secrets

        Secrets.to(system=self, providers=providers)

    def ipython(self):
        # TODO tunnel into python interpreter in cluster
        pass

    def notebook(
        self,
        persist: bool = False,
        sync_package_on_close: Optional[str] = None,
        port_forward: int = 8888,
    ):
        """Tunnel into and launch notebook from the cluster.

        Example:
            >>> rh.cluster("test-cluster").notebook()
        """
        tunnel, port_fwd = self.ssh_tunnel(local_port=port_forward, num_ports_to_try=10)
        try:
            install_cmd = "pip install jupyterlab"
            jupyter_cmd = f"jupyter lab --port {port_fwd} --no-browser"
            with self.pause_autostop():
                self.run(commands=[install_cmd, jupyter_cmd], stream_logs=True)

        finally:
            if sync_package_on_close:
                from runhouse.resources.packages.package import Package

                if sync_package_on_close == "./":
                    sync_package_on_close = rns_client.locate_working_dir()
                pkg = Package.from_string("local:" + sync_package_on_close)
                self._rsync(source=f"~/{pkg.name}", dest=pkg.local_path, up=False)
            if not persist:
                tunnel.stop()
                kill_jupyter_cmd = f"jupyter notebook stop {port_fwd}"
                self.run(commands=[kill_jupyter_cmd])

    def remove_conda_env(
        self,
        env: Union[str, "CondaEnv"],
    ):
        """Remove conda env from the cluster.

        Example:
            >>> rh.ondemand_cluster("rh-cpu").remove_conda_env("my_conda_env")
        """
        env_name = env if isinstance(env, str) else env.env_name
        self.run([f"conda env remove -n {env_name}"])

    def download_cert(self):
        """Download certificate from the cluster (Note: user must have access to the cluster)"""
        self.client.get_certificate()
        logger.info(
            f"Latest TLS certificate for {self.name} saved to local path: {self.cert_config.cert_path}"
        )
