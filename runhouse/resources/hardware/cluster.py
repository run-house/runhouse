import contextlib
import copy
import importlib
import json
import logging
import re
import subprocess
import sys
import threading
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from runhouse.resources.envs.utils import run_with_logs

from runhouse.rns.utils.api import ResourceAccess, ResourceVisibility
from runhouse.servers.http.certs import TLSCertConfig

# Filter out DeprecationWarnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import requests.exceptions

from runhouse.constants import (
    CLI_RESTART_CMD,
    CLI_STOP_CMD,
    CLUSTER_CONFIG_PATH,
    DEFAULT_HTTP_PORT,
    DEFAULT_HTTPS_PORT,
    DEFAULT_RAY_PORT,
    DEFAULT_SERVER_PORT,
    EMPTY_DEFAULT_ENV_NAME,
    LOCALHOST,
    RESERVED_SYSTEM_NAMES,
)
from runhouse.globals import obj_store, rns_client
from runhouse.resources.envs.utils import _get_env_from
from runhouse.resources.hardware.utils import (
    _current_cluster,
    _unnamed_default_env_name,
    ServerConnectionType,
)
from runhouse.resources.resource import Resource

from runhouse.servers.http import HTTPClient

logger = logging.getLogger(__name__)


class Cluster(Resource):
    RESOURCE_TYPE = "cluster"
    REQUEST_TIMEOUT = 5  # seconds

    DEFAULT_SSH_PORT = 22
    EMPTY_DEFAULT_ENV_NAME = EMPTY_DEFAULT_ENV_NAME

    def __init__(
        self,
        # Name will almost always be provided unless a "local" cluster is created
        name: Optional[str] = None,
        ips: List[str] = None,
        creds: "Secret" = None,
        default_env: "Env" = None,
        server_host: str = None,
        server_port: int = None,
        ssh_port: int = None,
        client_port: int = None,
        server_connection_type: str = None,
        ssl_keyfile: str = None,
        ssl_certfile: str = None,
        domain: str = None,
        den_auth: bool = False,
        use_local_telemetry: bool = False,
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

        self._rpc_tunnel = None
        self._creds = creds

        self.ips = ips
        self.client = None
        self.den_auth = den_auth or False
        self.cert_config = TLSCertConfig(cert_path=ssl_certfile, key_path=ssl_keyfile)

        self.ssl_certfile = ssl_certfile
        self.ssl_keyfile = ssl_keyfile
        self.server_connection_type = server_connection_type
        self.server_port = server_port
        self.client_port = client_port
        self.ssh_port = ssh_port or self.DEFAULT_SSH_PORT
        self.server_host = server_host
        self.domain = domain
        self.use_local_telemetry = use_local_telemetry

        self._default_env = _get_env_from(default_env)

    @property
    def address(self):
        return self.ips[0] if isinstance(self.ips, List) else None

    @address.setter
    def address(self, addr):
        self.ips = self.ips or [None]
        self.ips[0] = addr

    @property
    def creds_values(self) -> Dict:
        if not self._creds:
            return {}

        return self._creds.values

    @property
    def default_env(self):
        from runhouse.resources.envs import Env

        return (
            self._default_env
            if self._default_env
            else Env(name=EMPTY_DEFAULT_ENV_NAME, working_dir="./")
        )

    @default_env.setter
    def default_env(self, env):
        self._default_env = _get_env_from(env)
        if not self._default_env.name:
            self._default_env.name = _unnamed_default_env_name(self.name)

        if self.is_up():
            self.check_server()
            self._default_env.to(self)
            self.save_config_to_cluster()

            logger.info(
                "The cluster default env has been updated. "
                "Run `cluster.restart_server()` to restart the Runhouse server on the new default env."
            )

    def save_config_to_cluster(self, node: str = None):
        config = self.config(condensed=False)
        config.pop("creds")
        json_config = f"{json.dumps(config)}"

        self.run(
            [
                f"mkdir -p ~/.rh; touch {CLUSTER_CONFIG_PATH}; echo '{json_config}' > {CLUSTER_CONFIG_PATH}"
            ],
            node=node or "all",
        )

    def save(self, name: str = None, overwrite: bool = True, folder: str = None):
        """Overrides the default resource save() method in order to also update
        the cluster config on the cluster itself.
        """
        on_this_cluster = self.on_this_cluster()

        # Save the cluster sub-resources (ex: ssh creds) using the top level folder of the cluster if the the rns
        # address has been set, otherwise use the user's current folder (set in local .rh config)
        base_folder = rns_client.base_folder(self.rns_address)
        folder = (
            folder or f"/{base_folder}" if base_folder else rns_client.current_folder
        )

        super().save(name=name, overwrite=overwrite, folder=folder)

        # Running save will have updated the cluster's
        # RNS address. We need to update the name
        # used in the config on the cluster so that
        # self.on_this_cluster() will still work as expected.
        if on_this_cluster:
            obj_store.set_cluster_config_value("name", self.rns_address)
        elif self.client:
            self.client.set_cluster_name(self.rns_address)

        return self

    def delete_configs(self):
        if self._creds:
            logger.debug(
                f"Attempting to delete creds associated with cluster {self.name}"
            )
            rns_client.delete_configs(self._creds)

        super().delete_configs()

    def _save_sub_resources(self, folder: str = None):
        from runhouse.resources.envs import Env
        from runhouse.resources.secrets import Secret

        if self._creds and isinstance(self._creds, Secret):
            self._creds.save(folder=folder)

        if self._default_env and isinstance(self._default_env, Env):
            if not self._default_env.name:
                self._default_env.name = _unnamed_default_env_name(self.name)
            self._default_env.save()

    @classmethod
    def from_config(cls, config: dict, dryrun=False, _resolve_children=True):
        resource_subtype = config.get("resource_subtype")
        if _resolve_children:
            config = cls._check_for_child_configs(config)

        if resource_subtype == "Cluster":
            return Cluster(**config, dryrun=dryrun)
        elif resource_subtype == "OnDemandCluster":
            from .on_demand_cluster import OnDemandCluster

            return OnDemandCluster(**config, dryrun=dryrun)
        elif resource_subtype == "SageMakerCluster":
            from .sagemaker.sagemaker_cluster import SageMakerCluster

            return SageMakerCluster(**config, dryrun=dryrun)
        else:
            raise ValueError(f"Unknown cluster type {resource_subtype}")

    def config(self, condensed=True):
        config = super().config(condensed)
        self.save_attrs_to_config(
            config,
            [
                "ips",
                "server_port",
                "server_host",
                "server_connection_type",
                "domain",
                "den_auth",
                "use_local_telemetry",
                "ssh_port",
                "client_port",
            ],
        )
        creds = self._resource_string_for_subconfig(self._creds, condensed)

        # user A shares cluster with user B, with "write" permissions. If user B will save the cluster to Den, we
        # would NOT like that the loaded secret will overwrite the original secret that was created and shared by
        # user A.
        if creds and "loaded_secret_" in creds:
            creds = creds.replace("loaded_secret_", "")

        config["creds"] = creds
        config["api_server_url"] = rns_client.api_server_url

        if self._default_env:
            default_env = self._resource_string_for_subconfig(
                self._default_env, condensed
            )
            config["default_env"] = default_env

        if self._use_custom_certs:
            config["ssl_certfile"] = self.cert_config.cert_path
            config["ssl_keyfile"] = self.cert_config.key_path

        return config

    def endpoint(self, external=False):
        """Endpoint for the cluster's Daemon server. If external is True, will only return the external url,
        and will return None otherwise (e.g. if a tunnel is required). If external is False, will either return
        the external url if it exists, or will set up the connection (based on connection_type) and return
        the internal url (including the local connected port rather than the sever port). If cluster is not up,
        returns None.
        """
        if not (self.address or self.on_this_cluster()):
            return None

        client_port = self.client_port or self.server_port

        if self.server_connection_type in [
            ServerConnectionType.NONE,
            ServerConnectionType.TLS,
        ]:
            url_base = (
                "https"
                if self.server_connection_type == ServerConnectionType.TLS
                else "http"
            )

            # Client port gets set to the server port if it was not set.
            # In the case of local, testing clusters, the client port will be set to something else
            # since we need to port forward in order to hit localhost.
            if client_port not in [DEFAULT_HTTP_PORT, DEFAULT_HTTPS_PORT]:
                return f"{url_base}://{self.server_address}:{client_port}"
            else:
                return f"{url_base}://{self.server_address}"

        if external:
            return None

        if self.server_connection_type in [
            ServerConnectionType.SSH,
            ServerConnectionType.AWS_SSM,
        ]:
            self.check_server()
            return f"http://{LOCALHOST}:{client_port}"

    def _client(self, restart_server=True):
        if self.on_this_cluster():
            # Previously (before calling within the same cluster worked) returned None
            return obj_store
        if not self.client:
            self.check_server(restart_server=restart_server)
        return self.client

    @property
    def server_address(self):
        """Address to use in the requests made to the cluster. If creating an SSH tunnel with the cluster,
        ths will be set to localhost, otherwise will use the cluster's domain (if provided), or its
        public IP address."""
        if self.domain:
            return self.domain

        if self.server_host in [LOCALHOST, "localhost"]:
            return LOCALHOST

        return self.address

    @property
    def is_shared(self) -> bool:
        from runhouse import Secret

        ssh_creds = self.creds_values
        if not ssh_creds:
            return False

        ssh_private_key = ssh_creds.get("ssh_private_key")
        ssh_private_key_path = Path(ssh_private_key).expanduser()
        secrets_base_dir = Path(Secret.DEFAULT_DIR).expanduser()

        # Check if the key path is saved down in the local .rh directory, which we only do for shared credentials
        if str(ssh_private_key_path).startswith(str(secrets_base_dir)):
            return True

        return f"{self._creds.name}/" in ssh_creds.get("ssh_private_key", "")

    def is_up(self) -> bool:
        """Check if the cluster is up.

        Example:
            >>> rh.cluster("rh-cpu").is_up()
        """
        return self.on_this_cluster() or self.address is not None

    def up_if_not(self):
        """Bring up the cluster if it is not up. No-op if cluster is already up.
        This only applies to on-demand clusters, and has no effect on self-managed clusters.

        Example:
            >>> rh.cluster("rh-cpu").up_if_not()
        """
        if not self.is_up():
            # Don't store stale IPs
            self.ips = None
            self.up()
        return self

    def up(self):
        raise NotImplementedError(
            f"Cluster <{self.name}> does not have an up method. It must be brought up manually."
        )

    def keep_warm(self):
        logger.info(
            f"cluster.keep_warm will have no effect on self-managed cluster {self.name}."
        )
        return self

    def _sync_default_env_to_cluster(self):
        """Install and set up the default env requirements on the cluster. This does not put the env resource
        on the cluster or initialize the servlet. It also does not set any env vars."""
        if not self._default_env:
            return

        logging.info(f"Syncing default env {self._default_env.name} to cluster")
        self._default_env.install(cluster=self)

    def _sync_runhouse_to_cluster(self, _install_url=None, env=None):
        if self.on_this_cluster():
            return

        if not self.address:
            raise ValueError(f"No address set for cluster <{self.name}>. Is it up?")

        env = env or self.default_env

        local_rh_package_path = Path(importlib.util.find_spec("runhouse").origin).parent

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
                node="all",
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

        for node in self.ips:
            status_codes = self.run(
                [rh_install_cmd],
                node=node,
                env=env,
                stream_logs=True,
            )

            if status_codes[0][0] != 0:
                raise ValueError(
                    f"Error installing runhouse on cluster <{self.name}> node <{node}>"
                )

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
        self.check_server()
        env = _get_env_from(env) if env else self.default_env
        env.reqs = env._reqs + reqs
        env.to(self)

    def get(self, key: str, default: Any = None, remote=False):
        """Get the result for a given key from the cluster's object store. To raise an error if the key is not found,
        use `cluster.get(key, default=KeyError)`."""
        self.check_server()
        if self.on_this_cluster():
            return obj_store.get(key, default=default, remote=remote)
        try:
            res = self.client.get(
                key,
                default=default,
                remote=remote,
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

    # TODO This should accept an env (for env var secrets and docker envs).
    def add_secrets(
        self, provider_secrets: List[str or "Secret"], env: Union[str, "Env"] = None
    ):
        """Copy secrets from current environment onto the cluster"""
        self.check_server()
        self.sync_secrets(provider_secrets, env=env or self.default_env)

    def put(self, key: str, obj: Any, env=None):
        """Put the given object on the cluster's object store at the given key."""
        self.check_server()
        if self.on_this_cluster():
            return obj_store.put(key, obj, env=env)
        return self.client.put_object(key, obj, env=env or self.default_env.name)

    def put_resource(
        self, resource: Resource, state: Dict = None, dryrun: bool = False, env=None
    ):
        """Put the given resource on the cluster's object store. Returns the key (important if name is not set)."""
        self.check_server()

        if resource.RESOURCE_TYPE == "env" and not resource.name:
            resource.name = self.default_env.name

        # Logic to get env_name from different ways env can be provided
        env = env or (
            resource.env
            if hasattr(resource, "env")
            else resource.name or resource.env_name
            if resource.RESOURCE_TYPE == "env"
            else self.default_env
        )
        if resource.RESOURCE_TYPE == "env" and not resource.name:
            resource.name = self.default_env.name

        if env and not isinstance(env, str):
            env = _get_env_from(env)
            env_name = env.name or self.default_env.name
        else:
            env_name = env

        # Env name could somehow be a full length `username/base_env`, trim it down to just the env name
        if env_name:
            env_name = env_name.split("/")[-1]

        state = state or {}
        if self.on_this_cluster():
            data = (resource.config(condensed=False), state, dryrun)
            return obj_store.put_resource(serialized_data=data, env_name=env_name)
        return self.client.put_resource(
            resource, state=state or {}, env_name=env_name, dryrun=dryrun
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
        config = _current_cluster("config")
        return config is not None and config.get("name") == (
            self.rns_address or self.name
        )

    # ----------------- RPC Methods ----------------- #

    def connect_server_client(self, force_reconnect=False):
        if not self.address:
            raise ValueError(f"No address set for cluster <{self.name}>. Is it up?")

        if self._rpc_tunnel and force_reconnect:
            self._rpc_tunnel.terminate()
            self._rpc_tunnel = None

        if self.server_connection_type in [
            ServerConnectionType.SSH,
            ServerConnectionType.AWS_SSM,
        ]:
            # Case 1: Server connection requires SSH tunnel, but we don't have one up yet
            self._rpc_tunnel = self.ssh_tunnel(
                local_port=self.server_port,
                remote_port=self.server_port,
                num_ports_to_try=10,
            )
            self.client_port = self._rpc_tunnel.local_bind_port

            ssh_user = self.creds_values.get("ssh_user")
            password = self.creds_values.get("password")
            auth = (ssh_user, password) if ssh_user and password else None

            # Connecting to localhost because it's tunneled into the server at the specified port.
            # As long as the tunnel was initialized,
            # self.client_port has been set to the correct port
            self.client = HTTPClient(
                host=LOCALHOST,
                port=self.client_port,
                auth=auth,
                resource_address=self.rns_address,
                system=self,
            )

        else:
            # Case 2: We're making a direct connection to the server, either via HTTP or HTTPS
            if self.server_connection_type not in [
                ServerConnectionType.NONE,
                ServerConnectionType.TLS,
            ]:
                raise ValueError(
                    f"Unknown server connection type {self.server_connection_type}."
                )

            cert_path = None
            if self._use_https and not (self.domain and self._use_caddy):
                # Only use the cert path if HTTPS is enabled and not providing a domain with Caddy
                cert_path = self.cert_config.cert_path

            self.client_port = self.client_port or self.server_port

            self.client = HTTPClient(
                host=self.server_address,
                port=self.client_port,
                cert_path=cert_path,
                use_https=self._use_https,
                resource_address=self.rns_address,
                system=self,
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
                logger.debug(f"Checking server {self.name}")
                self.client.check_server()
                logger.info(f"Server {self.name} is up.")
            except (
                requests.exceptions.ConnectionError,
                requests.exceptions.ReadTimeout,
                requests.exceptions.ChunkedEncodingError,
            ):
                # It's possible that the cluster went down while we were trying to install packages.
                if not self.is_up():
                    logger.info(f"Server {self.name} is down.")
                    self.up_if_not()
                elif restart_server:
                    logger.info(
                        f"Server {self.name} is up, but the Runhouse API server may not be up."
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
                                logger.error(error)
                            time.sleep(5)
                raise ValueError(f"Could not connect to server {self.name}")

        return

    def status(self, resource_address: str = None):
        """Loads the status of the Runhouse daemon running on the cluster."""
        # Note: If running outside a local cluster need to include a resource address to construct the cluster subtoken
        # Allow for specifying a resource address explicitly in case the resource has no rns address yet
        try:
            self.check_server()
            if self.on_this_cluster():
                return obj_store.status()
            return self.client.status(
                resource_address=resource_address or self.rns_address
            )
        except ValueError as e:
            raise e

    def ssh_tunnel(
        self, local_port, remote_port=None, num_ports_to_try: int = 0
    ) -> "SkySSHRunner":
        from runhouse.resources.hardware.sky_ssh_runner import ssh_tunnel

        return ssh_tunnel(
            address=self.address,
            ssh_creds=self.creds_values,
            local_port=local_port,
            ssh_port=self.ssh_port,
            remote_port=remote_port,
            num_ports_to_try=num_ports_to_try,
        )

    @property
    def _use_https(self) -> bool:
        """Use HTTPS if server connection type is set to ``tls``"""

        return (
            self.server_connection_type == ServerConnectionType.TLS
            if self.server_connection_type is not None
            else False
        )

    @property
    def _use_caddy(self) -> bool:
        """Use Caddy if the server port is set to the default HTTP (80) or HTTPS (443) port.
        Note: Caddy will serve as a reverse proxy, forwarding traffic from the server port to the Runhouse API
        server running on port 32300."""
        return self.server_port in [DEFAULT_HTTP_PORT, DEFAULT_HTTPS_PORT]

    @property
    def _use_custom_certs(self):
        """Use custom certs when HTTPS is not enabled, or when HTTPS is enabled, Caddy is enabled,
        and a domain is provided."""
        return self._use_https and not (self._use_caddy and self.domain is not None)

    def _start_ray_workers(self, ray_port, env):
        for host in self.ips:
            if host == self.address:
                # This is the master node, skip
                continue
            logger.info(
                f"Starting Ray on worker {host} with head node at {self.address}:{ray_port}."
            )
            self.run(
                commands=[
                    f"ray start --address={self.address}:{ray_port} --disable-usage-stats",
                ],
                node=host,
                env=env,
            )

    def restart_server(
        self,
        _rh_install_url: str = None,
        resync_rh: bool = True,
        restart_ray: bool = True,
        restart_proxy: bool = False,
    ):
        """Restart the RPC server.

        Args:
            resync_rh (bool): Whether to resync runhouse. (Default: ``True``)
            restart_ray (bool): Whether to restart Ray. (Default: ``True``)
            env (str or Env, optional): Specified environment to restart the server on. (Default: ``None``)
            restart_proxy (bool): Whether to restart Caddy on the cluster, if configured. (Default: ``False``)

        Example:
            >>> rh.cluster("rh-cpu").restart_server()
        """
        logger.info(f"Restarting Runhouse API server on {self.name}.")

        default_env = _get_env_from(self._default_env) if self._default_env else None
        if default_env:
            self._sync_default_env_to_cluster()

        if resync_rh:
            self._sync_runhouse_to_cluster(_install_url=_rh_install_url)
            logger.debug("Finished syncing Runhouse to cluster.")

        https_flag = self._use_https
        caddy_flag = self._use_caddy
        domain = self.domain
        use_local_telemetry = self.use_local_telemetry

        cluster_key_path = None
        cluster_cert_path = None

        if https_flag:
            # Make sure certs are copied to the cluster (where relevant)
            base_cluster_dir = self.cert_config.DEFAULT_CLUSTER_DIR
            cluster_key_path = f"{base_cluster_dir}/{self.cert_config.PRIVATE_KEY_NAME}"
            cluster_cert_path = f"{base_cluster_dir}/{self.cert_config.CERT_NAME}"

            if domain and caddy_flag:
                # Certs generated by Caddy are stored in the data directory path on the cluster
                # https://caddyserver.com/docs/conventions#data-directory

                # Reset to None - Caddy will automatically generate certs on the cluster
                cluster_key_path = None
                cluster_cert_path = None
            else:
                # Rebuild on restart to ensure the correct subject name is included in the cert SAN
                # Cert subject name needs to match the target (IP address or domain)
                self.cert_config.generate_certs(
                    address=self.address, domain=self.domain
                )
                self._copy_certs_to_cluster()

            if caddy_flag and not self.domain:
                # Update pointers to the cert and key files as stored on the cluster for Caddy to use
                base_caddy_dir = self.cert_config.CADDY_CLUSTER_DIR
                cluster_key_path = (
                    f"{base_caddy_dir}/{self.cert_config.PRIVATE_KEY_NAME}"
                )
                cluster_cert_path = f"{base_caddy_dir}/{self.cert_config.CERT_NAME}"

        # Update the cluster config on the cluster
        self.save_config_to_cluster()

        cmd = (
            CLI_RESTART_CMD
            + (" --restart-ray" if restart_ray else "")
            + (" --use-https" if https_flag else "")
            + (" --use-caddy" if caddy_flag else "")
            + (" --restart-proxy" if restart_proxy and caddy_flag else "")
            + (f" --ssl-certfile {cluster_cert_path}" if self._use_custom_certs else "")
            + (f" --ssl-keyfile {cluster_key_path}" if self._use_custom_certs else "")
            + (f" --domain {domain}" if domain else "")
            + (" --use-local-telemetry" if use_local_telemetry else "")
            + f" --port {self.server_port}"
            + f" --api-server-url {rns_client.api_server_url}"
            + (f" --default-env-name {self.default_env.name}")
            + (
                f" --conda-env {self.default_env.env_name}"
                if self.default_env.config().get("resource_subtype", None) == "CondaEnv"
                else ""
            )
        )

        status_codes = self.run(
            commands=[cmd], env=self._default_env, node=self.address
        )
        if not status_codes[0][0] == 0:
            raise ValueError(f"Failed to restart server {self.name}")

        if https_flag:
            rns_address = self.rns_address or self.name
            if not rns_address:
                raise ValueError("Cluster must have a name in order to enable HTTPS.")

            if not self.client:
                logger.debug("Reconnecting server client. Server restarted with HTTPS.")
                self.connect_server_client()

            # Refresh the client params to use HTTPS
            self.client.use_https = https_flag

        if restart_ray and len(self.ips) > 1:
            self._start_ray_workers(DEFAULT_RAY_PORT, env=self.default_env)

        self.put_resource(self.default_env)
        if default_env:
            from runhouse.resources.envs.utils import _process_env_vars

            env_vars = _process_env_vars(default_env.env_vars)
            if env_vars:
                self.call(default_env.name, "_set_env_vars", env_vars)

        return status_codes

    def stop_server(self, stop_ray: bool = True, env: Union[str, "Env"] = None):
        """Stop the RPC server.

        Args:
            stop_ray (bool): Whether to stop Ray. (Default: `True`)
            env (str or Env, optional): Specified environment to stop the server on. (Default: ``None``)
        """
        cmd = CLI_STOP_CMD if stop_ray else f"{CLI_STOP_CMD} --no-stop-ray"

        status_codes = self.run([cmd], env=env or self._default_env, stream_logs=False)
        assert status_codes[0][0] == 1

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
            run_async (bool): Run the method asynchronously and return an awaitable.
            *args: Positional arguments to pass to the method.
            **kwargs: Keyword arguments to pass to the method.

        Example:
            >>> cluster.call("my_module", "my_method", arg1, arg2, kwarg1=kwarg1)
        """
        self.check_server()
        # Note: might be single value, might be a generator!
        if self.on_this_cluster():
            method_to_call = obj_store.acall if run_async else obj_store.call
            return method_to_call(
                module_name,
                method_name,
                data={"args": args, "kwargs": kwargs},
                stream_logs=stream_logs,
                run_name=run_name,
                # remote=remote,
                serialization=None,
            )
        method_to_call = (
            self.client.acall_module_method
            if run_async
            else self.client.call_module_method
        )
        return method_to_call(
            module_name,
            method_name,
            resource_address=self.rns_address,
            stream_logs=stream_logs,
            data={"args": args, "kwargs": kwargs},
            run_name=run_name,
            remote=remote,
            save=save,
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
            self._rpc_tunnel.terminate()

    def __getstate__(self):
        """Delete non-serializable elements (e.g. thread locks) before pickling."""
        state = self.__dict__.copy()
        state["client"] = None
        state["_rpc_tunnel"] = None
        return state

    # ----------------- SSH Methods ----------------- #
    def _rsync(
        self,
        source: str,
        dest: str,
        up: bool,
        node: str = None,
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
        # Theoretically we could reuse this logic from SkyPilot which Rsyncs to all nodes in parallel:
        # https://github.com/skypilot-org/skypilot/blob/v0.4.1/sky/backends/cloud_vm_ray_backend.py#L3094
        # This is an interesting policy... by default we're syncing to all nodes if the cluster is multinode.
        # If we need to change it to be greedier we can.
        if up and (node == "all" or (len(self.ips) > 1 and not node)):
            for node in self.ips:
                self._rsync(
                    source,
                    dest,
                    up=up,
                    node=node,
                    contents=contents,
                    filter_options=filter_options,
                    stream_logs=stream_logs,
                )
            return

        from runhouse.resources.hardware.sky_ssh_runner import SkySSHRunner, SshMode

        # If no address provided explicitly use the head node address
        node = node or self.address
        # FYI, could be useful: https://github.com/gchamon/sysrsync
        if contents:
            source = source + "/" if not source.endswith("/") else source
            dest = dest + "/" if not dest.endswith("/") else dest

        # If we're already on this cluster (and node, if multinode), this is just a local rsync
        if self.on_this_cluster() and node == self.address:
            if Path(source).expanduser().resolve() == Path(dest).expanduser().resolve():
                return

            if not up:
                # If we're not uploading, we're downloading
                source, dest = dest, source

            dest = Path(dest).expanduser()
            if not dest.exists():
                dest.mkdir(parents=True, exist_ok=True)
            if not dest.is_dir():
                raise ValueError(f"Destination {dest} is not a directory.")
            dest = str(dest) + "/" if contents else str(dest)

            cmd = [
                "rsync",
                "-avz",
                source,
                dest,
            ]  # -a is archive mode, -v is verbose, -z is compress
            if filter_options:
                cmd += [filter_options]

            subprocess.run(cmd, check=True, capture_output=not stream_logs, text=True)
            return

        ssh_credentials = copy.copy(self.creds_values) or {}
        ssh_credentials.pop("ssh_host", node)
        pwd = ssh_credentials.pop("password", None)
        ssh_credentials.pop("private_key", None)
        ssh_credentials.pop("public_key", None)
        ssh_control_name = ssh_credentials.pop(
            "ssh_control_name", f"{node}:{self.ssh_port}"
        )

        runner = SkySSHRunner(
            node,
            **ssh_credentials,
            ssh_control_name=ssh_control_name,
            port=self.ssh_port,
        )
        if not pwd:
            if up:
                runner.run(
                    ["mkdir", "-p", dest],
                    stream_logs=False,
                    ssh_mode=SshMode.INTERACTIVE,
                )
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
                ssh = pexpect.spawn(ssh_command, encoding="utf-8", timeout=None)
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
            ssh = pexpect.spawn(rsync_cmd, encoding="utf-8", timeout=None)
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
        from runhouse.resources.hardware.sky_ssh_runner import SkySSHRunner, SshMode

        creds = self.creds_values
        runner = SkySSHRunner(
            ip=self.address,
            ssh_user=creds["ssh_user"],
            port=self.ssh_port,
            ssh_private_key=creds["ssh_private_key"],
        )
        subprocess.run(
            runner._ssh_base_command(ssh_mode=SshMode.INTERACTIVE, port_forward=None)
        )

    def _ping(self, timeout=5):
        ssh_call = threading.Thread(target=lambda: self.run(['echo "hello"']))
        ssh_call.start()
        ssh_call.join(timeout=timeout)
        if ssh_call.is_alive():
            raise TimeoutError("SSH call timed out")
        return True

    def _copy_certs_to_cluster(self):
        """Copy local certs to the cluster. Destination on the cluster depends on whether Caddy is enabled. This is
        to ensure that the Caddy service has the necessary access to load the certs when the service is started."""
        from runhouse import folder

        # Copy to the home directory by default
        local_certs_path = self.cert_config.key_path
        dest = self.cert_config.DEFAULT_CLUSTER_DIR
        folder(path=Path(local_certs_path).parent).to(self, path=dest)

        if self._use_caddy:
            # Move to the Caddy directory to ensure the daemon has access to the certs
            src = self.cert_config.DEFAULT_CLUSTER_DIR
            dest = self.cert_config.CADDY_CLUSTER_DIR
            self._run_commands_with_ssh(
                [
                    f"sudo mkdir -p {dest}",
                    f"sudo mv {src}/* {dest}/",
                    f"sudo rm -r {src}",
                ]
            )

        logger.debug(f"Copied local certs onto the cluster in path: {dest}")

    def run(
        self,
        commands: List[str],
        env: Union["Env", str] = None,
        stream_logs: bool = True,
        port_forward: Union[None, int, Tuple[int, int]] = None,
        require_outputs: bool = True,
        node: Optional[str] = None,
        run_name: Optional[str] = None,
        _ssh_mode: str = "interactive",  # Note, this only applies for non-password SSH
    ) -> list:
        """Run a list of shell commands on the cluster. If `run_name` is provided, the commands will be
        sent over to the cluster before being executed and a Run object will be created.

        Example:
            >>> cpu.run(["pip install numpy"])
            >>> cpu.run(["pip install numpy"], env="my_conda_env"])
            >>> cpu.run(["python script.py"], run_name="my_exp")
            >>> cpu.run(["python script.py"], node="3.89.174.234")
        """
        from runhouse.resources.envs import Env

        if isinstance(commands, str):
            commands = [commands]

        if isinstance(env, Env) and not env.name:
            env = self._default_env
        env = env or self.default_env

        if node == "all":
            res_list = []
            for node in self.ips:
                res = self.run(
                    commands=commands,
                    env=env,
                    stream_logs=stream_logs,
                    port_forward=port_forward,
                    require_outputs=require_outputs,
                    node=node,
                    run_name=run_name,
                    _ssh_mode=_ssh_mode,
                )
                res_list.append(res)
            return res_list

        # TODO [DG] suspend autostop while running

        if env and not port_forward and not node:
            env_name = (
                env
                if isinstance(env, str)
                else env.name
                if isinstance(env, Env)
                else None
            )
            return_codes = []
            for command in commands:
                ret_code = self.call(
                    env_name,
                    "_run_command",
                    command,
                    require_outputs=require_outputs,
                    stream_logs=stream_logs,
                )
                return_codes.append(ret_code)
            return return_codes

        env = _get_env_from(env)
        cmd_prefix = env._run_cmd if isinstance(env, Env) else ""

        if self.on_this_cluster():
            return_codes = []
            location_str = "locally" if not self.name else f"on {self.name}"
            for command in commands:
                command = f"{cmd_prefix} {command}" if cmd_prefix else command
                logger.info(f"Running command {location_str}: {command}")
                ret_code = run_with_logs(
                    command, stream_logs=stream_logs, require_outputs=require_outputs
                )
                return_codes.append(ret_code)
            return return_codes

        if not run_name:
            # If not creating a Run then just run the commands via SSH and return
            return self._run_commands_with_ssh(
                commands,
                cmd_prefix,
                stream_logs,
                node,
                port_forward,
                require_outputs,
                _ssh_mode=_ssh_mode,
            )

        # Create and save the Run locally
        from runhouse.resources.provenance import run

        with run(name=run_name, cmds=commands, overwrite=True) as r:
            return_codes = self._run_commands_with_ssh(
                commands,
                cmd_prefix,
                stream_logs,
                node,
                port_forward,
                require_outputs,
            )

        # Register the completed Run
        r._register_cmd_run_completion(return_codes)
        logger.debug(f"Saved Run to path: {r.folder.path}")
        return return_codes

    def _run_commands_with_ssh(
        self,
        commands: list,
        cmd_prefix: str = "",
        stream_logs: bool = True,
        node: str = None,
        port_forward: int = None,
        require_outputs: bool = True,
        _ssh_mode: str = "interactive",  # Note, this only applies for non-password SSH
    ):
        from runhouse.resources.hardware.sky_ssh_runner import SkySSHRunner, SshMode

        return_codes = []

        ssh_credentials = copy.copy(self.creds_values)
        host = ssh_credentials.pop("ssh_host", node or self.address)
        pwd = ssh_credentials.pop("password", None)
        ssh_credentials.pop("private_key", None)
        ssh_credentials.pop("public_key", None)
        ssh_control_name = ssh_credentials.pop(
            "ssh_control_name", f"{node}:{self.ssh_port}"
        )

        runner = SkySSHRunner(
            host,
            **ssh_credentials,
            ssh_control_name=ssh_control_name,
            port=self.ssh_port,
        )

        if not pwd:
            for command in commands:
                command = f"{cmd_prefix} {command}" if cmd_prefix else command
                logger.info(f"Running command on {self.name}: {command}")
                ssh_mode = (
                    SshMode.INTERACTIVE
                    if _ssh_mode == "interactive"
                    else SshMode.NON_INTERACTIVE
                    if _ssh_mode == "non_interactive"
                    else SshMode.LOGIN
                    if _ssh_mode == "login"
                    else None
                )
                if not ssh_mode:
                    raise ValueError(f"Invalid SSH mode: {_ssh_mode}.")
                ret_code = runner.run(
                    command,
                    require_outputs=require_outputs,
                    stream_logs=stream_logs,
                    port_forward=port_forward,
                    ssh_mode=ssh_mode,
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
                ssh = pexpect.spawn(ssh_command, encoding="utf-8", timeout=None)
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
        node: str = None,
        port_forward: Optional[int] = None,
        run_name: Optional[str] = None,
    ):
        """Run a list of python commands on the cluster, or a specific cluster node if its IP is provided.

        Example:
            >>> cpu.run_python(['import numpy', 'print(numpy.__version__)'])
            >>> cpu.run_python(["print('hello')"])
            >>> cpu.run_python(["print('hello')"], node="3.89.174.234")

        Note:
            Running Python commands with nested quotes can be finicky. If using nested quotes,
            try to wrap the outer quote with double quotes (") and the inner quotes with a single quote (').
        """
        cmd_prefix = "python3 -c"
        command_str = "; ".join(commands)
        command_str_repr = (
            repr(repr(command_str))[2:-2]
            if self.creds_values.get("password")
            else command_str
        )
        formatted_command = f'{cmd_prefix} "{command_str_repr}"'

        # If invoking a run as part of the python commands also return the Run object
        return_codes = self.run(
            [formatted_command],
            env=env or self._default_env,
            stream_logs=stream_logs,
            node=node,
            port_forward=port_forward,
            run_name=run_name,
        )

        return return_codes

    def sync_secrets(
        self,
        providers: Optional[List[str or "Secret"]] = None,
        env: Union[str, "Env"] = None,
    ):
        """Send secrets for the given providers.

        Args:
            providers(List[str] or None): List of providers to send secrets for.
                If `None`, all providers configured in the environment will by sent.

        Example:
            >>> cpu.sync_secrets(secrets=["aws", "lambda"])
        """
        self.check_server()
        from runhouse.resources.secrets import Secret

        env = env or self._default_env
        if isinstance(env, str):
            from runhouse.resources.envs import Env

            env = Env.from_name(env)

        secrets = []
        if providers:
            for secret in providers:
                secrets.append(
                    Secret.from_name(secret) if isinstance(secret, str) else secret
                )
        else:
            secrets = Secret.local_secrets()
            enabled_provider_secrets = Secret.extract_provider_secrets()
            secrets.update(enabled_provider_secrets)
            secrets = secrets.values()

        for secret in secrets:
            secret.to(self, env=env)

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
        # Roughly trying to follow:
        # https://towardsdatascience.com/using-jupyter-notebook-running-on-a-remote-docker-container-via-ssh-ea2c3ebb9055
        # https://docs.ray.io/en/latest/ray-core/using-ray-with-jupyter.html

        tunnel = self.ssh_tunnel(
            local_port=port_forward,
            num_ports_to_try=10,
        )
        port_fwd = tunnel.local_bind_port

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
                tunnel.terminate()
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
        self.check_server()
        self.client.get_certificate()
        logger.info(
            f"Latest TLS certificate for {self.name} saved to local path: {self.cert_config.cert_path}"
        )

    def enable_den_auth(self, flush=True):
        """Enable Den auth on the cluster."""
        self.check_server()
        if self.on_this_cluster():
            raise ValueError("Cannot toggle Den Auth live on the cluster.")
        else:
            self.den_auth = True
            self.client.set_settings({"den_auth": True, "flush_auth_cache": flush})
        return self

    def disable_den_auth(self):
        self.check_server()
        if self.on_this_cluster():
            raise ValueError("Cannot toggle Den Auth live on the cluster.")
        else:
            self.den_auth = False
            self.client.set_settings({"den_auth": False})
        return self

    def set_connection_defaults(self, **kwargs):
        if self.server_host and (
            "localhost" in self.server_host or ":" in self.server_host
        ):
            # If server_connection_type is not specified, we
            # assume we can hit the server directly via HTTP
            self.server_connection_type = (
                self.server_connection_type or ServerConnectionType.NONE
            )
            if ":" in self.server_host:
                # e.g. "localhost:23324" or <real_ip>:<custom port> (e.g. a port is already open to the server)
                self.server_host, self.client_port = self.server_host.split(":")
                kwargs["client_port"] = self.client_port

        self.server_connection_type = self.server_connection_type or (
            ServerConnectionType.TLS
            if self.ssl_certfile or self.ssl_keyfile
            else ServerConnectionType.SSH
        )

        if self.server_port is None:
            if self.server_connection_type == ServerConnectionType.TLS:
                self.server_port = DEFAULT_HTTPS_PORT
            elif self.server_connection_type == ServerConnectionType.NONE:
                self.server_port = DEFAULT_HTTP_PORT
            else:
                self.server_port = DEFAULT_SERVER_PORT

        if self.name in RESERVED_SYSTEM_NAMES:
            raise ValueError(
                f"Cluster name {self.name} is a reserved name. Please use a different name which is not one of "
                f"{RESERVED_SYSTEM_NAMES}."
            )

    def share(
        self,
        users: Union[str, List[str]] = None,
        access_level: Union[ResourceAccess, str] = ResourceAccess.READ,
        visibility: Optional[Union[ResourceVisibility, str]] = None,
        notify_users: bool = True,
        headers: Optional[Dict] = None,
    ) -> Tuple[Dict[str, ResourceAccess], Dict[str, ResourceAccess]]:

        # save cluster and creds if not saved
        self.save()

        # share creds
        logger.info(
            "Sharing cluster credentials, which enables the recipient to SSH into the cluster."
        )
        self._creds.share(
            users=users,
            access_level=access_level,
            visibility=visibility,
            notify_users=notify_users,
            headers=headers,
        )

        # share cluster
        return super().share(
            users=users,
            access_level=access_level,
            visibility=visibility,
            notify_users=notify_users,
            headers=headers,
        )

    @classmethod
    def _check_for_child_configs(cls, config: dict):
        """Overload by child resources to load any resources they hold internally."""
        from runhouse.resources.envs.env import Env
        from runhouse.resources.secrets.secret import Secret
        from runhouse.resources.secrets.utils import load_config, setup_cluster_creds

        creds = config.pop("creds", None) or config.pop("ssh_creds", None)

        if isinstance(creds, str):
            creds = Secret.from_config(config=load_config(name=creds))
        elif isinstance(creds, dict):
            if "name" in creds.keys():
                creds = Secret.from_config(creds)
            else:
                creds = setup_cluster_creds(creds, config["name"])
        config["creds"] = creds

        default_env = config.pop("default_env", None)
        if isinstance(default_env, str):
            default_env = Env.from_name(default_env, _resolve_children=False)
        elif isinstance(default_env, dict):
            default_env = Env.from_config(default_env)
        if default_env:
            config["default_env"] = default_env

        return config
