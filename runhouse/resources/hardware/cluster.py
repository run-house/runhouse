import contextlib
import copy
import importlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import requests
import yaml

from runhouse.resources.hardware.utils import (
    _setup_creds_from_dict,
    _setup_default_creds,
    ClusterStatus,
    get_clusters_from_den,
    get_running_and_not_running_clusters,
    get_unsaved_live_clusters,
    parse_filters,
)

from runhouse.resources.images import ImageSetupStepType

from runhouse.rns.utils.api import ResourceAccess, ResourceVisibility
from runhouse.servers.http.certs import TLSCertConfig
from runhouse.utils import (
    _process_env_vars,
    conda_env_cmd,
    create_conda_env_on_cluster,
    find_locally_installed_version,
    install_conda,
    locate_working_dir,
    run_command_with_password_login,
    run_setup_command,
    ThreadWithException,
)

# Filter out DeprecationWarnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import requests.exceptions

from runhouse.constants import (
    CLI_RESTART_CMD,
    CLI_START_CMD,
    CLI_STOP_CMD,
    CLUSTER_CONFIG_PATH,
    DEFAULT_DASK_PORT,
    DEFAULT_HTTP_PORT,
    DEFAULT_HTTPS_PORT,
    DEFAULT_PROCESS_NAME,
    DEFAULT_RAY_PORT,
    DEFAULT_SERVER_PORT,
    DEFAULT_STATUS_CHECK_INTERVAL,
    LOCALHOST,
    NUM_PORTS_TO_TRY,
    RESERVED_SYSTEM_NAMES,
)
from runhouse.globals import configs, obj_store, rns_client
from runhouse.logger import get_logger

from runhouse.resources.envs.utils import _get_env_from
from runhouse.resources.hardware.utils import (
    _current_cluster,
    _run_ssh_command,
    ServerConnectionType,
)
from runhouse.resources.resource import Resource

from runhouse.servers.http import HTTPClient
from runhouse.servers.http.http_utils import CreateProcessParams

logger = get_logger(__name__)


class Cluster(Resource):
    RESOURCE_TYPE = "cluster"
    REQUEST_TIMEOUT = 5  # seconds

    DEFAULT_SSH_PORT = 22
    DEFAULT_PROCESS_NAME = DEFAULT_PROCESS_NAME

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
        ssh_properties: Dict = None,
        den_auth: bool = False,
        dryrun: bool = False,
        skip_creds: bool = False,
        image: Optional["Image"] = None,
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
        if default_env:
            raise ValueError(
                "`default_env` argument has been deprecated. Please refer to the Image class for an updated "
                "approach to set up a default env."
            )

        super().__init__(name=name, dryrun=dryrun)

        self._rpc_tunnel = None

        self._ips = ips
        self._http_client = None
        self.den_auth = den_auth or False
        self.cert_config = TLSCertConfig(cert_path=ssl_certfile, key_path=ssl_keyfile)

        self.ssl_certfile = ssl_certfile
        self.ssl_keyfile = ssl_keyfile
        self.server_connection_type = server_connection_type
        self.server_port = server_port
        self.client_port = client_port
        self.ssh_port = ssh_port or self.DEFAULT_SSH_PORT
        self.ssh_properties = ssh_properties or {}
        self.server_host = server_host
        self.domain = domain
        self.compute_properties = {}

        self.reqs = []

        if skip_creds and not creds:
            self._creds = None
        else:
            self._setup_creds(creds)
        self.image = image

    @property
    def ips(self):
        return self._ips

    @property
    def internal_ips(self):
        return self.ips

    @property
    def head_ip(self):
        return self.ips[0] if self.ips else None

    @property
    def address(self):
        return self.head_ip

    @property
    def client(self):
        def check_connect_server():
            connect_call = threading.Thread(
                target=self.connect_server_client, kwargs={"force_reconnect": True}
            )
            connect_call.start()
            connect_call.join(timeout=5)
            if connect_call.is_alive():
                return False
            return True

        if not self._http_client and not check_connect_server():
            if self.__class__.__name__ == "OnDemandCluster":
                self._update_from_sky_status(dryrun=False)
            if not self._ping(retry=False):
                raise ConnectionError(
                    f"Could not reach {self.name} {self.head_ip}. Is cluster up?"
                )
            if not check_connect_server():
                raise ConnectionError(
                    f"Timed out trying to form connection for cluster {self.name}."
                )

        if not self._http_client:
            raise ConnectionError(
                f"Error occurred trying to form connection for cluster {self.name}."
            )

        try:
            self._http_client.check_server()
        except (
            requests.exceptions.ConnectionError,
            requests.exceptions.ReadTimeout,
            requests.exceptions.ChunkedEncodingError,
            ValueError,
        ) as e:
            raise ConnectionError(f"Check server failed: {e}.")
        return self._http_client

    @property
    def creds_values(self) -> Dict:
        if not self._creds:
            return {}

        return {**self._creds.values, **self.ssh_properties}

    @property
    def docker_user(self) -> Optional[str]:
        return None

    def save_config_to_cluster(
        self,
        node: str = None,
    ):
        config = self.config(condensed=False)

        # popping creds, because we don't want to save secret creds on the cluster.
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

        Args:
            name (str, optional): Name to save the cluster as, if different from its existing name. (Default: ``None``)
            overwrite (bool, optional): Whether to overwrite the existing saved resource, if it exists.
                (Default: ``True``)
            folder (str, optional): Folder to save the config in, if saving locally. If None and saving locally,
                will be saved in the ``~/.rh`` directory. (Default: ``None``)
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
        # Den address. We need to update the name
        # used in the config on the cluster so that
        # self.on_this_cluster() will still work as expected.
        if on_this_cluster:
            obj_store.set_cluster_config_value("name", self.rns_address)
        elif self._http_client:
            self.call_client_method("set_cluster_name", self.rns_address)

        return self

    def delete_configs(self, delete_creds: bool = False):
        """Delete configs for the cluster"""
        if delete_creds and self._creds:
            logger.debug(
                f"Attempting to delete creds associated with cluster {self.name}"
            )
            rns_client.delete_configs(self._creds)

        super().delete_configs()

    def _setup_creds(self, ssh_creds: Union[Dict, "Secret", str]):
        """Setup cluster credentials from user provided ssh_creds"""
        from runhouse.resources.secrets import Secret

        if isinstance(ssh_creds, Secret):
            self._creds = ssh_creds
            return
        elif isinstance(ssh_creds, str):
            self._creds = Secret.from_name(ssh_creds)
            return

        if not ssh_creds:
            from runhouse.resources.hardware.on_demand_cluster import OnDemandCluster

            cluster_subtype = (
                "OnDemandCluster" if isinstance(self, OnDemandCluster) else "Cluster"
            )
            self._creds = _setup_default_creds(cluster_subtype)
        elif isinstance(ssh_creds, Dict):
            creds, ssh_properties = _setup_creds_from_dict(ssh_creds, self.name)
            self._creds = creds
            self.ssh_properties = ssh_properties

    def _should_save_creds(self, folder: str = None) -> bool:
        """Checks whether to save the creds associated with the cluster.
        Only do so as part of the save() if the user making the call is the creator"""

        from runhouse.resources.secrets import Secret

        local_default_folder = folder or configs.username
        # if not self.rns_address => we are saving the cluster first time in den
        # else, need to check if the username of the current saver is included in the rns_address.
        should_save_creds = (
            (not self.rns_address or local_default_folder in self.rns_address)
            and self._creds
            and isinstance(self._creds, Secret)
        )

        if should_save_creds:
            # update secret name if it already exists in den w/ different config, avoid overwriting
            try:
                if self._creds.rns_address:
                    saved_secret = Secret.from_name(self._creds.rns_address)
                    if saved_secret and (
                        saved_secret.config(values=False)
                        != self._creds.config(values=False)
                        or (saved_secret.values != self._creds.values)
                    ):
                        new_creds = copy.deepcopy(self._creds)
                        new_creds.name = f"{self.name}-ssh-secret"
                        self._creds = new_creds
            except ValueError:
                pass

        return should_save_creds

    def _save_sub_resources(self, folder: str = None):

        creds_folder = (
            folder if (not self._creds or not self._creds._rns_folder) else None
        )
        if self._should_save_creds(creds_folder):
            # Only automatically set the creds folder if it doesn't have one yet
            # allows for org SSH keys to be associated with the user.
            self._creds.save(folder=creds_folder)

    @classmethod
    def from_name(
        cls,
        name: str,
        load_from_den: bool = True,
        dryrun: bool = False,
        _alt_options: Dict = None,
        _resolve_children: bool = True,
    ):
        cluster = super().from_name(
            name=name,
            load_from_den=load_from_den,
            dryrun=dryrun,
            _alt_options=_alt_options,
            _resolve_children=_resolve_children,
        )
        if cluster and cluster._creds and not dryrun:
            from runhouse.resources.secrets import Secret
            from runhouse.resources.secrets.provider_secrets.ssh_secret import SSHSecret

            if isinstance(cluster._creds, SSHSecret):
                cluster._creds.write()
            elif isinstance(cluster._creds, Secret):
                # old version of cluster creds or password only
                private_key_path = cluster._creds.values.get("ssh_private_key")
                if private_key_path:
                    ssh_creds = SSHSecret(
                        path=private_key_path,
                        values=cluster._creds.values,
                    )
                    ssh_creds.write()

        return cluster

    @classmethod
    def from_config(
        cls, config: Dict, dryrun: bool = False, _resolve_children: bool = True
    ):
        resource_subtype = config.get("resource_subtype")
        if _resolve_children:
            config = cls._check_for_child_configs(config)

        if resource_subtype == "Cluster":
            return Cluster(**config, dryrun=dryrun)
        elif resource_subtype == "OnDemandCluster":
            from .on_demand_cluster import OnDemandCluster

            return OnDemandCluster(**config, dryrun=dryrun)
        else:
            raise ValueError(f"Unknown cluster type {resource_subtype}")

    def config(self, condensed: bool = True):
        config = super().config(condensed)
        if config.get("resource_subtype") == "Cluster":
            config["ips"] = self._ips
        self.save_attrs_to_config(
            config,
            [
                "server_port",
                "server_host",
                "server_connection_type",
                "domain",
                "den_auth",
                "ssh_port",
                "client_port",
                "ssh_properties",
            ],
        )
        creds = (
            self._resource_string_for_subconfig(self._creds, condensed)
            if hasattr(self, "_creds") and self._creds
            else None
        )
        if creds:
            if "loaded_secret_" in creds:
                # user A shares cluster with user B, with "write" permissions. If user B will save the cluster to Den, we
                # would NOT like that the loaded secret will overwrite the original secret that was created and shared by
                # user A.
                creds = creds.replace("loaded_secret_", "")
            config["creds"] = creds

        config["api_server_url"] = rns_client.api_server_url

        if self._use_custom_certs:
            config["ssl_certfile"] = self.cert_config.cert_path
            config["ssl_keyfile"] = self.cert_config.key_path

        return config

    def endpoint(self, external: bool = False):
        """Endpoint for the cluster's Daemon server.

        Args:
            external (bool, optional): If ``True``, will only return the external url, and will return ``None``
                otherwise (e.g. if a tunnel is required). If set to ``False``, will either return the external url
                if it exists, or will set up the connection (based on connection_type) and return the internal url
                (including the local connected port rather than the sever port). If cluster is not up, returns
                `None``. (Default: ``False``)
        """
        if not self.head_ip or self.on_this_cluster():
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

        if self.server_connection_type == ServerConnectionType.SSH:
            self.client.check_server()
            return f"http://{LOCALHOST}:{client_port}"

    def _client(self):
        if self.on_this_cluster():
            # Previously (before calling within the same cluster worked) returned None
            return obj_store
        if not self._http_client:
            self.client.check_server()
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

        return self.head_ip

    @property
    def is_shared(self) -> bool:
        from runhouse import Secret

        ssh_creds = self.creds_values
        if not ssh_creds:
            return False

        ssh_private_key = ssh_creds.get("ssh_private_key")
        if ssh_private_key:
            ssh_private_key_path = Path(ssh_private_key).expanduser()
            secrets_base_dir = Path(Secret.DEFAULT_DIR).expanduser()

            # Check if the key path is saved down in the local .rh directory, which we only do for shared credentials
            if str(ssh_private_key_path).startswith(str(secrets_base_dir)):
                return True
            return f"{self._creds.name}/" in ssh_private_key
        return False

    def _command_runner(
        self, node: Optional[str] = None, use_docker_exec: Optional[bool] = False
    ) -> "CommandRunner":
        from runhouse.resources.hardware.sky_command_runner import (
            SkyKubernetesRunner,
            SkySSHRunner,
        )

        if node == "all":
            raise ValueError(
                "CommandRunner can only be instantiated for individual nodes"
            )

        node = node or self.head_ip

        if self.compute_properties.get("cloud") == "kubernetes":
            namespace = self.compute_properties.get("namespace", None)
            node_idx = self.ips.index(node)
            pod_name = self.compute_properties.get("pod_names", None)[node_idx]

            runner = SkyKubernetesRunner(
                (namespace, pod_name), docker_user=self.docker_user
            )
        else:
            ssh_credentials = copy.copy(self.creds_values) or {}
            ssh_control_name = ssh_credentials.pop(
                "ssh_control_name", f"{node}:{self.ssh_port}"
            )

            runner = SkySSHRunner(
                (node, self.ssh_port),
                ssh_user=ssh_credentials.get("ssh_user"),
                ssh_private_key=ssh_credentials.get("ssh_private_key"),
                ssh_proxy_command=ssh_credentials.get("ssh_proxy_command"),
                ssh_control_name=ssh_control_name,
                docker_user=self.docker_user if not use_docker_exec else None,
                use_docker_exec=use_docker_exec,
            )

        return runner

    def is_up(self) -> bool:
        """Check if the cluster is up.

        Example:
            >>> rh.cluster("rh-cpu").is_up()
        """
        return self.on_this_cluster() or self._ping()

    def _is_server_up(self) -> bool:
        try:
            self.client.check_server()
            return True
        except ValueError:
            return False
        except ConnectionError:
            return False

    def up_if_not(self, verbose: bool = True):
        """Bring up the cluster if it is not up. No-op if cluster is already up.
        This only applies to on-demand clusters, and has no effect on self-managed clusters.

        Args:
            verbose (bool, optional): Whether to stream logs from Den if the cluster is being launched. Only
                relevant if launching via Den. (Default: `True`)

        Example:
            >>> rh.cluster("rh-cpu").up_if_not()
        """
        if not self.is_up():
            self.up(verbose=verbose, force=False)
        return self

    def up(self, verbose: bool = True, force: bool = False):
        raise NotImplementedError(
            f"Cluster <{self.name}> does not have an up method. It must be brought up manually."
        )

    def keep_warm(self):
        logger.info(
            f"cluster.keep_warm will have no effect on self-managed cluster {self.name}."
        )
        return self

    def _sync_image_to_cluster(self):
        """
        Image stuff that needs to happen over SSH because the daemon won't be up yet, so we can't
        use the HTTP client.
        """
        env_vars = {}
        log_level = os.getenv("RH_LOG_LEVEL")
        if log_level:
            # add log level to the default env to ensure it gets set on the cluster when the server is restarted
            env_vars["RH_LOG_LEVEL"] = log_level
            logger.info(f"Using log level {log_level} on cluster's default env")

        if not configs.observability_enabled:
            env_vars["disable_observability"] = "True"
            logger.info("Disabling observability on the cluster")

        if not self.image:
            return [], env_vars  # secrets, env vars

        if self.image.image_id and self.config().get("resource_subtype") == "Cluster":
            logger.error(
                "``image_id`` is only supported for OnDemandCluster, not static Clusters."
            )

        logger.info(f"Syncing default image {self.image} to cluster.")

        secrets_to_sync = []

        for setup_step in self.image.setup_steps:
            for node in self.ips:
                if setup_step.step_type == ImageSetupStepType.SETUP_CONDA_ENV:
                    self.create_conda_env(
                        env_name=setup_step.kwargs.get("conda_env_name"),
                        conda_yaml=setup_step.kwargs.get("conda_yaml"),
                    )
                elif setup_step.step_type == ImageSetupStepType.PACKAGES:
                    self.install_packages(
                        setup_step.kwargs.get("reqs"),
                        conda_env_name=setup_step.kwargs.get("conda_env_name"),
                        node=node,
                    )
                elif setup_step.step_type == ImageSetupStepType.CMD_RUN:
                    command = setup_step.kwargs.get("cmd")
                    if setup_step.conda_env_name:
                        command = conda_env_cmd(command, setup_step.conda_env_name)
                    run_setup_command(
                        cmd=command,
                        cluster=self,
                        env_vars=env_vars,
                        stream_logs=True,
                        node=node,
                    )
                elif setup_step.step_type == ImageSetupStepType.SYNC_SECRETS:
                    secrets_to_sync += setup_step.kwargs.get("providers")
                elif setup_step.step_type == ImageSetupStepType.RSYNC:
                    self.rsync(
                        source=setup_step.kwargs.get("source"),
                        dest=setup_step.kwargs.get("dest"),
                        node=node,
                        up=True,
                        contents=setup_step.kwargs.get("contents"),
                        filter_options=setup_step.kwargs.get("filter_options"),
                    )
                elif setup_step.step_type == ImageSetupStepType.SET_ENV_VARS:
                    image_env_vars = _process_env_vars(
                        setup_step.kwargs.get("env_vars")
                    )
                    env_vars.update(image_env_vars)

        return secrets_to_sync, env_vars

    def _sync_runhouse_to_cluster(
        self,
        _install_url: Optional[str] = None,
        local_rh_package_path: Optional[Path] = None,
    ):
        if self.on_this_cluster():
            return

        if not self.ips:
            raise ValueError(f"No IPs set for cluster <{self.name}>. Is it up?")

        remote_ray_version_call = self.run(["ray --version"], node="all")
        ray_installed_remotely = remote_ray_version_call[0][0][0] == 0
        if not ray_installed_remotely:
            local_ray_version = find_locally_installed_version("ray")

            # if Ray is installed locally, install the same version on the cluster
            if local_ray_version:
                ray_install_cmd = f"python3 -m pip install ray=={local_ray_version}"
                self.run([ray_install_cmd], node="all", stream_logs=True)

        # If local_rh_package_path is provided, install the package from the local path
        if local_rh_package_path:
            local_rh_package_path = local_rh_package_path.parent
            dest_path = f"~/{local_rh_package_path.name}"

            self.rsync(
                source=str(local_rh_package_path),
                dest=dest_path,
                node="all",
                up=True,
                contents=True,
                filter_options="- docs/",
            )
            rh_install_cmd = f"python3 -m pip install {dest_path}"

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
                stream_logs=True,
            )

            if status_codes[0][0] != 0:
                raise ValueError(
                    f"Error installing runhouse on cluster <{self.name}> node <{node}>"
                )

    def install_packages(
        self,
        reqs: List[Union["Package", str]],
        node: Optional[str] = None,
        conda_env_name: Optional[str] = None,
    ):
        """Install the given packages on the cluster.

        Args:
            reqs (List[Package or str]): List of packages to install on cluster and env.
            env (Env or str): Environment to install package on. If left empty, defaults to base environment.
                (Default: ``None``)

        Example:
            >>> cluster.install_packages(reqs=["accelerate", "diffusers"])
            >>> cluster.install_packages(reqs=["accelerate", "diffusers"], env="my_conda_env")
        """
        for req in reqs:
            if not node:
                self.install_package(req, conda_env_name=conda_env_name)
            else:
                self.install_package_over_ssh(
                    req, node=node, conda_env_name=conda_env_name
                )

    def get(self, key: str, default: Any = None, remote=False):
        """Get the result for a given key from the cluster's object store.

        Args:
            key (str): Key to get from the cluster's object store.
            default (Any, optional): What to return if the key is not found. To raise an error, pass in
                ``KeyError``. (Default: None)
            remote (bool, optional): Whether to get the remote object, rather than the object in full.
                (Default: ``False``)
        """
        if self.on_this_cluster():
            return obj_store.get(key, default=default, remote=remote)
        try:
            res = self.call_client_method(
                "get",
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

    def put(self, key: str, obj: Any, env: str = None):
        """Put the given object on the cluster's object store at the given key.

        Args:
            key (str): Key to assign the object in the object store.
            obj (Any): Object to put in the object store
            env (str, optional): Env of the object store to put the object in. (Default: ``None``)
        """
        if self.on_this_cluster():
            return obj_store.put(key, obj, env=env)
        return self.call_client_method(
            "put_object", key, obj, env=env or DEFAULT_PROCESS_NAME
        )

    def put_resource(
        self,
        resource: Resource,
        state: Dict = None,
        dryrun: bool = False,
        process: Optional[str] = None,
    ):
        """Put the given resource on the cluster's object store. Returns the key (important if name is not set).

        Args:
            resource (Resource): Key to assign the object in the object store.
            state (Dict, optional): Dict of resource attributes to override. (Default: ``False``)
            dryrun (bool, optional): Whether to put the resource in dryrun mode or not. (Default: ``False``)
            env (str, optional): Env of the object store to put the object in. (Default: ``None``)
        """
        if resource.RESOURCE_TYPE == "env" and not resource.name:
            # TODO - should this just throw an error?
            resource.name = DEFAULT_PROCESS_NAME

        # Logic to get env_name from different ways env can be provided
        env_name = process or (
            resource.process
            if hasattr(resource, "process")
            else resource.name or resource.env_name
            if resource.RESOURCE_TYPE == "env"
            else DEFAULT_PROCESS_NAME
        )

        # Env name could somehow be a full length `username/base_env`, trim it down to just the env name
        if env_name:
            env_name = env_name.split("/")[-1]

        state = state or {}
        if self.on_this_cluster():
            data = (resource.config(condensed=False), state, dryrun)
            return obj_store.put_resource(serialized_data=data, env_name=env_name)
        return self.call_client_method(
            "put_resource",
            resource,
            state=state or {},
            env_name=env_name,
            dryrun=dryrun,
        )

    def rename(self, old_key: str, new_key: str):
        """Rename a key in the cluster's object store.

        Args:
            old_key (str): Original key to rename.
            new_key (str): Name to reassign the object.
        """
        if self.on_this_cluster():
            return obj_store.rename(old_key, new_key)
        return self.call_client_method("rename_object", old_key, new_key)

    def keys(self, env: str = None):
        """List all keys in the cluster's object store.

        Args:
            env (str, optional): Env in which to list out the keys for.
        """
        if self.on_this_cluster():
            return obj_store.keys()
        res = self.call_client_method("keys", env=env)
        return res

    def delete(self, keys: Union[None, str, List[str]]):
        """Delete the given items from the cluster's object store. To delete all items, use `cluster.clear()`

        Args:
            keys (str or List[str]): key or list of keys to delete from the object store.
        """
        if isinstance(keys, str):
            keys = [keys]
        if self.on_this_cluster():
            return obj_store.delete(keys)
        return self.call_client_method("delete", keys)

    def clear(self):
        """Clear the cluster's object store."""
        if self.on_this_cluster():
            return obj_store.clear()
        return self.call_client_method("delete")

    def on_this_cluster(self):
        """Whether this function is being called on the same cluster."""
        config = _current_cluster("config")
        return config is not None and config.get("name") == (
            self.rns_address or self.name
        )

    # ----------------- RPC Methods ----------------- #

    def call_client_method(self, client_method_name, *args, **kwargs):
        method = getattr(self.client, client_method_name)
        try:
            return method(*args, **kwargs)
        except (ConnectionError, requests.exceptions.ConnectionError):
            try:
                self._http_client = None
                method = getattr(self.client, client_method_name)
            except:
                raise ConnectionError("Could not connect to Runhouse server.")

            return method(*args, **kwargs)
        except Exception as e:
            raise e

    def connect_tunnel(self, force_reconnect=False):
        if self._rpc_tunnel and force_reconnect:
            self._rpc_tunnel.terminate()
            self._rpc_tunnel = None

        if not self._rpc_tunnel:
            self._rpc_tunnel = self.ssh_tunnel(
                local_port=self.server_port,
                remote_port=self.server_port,
                num_ports_to_try=NUM_PORTS_TO_TRY,
            )

    def connect_server_client(self, force_reconnect=False):
        if not self.ips:
            raise ValueError(f"No IPs set for cluster <{self.name}>. Is it up?")

        if self.server_connection_type == ServerConnectionType.SSH:
            # For a password cluster, the 'ssh_tunnel' command assumes a Control Master is already set up with
            # an authenticated password.
            # TODO: I wonder if this authentication ever goes dry, and our SSH tunnel would need to be
            # re-established, would require a password, and then fail. We should really figure out how to
            # authenticate with a password in the SSH tunnel command. But, this is a fine hack for now.
            if self.creds_values.get("password") is not None:
                self._run_commands_with_runner(
                    ["echo 'Initiating password connection.'"]
                )

            # Case 1: Server connection requires SSH tunnel
            self.connect_tunnel(force_reconnect=force_reconnect)
            self.client_port = self._rpc_tunnel.local_bind_port

            # Connecting to localhost because it's tunneled into the server at the specified port.
            # As long as the tunnel was initialized,
            # self.client_port has been set to the correct port
            self._http_client = HTTPClient(
                host=LOCALHOST,
                port=self.client_port,
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

            self._http_client = HTTPClient(
                host=self.server_address,
                port=self.client_port,
                cert_path=cert_path,
                use_https=self._use_https,
                resource_address=self.rns_address,
                system=self,
            )

    def status(self, send_to_den: bool = False):
        """Load the status of the Runhouse daemon running on a cluster.

        Args:
            send_to_den (bool, optional): Whether to send and update the status in Den. Only applies to
                clusters that are saved to Den. (Default: ``False``)
        """

        # Note: If running outside a local cluster need to include a resource address to construct the cluster subtoken
        # Allow for specifying a resource address explicitly in case the resource has no rns address yet
        if self.on_this_cluster():
            status, den_resp_status_code = obj_store.status(send_to_den=send_to_den)
        else:
            status, den_resp_status_code = self.call_client_method(
                "status",
                send_to_den=send_to_den,
            )

        if send_to_den:
            if den_resp_status_code == 404:
                logger.info(
                    "Cluster has not yet been saved to Den, cannot update status or logs."
                )

            elif den_resp_status_code != 200:
                logger.warning("Failed to send cluster status to Den")

        if not configs.observability_enabled and status.get("env_servlet_processes"):
            logger.warning(
                "Cluster observability is disabled. Metrics are stale and will "
                "no longer be collected. To re-enable observability, please "
                "run `rh.configs.enable_observability()` and restart the server (`cluster.restart_server()`)."
            )

        return status

    def ssh_tunnel(
        self, local_port, remote_port=None, num_ports_to_try: int = 0
    ) -> "SshTunnel":
        from runhouse.resources.hardware.ssh_tunnel import ssh_tunnel

        cloud = self.compute_properties.get("cloud")
        return ssh_tunnel(
            address=self.head_ip,
            ssh_creds=self.creds_values,
            docker_user=self.docker_user,
            local_port=local_port,
            ssh_port=self.ssh_port,
            remote_port=remote_port,
            num_ports_to_try=num_ports_to_try,
            cloud=cloud,
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

    def _start_ray_workers(self, ray_port, env_vars):
        internal_head_ip = self.internal_ips[0]
        worker_ips = self.ips[
            1:
        ]  # Using external worker address here because we're running from local

        for host in worker_ips:
            logger.info(
                f"Starting Ray on worker {host} with head node at {internal_head_ip}:{ray_port}."
            )
            run_setup_command(
                cmd=f"ray start --address={internal_head_ip}:{ray_port} --disable-usage-stats",
                cluster=self,
                env_vars=env_vars,
                node=host,
                stream_logs=True,
            )

    def _start_or_restart_helper(
        self,
        base_cli_cmd: str,
        _rh_install_url: str = None,
        resync_rh: Optional[bool] = None,
        restart_ray: bool = True,
        restart_proxy: bool = False,
    ):
        from runhouse.resources.envs import Env

        image_secrets, image_env_vars = self._sync_image_to_cluster()

        # If resync_rh is not explicitly False, check if Runhouse is installed editable
        local_rh_package_path = None
        if resync_rh is not False:
            local_rh_package_path = Path(
                importlib.util.find_spec("runhouse").origin
            ).parent

            installed_editable_locally = (
                not _rh_install_url
                and local_rh_package_path.parent.name == "runhouse"
                and (local_rh_package_path.parent / "setup.py").exists()
            )

            if installed_editable_locally:
                logger.debug("Runhouse is installed locally in editable mode.")
                resync_rh = True
            else:
                # We only want this to be set if it was installed editable locally
                local_rh_package_path = None

        # If resync_rh is still not confirmed to happen, check if Runhouse is installed on the cluster
        if resync_rh is None:
            return_codes = self.run(["runhouse --version"], node="all")
            if return_codes[0][0][0] != 0:
                logger.debug("Runhouse is not installed on the cluster.")
                resync_rh = True

        if resync_rh:
            self._sync_runhouse_to_cluster(
                _install_url=_rh_install_url,
                local_rh_package_path=local_rh_package_path,
            )
            logger.debug("Finished syncing Runhouse to cluster.")

        https_flag = self._use_https
        caddy_flag = self._use_caddy
        domain = self.domain

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
                    address=self.head_ip, domain=self.domain
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

        # Save a limited version of the local ~/.rh config to the cluster with the user's
        # if such does not exist on the cluster
        if rns_client.token:
            user_config = {
                "token": rns_client.cluster_token(resource_address=rns_client.username),
                "username": rns_client.username,
                "default_folder": rns_client.default_folder,
            }

            yaml_path = Path(f"{tempfile.mkdtemp()}/config.yaml")
            with open(yaml_path, "w") as config_file:
                yaml.safe_dump(user_config, config_file)

            try:
                self.rsync(
                    source=yaml_path,
                    dest="~/.rh/",
                    up=True,
                    ignore_existing=True,
                )
                logger.debug("saved config.yaml on the cluster")
            finally:
                shutil.rmtree(yaml_path.parent)

        restart_cmd = (
            base_cli_cmd
            + (" --restart-ray" if restart_ray else "")
            + (" --use-https" if https_flag else "")
            + (" --use-caddy" if caddy_flag else "")
            + (" --restart-proxy" if restart_proxy and caddy_flag else "")
            + (f" --ssl-certfile {cluster_cert_path}" if self._use_custom_certs else "")
            + (f" --ssl-keyfile {cluster_key_path}" if self._use_custom_certs else "")
            + (f" --domain {domain}" if domain else "")
            + f" --port {self.server_port}"
            + f" --api-server-url {rns_client.api_server_url}"
            + (
                f" --conda-env {self.image.conda_env_name}"
                if self.image and self.image.conda_env_name
                else ""
            )
            + " --from-python"
        )

        if self.image and self.image.conda_env_name:
            restart_cmd = conda_env_cmd(restart_cmd, self.image.conda_env_name)
        status_codes = run_setup_command(
            cmd=restart_cmd,
            cluster=self,
            env_vars=image_env_vars,
            stream_logs=True,
            node=self.head_ip,
        )

        if not status_codes[0] == 0:
            raise ValueError(f"Failed to restart server {self.name}")

        if https_flag:
            rns_address = self.rns_address or self.name
            if not rns_address:
                raise ValueError("Cluster must have a name in order to enable HTTPS.")

            if not self._http_client:
                logger.debug("Reconnecting server client. Server restarted with HTTPS.")
                self.connect_server_client()

            # Refresh the client params to use HTTPS
            self.client.use_https = https_flag

        if restart_ray and len(self.ips) > 1:
            self._start_ray_workers(DEFAULT_RAY_PORT, env_vars=image_env_vars)

        self.put_resource(Env(name=DEFAULT_PROCESS_NAME))

        if image_secrets:
            self.sync_secrets(image_secrets)

        return status_codes

    def restart_server(
        self,
        _rh_install_url: str = None,
        resync_rh: Optional[bool] = None,
        restart_ray: bool = True,
        restart_proxy: bool = False,
    ):
        """Restart the RPC server.

        Args:
            resync_rh (bool): Whether to Resync runhouse. If ``False`` will not resync Runhouse onto the cluster.
                If ``None``, will sync if Runhouse is not installed on the cluster or if locally it is installed
                as editable. (Default: ``None``)
            restart_ray (bool): Whether to restart Ray. (Default: ``True``)
            restart_proxy (bool): Whether to restart Caddy on the cluster, if configured. (Default: ``False``)

        Example:
            >>> rh.cluster("rh-cpu").restart_server()
        """
        logger.info(f"Restarting Runhouse API server on {self.name}.")

        return self._start_or_restart_helper(
            base_cli_cmd=CLI_RESTART_CMD,
            _rh_install_url=_rh_install_url,
            resync_rh=resync_rh,
            restart_ray=restart_ray,
            restart_proxy=restart_proxy,
        )

    def start_server(
        self,
        _rh_install_url: str = None,
        resync_rh: Optional[bool] = None,
        restart_ray: bool = True,
        restart_proxy: bool = False,
    ):
        """Restart the RPC server.

        Args:
            resync_rh (bool): Whether to Resync runhouse. If ``False`` will not resync Runhouse onto the cluster.
                If ``None``, will sync if Runhouse is not installed on the cluster or if locally it is installed
                as editable. (Default: ``None``)
            restart_ray (bool): Whether to restart Ray. (Default: ``True``)
            restart_proxy (bool): Whether to restart Caddy on the cluster, if configured. (Default: ``False``)

        Example:
            >>> rh.cluster("rh-cpu").start_server()
        """
        logger.debug(f"Starting Runhouse API server on {self.name}.")

        return self._start_or_restart_helper(
            base_cli_cmd=CLI_START_CMD,
            _rh_install_url=_rh_install_url,
            resync_rh=resync_rh,
            restart_ray=restart_ray,
            restart_proxy=restart_proxy,
        )

    def stop_server(
        self,
        stop_ray: bool = False,
        env: Union[str, "Env"] = None,
        cleanup_actors: bool = True,
    ):
        """Stop the RPC server.

        Args:
            stop_ray (bool, optional): Whether to stop Ray. (Default: `True`)
            env (str or Env, optional): Specified environment to stop the server on. (Default: ``None``)
            cleanup_actors (bool, optional): Whether to kill all Ray actors. (Default: ``True``)
        """
        cmd = CLI_STOP_CMD
        if stop_ray:
            cmd = cmd + " --stop-ray"
        if not cleanup_actors:
            cmd = cmd + " --no-cleanup-actors"

        status_codes = self.run([cmd], require_outputs=False)
        assert status_codes[0] == 0

    @contextlib.contextmanager
    def pause_autostop(self):
        """Context manager to temporarily pause autostop. Only for OnDemand clusters. There is no autostop
        for static clusters."""
        pass

    def call(
        self,
        module_name: str,
        method_name: str,
        *args,
        stream_logs: bool = True,
        run_name: str = None,
        remote: bool = False,
        run_async: bool = False,
        save: bool = False,
        **kwargs,
    ):
        """Call a method on a module that is in the cluster's object store.

        Args:
            module_name (str): Name of the module saved on system.
            method_name (str): Name of the method.
            stream_logs (bool, optional): Whether to stream logs from the method call. (Default: ``True``)
            run_name (str, optional): Name for the run. (Default: ``None``)
            remote (bool, optional): Return a remote object from the function, rather than the result proper.
                (Default: ``False``)
            run_async (bool, optional): Run the method asynchronously and return an awaitable. (Default: ``False``)
            save (bool, optional): Whether or not to save the call. (Default: ``False``)
            *args: Positional arguments to pass to the method.
            **kwargs: Keyword arguments to pass to the method.

        Example:
            >>> cluster.call("my_module", "my_method", arg1, arg2, kwarg1=kwarg1)
        """
        # Note: might be single value, might be a generator!
        if self.on_this_cluster():
            method_to_call = obj_store.acall if run_async else obj_store.call
            return method_to_call(
                module_name,
                method_name,
                data={"args": args, "kwargs": kwargs},
                stream_logs=stream_logs,
                run_name=run_name,
                remote=remote,
                serialization=None,
            )
        method_to_call = "acall_module_method" if run_async else "call_module_method"
        return self.call_client_method(
            method_to_call,
            module_name,
            method_name,
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
        state["_http_client"] = None
        state["_rpc_tunnel"] = None
        return state

    # ----------------- SSH Methods ----------------- #
    def rsync(
        self,
        source: str,
        dest: str,
        up: bool,
        node: str = None,
        contents: bool = False,
        filter_options: str = None,
        stream_logs: bool = False,
        ignore_existing: Optional[bool] = False,
    ):
        """
        Sync the contents of the source directory into the destination.

        Args:
            source (str): The source path.
            dest (str): The target path.
            up (bool): The direction of the sync. If ``True``, will rsync from local to cluster. If ``False``
              will rsync from cluster to local.
            node (Optional[str], optional): Specific cluster node to rsync to. If not specified will use the
                address of the cluster's head node.
            contents (Optional[bool], optional): Whether the contents of the source directory or the directory
                itself should be copied to destination.
                If ``True`` the contents of the source directory are copied to the destination, and the source
                directory itself is not created at the destination.
                If ``False`` the source directory along with its contents are copied ot the destination, creating
                an additional directory layer at the destination. (Default: ``False``).
            filter_options (Optional[str], optional): The filter options for rsync.
            stream_logs (Optional[bool], optional): Whether to stream logs to the stdout/stderr. (Default: ``False``).
            ignore_existing (Optional[bool], optional): Whether the rsync should skip updating files that already exist
                on the destination. (Default: ``False``).

        .. note::
            Ending ``source`` with a slash will copy the contents of the directory into dest,
            while omitting it will copy the directory itself (adding a directory layer).
        """
        # Theoretically we could reuse this logic from SkyPilot which Rsyncs to all nodes in parallel:
        # https://github.com/skypilot-org/skypilot/blob/v0.4.1/sky/backends/cloud_vm_ray_backend.py#L3094
        # This is an interesting policy... by default we're syncing to all nodes if the cluster is multinode.
        # If we need to change it to be greedier we can.
        if up and not Path(source).expanduser().exists():
            raise ValueError(f"Could not locate path to sync: {source}.")

        if up and (node == "all" or (len(self.ips) > 1 and not node)):
            for node in self.ips:
                self.rsync(
                    source,
                    dest,
                    up=up,
                    node=node,
                    contents=contents,
                    filter_options=filter_options,
                    stream_logs=stream_logs,
                    ignore_existing=ignore_existing,
                )
            return

        from runhouse.resources.hardware.sky_command_runner import SshMode

        # If no address provided explicitly use the head node address
        node = node or self.head_ip
        # FYI, could be useful: https://github.com/gchamon/sysrsync
        if contents:
            source = source + "/" if not source.endswith("/") else source
            dest = dest + "/" if not dest.endswith("/") else dest

        # If we're already on this cluster (and node, if multinode), this is just a local rsync
        if self.on_this_cluster() and node == self.head_ip:
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
            if ignore_existing:
                cmd += ["--ignore-existing"]
            if filter_options:
                cmd += [filter_options]

            subprocess.run(cmd, check=True, capture_output=not stream_logs, text=True)
            return

        ssh_credentials = copy.copy(self.creds_values) or {}
        ssh_credentials.pop("ssh_host", node)
        pwd = ssh_credentials.pop("password", None)
        ssh_credentials.pop("private_key", None)
        ssh_credentials.pop("public_key", None)

        runner = self._command_runner(node=node)
        if not pwd:
            if up:
                runner.run(
                    ["mkdir", "-p", dest],
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
                ignore_existing=ignore_existing,
            )

        else:
            if up:
                ssh_command = runner.run(
                    ["mkdir", "-p", dest],
                    return_cmd=True,
                    ssh_mode=SshMode.INTERACTIVE,
                )
                run_command_with_password_login(ssh_command, pwd, stream_logs=True)
            else:
                Path(dest).expanduser().parent.mkdir(parents=True, exist_ok=True)

            rsync_cmd = runner.rsync(
                source,
                dest,
                up=up,
                filter_options=filter_options,
                stream_logs=stream_logs,
                return_cmd=True,
                ignore_existing=ignore_existing,
            )
            run_command_with_password_login(rsync_cmd, pwd, stream_logs)

    def ssh(self):
        """SSH into the cluster

        Example:
            >>> rh.cluster("rh-cpu").ssh()
        """
        creds = self.creds_values
        _run_ssh_command(
            address=self.head_ip,
            ssh_user=creds["ssh_user"],
            ssh_port=self.ssh_port,
            ssh_private_key=creds["ssh_private_key"],
            docker_user=self.docker_user,
        )

    def _ping(self, timeout=5, retry=False):
        if not self.ips:
            return False

        def run_ssh_call():
            res = self._run_commands_with_runner(['echo "hello"'], stream_logs=False)
            if res[0][0] != 0:
                raise Exception

        ssh_call = ThreadWithException(target=run_ssh_call)
        try:
            ssh_call.start()
            ssh_call.join(timeout=timeout)
            if not ssh_call.is_alive():
                return True
        except:
            pass

        if retry:
            return self._ping(retry=False)
        return False

    def _copy_certs_to_cluster(self):
        """Copy local certs to the cluster. Destination on the cluster depends on whether Caddy is enabled. This is
        to ensure that the Caddy service has the necessary access to load the certs when the service is started."""
        # Copy to the home directory by default
        source = str(Path(self.cert_config.key_path).parent)
        dest = self.cert_config.DEFAULT_CLUSTER_DIR
        self.rsync(source, dest, up=True)

        if self._use_caddy:
            # Move to the Caddy directory to ensure the daemon has access to the certs
            src = self.cert_config.DEFAULT_CLUSTER_DIR
            dest = self.cert_config.CADDY_CLUSTER_DIR
            self._run_commands_with_runner(
                [
                    f"sudo mkdir -p {dest}",
                    f"sudo mv {src}/* {dest}/",
                    f"sudo rm -r {src}",
                ]
            )

        logger.debug(f"Copied local certs onto the cluster in path: {dest}")

        # private key should only live on the cluster
        Path(self.cert_config.key_path).unlink()

    def run(
        self,
        commands: Union[str, List[str]],
        env: Union["Env", str] = None,
        stream_logs: bool = True,
        require_outputs: bool = True,
        node: Optional[str] = None,
        _ssh_mode: str = "interactive",  # Note, this only applies for non-password SSH
    ) -> List:
        """Run a list of shell commands on the cluster.

        Args:
            commands (str or List[str]): Command or list of commands to run on the cluster.
            env (Env or str, optional): Env on the cluster to run the command in. If not provided,
                will be run in the default env. (Default: ``None``)
            stream_logs (bool, optional): Whether to stream log output as the command runs.
                (Default: ``True``)
            require_outputs (bool, optional): If ``True``, returns a Tuple (returncode, stdout, stderr).
                If ``False``, returns just the returncode. (Default: ``True``)
            node (str, optional): Node to run the commands on. If not provided, runs on head node.
                (Default: ``None``)

        Example:
            >>> cpu.run(["pip install numpy"])
            >>> cpu.run(["pip install numpy"], env="my_conda_env"])
            >>> cpu.run(["python script.py"])
            >>> cpu.run(["python script.py"], node="3.89.174.234")
        """
        from runhouse.resources.envs import Env

        if isinstance(commands, str):
            commands = [commands]

        if not env:
            if self.image and self.image.conda_env_name:
                commands = [
                    conda_env_cmd(cmd, self.image.conda_env_name) for cmd in commands
                ]
            env = DEFAULT_PROCESS_NAME

        env = _get_env_from(env)

        # If node is not specified, then we just use normal logic, knowing that we are likely on the head node
        if not node:
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

        # Node is specified, so we do everything via ssh
        else:
            if node == "all":
                res_list = []
                for node in self.ips:
                    res = self.run(
                        commands=commands,
                        env=env,
                        stream_logs=stream_logs,
                        require_outputs=require_outputs,
                        node=node,
                        _ssh_mode=_ssh_mode,
                    )
                    res_list.append(res)
                return res_list

            else:
                if env and not isinstance(env, str):
                    commands = [env._full_command(cmd) for cmd in commands]
                if self.on_this_cluster():
                    # TODO add log streaming
                    # Switch the external ip to an internal ip
                    node = self.internal_ips[self.ips.index(node)]
                    return_codes = obj_store.run_bash_command_on_node(
                        node_ip=node,
                        commands=commands,
                        require_outputs=require_outputs,
                    )
                    return return_codes

                return_codes = self._run_commands_with_runner(
                    commands,
                    cmd_prefix="",
                    stream_logs=stream_logs,
                    node=node,
                    require_outputs=require_outputs,
                    _ssh_mode=_ssh_mode,
                )

                return return_codes

    def _run_commands_with_runner(
        self,
        commands: list,
        env_vars: Dict = {},
        cmd_prefix: str = "",
        stream_logs: bool = True,
        node: str = None,
        require_outputs: bool = True,
        _ssh_mode: str = "interactive",  # Note, this only applies for non-password SSH
    ):
        from runhouse.resources.hardware.sky_command_runner import SshMode

        if isinstance(commands, str):
            commands = [commands]

        # If no address provided explicitly use the head node address
        node = node or self.head_ip

        return_codes = []

        ssh_credentials = copy.copy(self.creds_values)
        pwd = ssh_credentials.pop("password", None)
        ssh_credentials.pop("private_key", None)
        ssh_credentials.pop("public_key", None)

        runner = self._command_runner(
            node=node, use_docker_exec=self.docker_user is not None
        )

        env_var_prefix = (
            " ".join(f"{key}={val}" for key, val in env_vars.items())
            if env_vars
            else ""
        )

        for command in commands:
            command = f"{cmd_prefix} {command}" if cmd_prefix else command
            logger.info(f"Running command on {self.name}: {command}")

            # set env vars after log statement
            command = f"{env_var_prefix} {command}" if env_var_prefix else command

            if not pwd:
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
                    ssh_mode=ssh_mode,
                    quiet_ssh=True,
                )
                return_codes.append(ret_code)
            else:
                # We need to quiet the SSH output here or it will print
                # "Shared connection to ____ closed." at the end, which messes with the output.
                ssh_command = runner.run(
                    command,
                    require_outputs=require_outputs,
                    stream_logs=stream_logs,
                    return_cmd=True,
                    ssh_mode=SshMode.INTERACTIVE,
                    quiet_ssh=True,
                )
                command_run = run_command_with_password_login(
                    ssh_command, pwd, stream_logs
                )
                # Filter color characters from ssh.before, as otherwise sometimes random color characters
                # will be printed to the console.
                command_run.before = re.sub(r"\x1b\[[0-9;]*m", "", command_run.before)
                if require_outputs:
                    return_codes.append(
                        [
                            command_run.exitstatus,
                            command_run.before.strip(),
                            command_run.signalstatus,
                        ]
                    )
                else:
                    return_codes.append(command_run.exitstatus)

        return return_codes

    def run_python(
        self,
        commands: List[str],
        env: Union["Env", str] = None,
        stream_logs: bool = True,
        node: str = None,
    ):
        """Run a list of python commands on the cluster, or a specific cluster node if its IP is provided.

        Args:
            commands (List[str]): List of commands to run.
            env (Env or str, optional): Env to run the commands in. (Default: ``None``)
            stream_logs (bool, optional): Whether to stream logs. (Default: ``True``)
            node (str, optional): Node to run commands on. If not specified, runs on head node. (Default: ``None``)

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
            env=env,
            stream_logs=stream_logs,
            node=node,
        )

        return return_codes

    def create_conda_env(self, env_name: str, conda_yaml: Dict):
        install_conda(cluster=self)
        create_conda_env_on_cluster(
            env_name=env_name,
            conda_yaml=conda_yaml,
            cluster=self,
        )

    def sync_secrets(
        self,
        providers: Optional[List[str or "Secret"]] = None,
        env: Union[str, "Env"] = None,
    ):
        """Send secrets for the given providers.

        Args:
            providers(List[str] or None, optional): List of providers to send secrets for.
                If `None`, all providers configured in the environment will by sent. (Default: ``None``)
            env (str, Env, optional): Env to sync secrets into. (Default: ``None``)

        Example:
            >>> cpu.sync_secrets(secrets=["aws", "lambda"])
        """
        from runhouse.resources.envs import Env
        from runhouse.resources.secrets import Secret

        if isinstance(env, str):
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
            process = (
                env
                if isinstance(env, str)
                else env.name
                if isinstance(env, Env)
                else None
            )
            secret.to(self, process=process)

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

        from runhouse.resources.hardware.ssh_tunnel import is_port_in_use

        while is_port_in_use(port_forward):
            port_forward += 1

        tunnel = self.ssh_tunnel(
            local_port=port_forward,
            num_ports_to_try=NUM_PORTS_TO_TRY,
        )
        port_fwd = tunnel.remote_bind_port

        try:
            jupyter_cmd = f"jupyter lab --port {port_fwd} --no-browser"
            with self.pause_autostop():
                self.install_packages(["jupyterlab"])
                # TODO figure out why logs are not streaming here if we don't use ssh.
                # When we do, it may be better to switch it back because then jupyter is killed
                # automatically when the cluster is restarted (and the process is killed).
                self.run(commands=[jupyter_cmd], stream_logs=True, node=self.head_ip)

        finally:
            if sync_package_on_close:
                from runhouse.resources.packages.package import Package

                if sync_package_on_close == "./":
                    sync_package_on_close = locate_working_dir()
                pkg = Package.from_string("local:" + sync_package_on_close)
                self.rsync(source=f"~/{pkg.name}", dest=pkg.local_path, up=False)
            if not persist:
                tunnel.terminate()
                kill_jupyter_cmd = f"jupyter notebook stop {port_fwd}"
                self.run(commands=[kill_jupyter_cmd])

    def connect_dask(
        self,
        port: int = DEFAULT_DASK_PORT,
        scheduler_options: Dict = None,
        worker_options: Dict = None,
        client_timeout: str = "3s",
    ):
        local_scheduler_address = f"tcp://localhost:{port}"
        remote_scheduler_address = f"tcp://{self.internal_ips[0]}:{port}"

        # First check if dask is already running at the specified port
        from dask.distributed import Client

        # TODO: Handle case where we're on a worker node
        if not self.on_this_cluster():
            self.ssh_tunnel(
                local_port=port, remote_port=port, num_ports_to_try=NUM_PORTS_TO_TRY
            )

        try:
            # We need to connect to localhost both when we're on the head node and if we've formed
            # an SSH tunnel
            client = Client(local_scheduler_address, timeout=client_timeout)
            logger.info(f"Connected to Dask client {client}")
            return client
        except OSError:
            client = None

        logger.info(f"Starting Dask on {self.name}.")
        if scheduler_options:
            scheduler_options = " ".join(
                [f"--{key} {val}" for key, val in scheduler_options.items()]
            )
        else:
            scheduler_options = ""
        self.run(
            f"nohup dask scheduler --port {port} {scheduler_options} > dask_scheduler.out 2>&1 &",
            node=self.head_ip,
            stream_logs=True,
            require_outputs=True,
        )

        worker_options = worker_options or {}
        if "nworkers" not in worker_options:
            worker_options["nworkers"] = "auto"
        worker_options_str = " ".join(
            [f"--{key} {val}" for key, val in worker_options.items()]
        )

        # Note: We need to do this on the head node too, because this creates all the worker processes
        for node in self.ips:
            logger.info(f"Starting Dask worker on {node}.")
            # Connect to localhost if on the head node, otherwise use the internal ip of head node
            scheduler = (
                local_scheduler_address
                if node == self.head_ip
                else remote_scheduler_address
            )
            self.run(
                f"nohup dask worker {scheduler} {worker_options_str} > dask_worker.out 2>&1 &",
                node=node,
            )

        client = Client(local_scheduler_address, timeout=client_timeout)
        logger.info(f"Connected to Dask on {self.name}:{port} with client {client}.")
        return client

    def kill_dask(self):
        self.run("pkill -f 'dask scheduler'", node=self.head_ip)
        for node in self.ips:
            self.run("pkill -f 'dask worker'", node=node)

    def remove_conda_env(
        self,
        env: Union[str, "CondaEnv"],
    ):
        """Remove conda env from the cluster.

        Args:
            env (str or Env): Name of conda env to remove from the cluster, or Env resource
                representing the environment.

        Example:
            >>> rh.ondemand_cluster("rh-cpu").remove_conda_env("my_conda_env")
        """
        env_name = env if isinstance(env, str) else env.env_name
        self.run([f"conda env remove -n {env_name}"])

    def download_cert(self):
        """Download certificate from the cluster (Note: user must have access to the cluster)"""
        self.call_client_method("get_certificate")
        logger.info(
            f"Latest TLS certificate for {self.name} saved to local path: {self.cert_config.cert_path}"
        )

    def enable_den_auth(self, flush: bool = True):
        """Enable Den auth on the cluster.

        Args:
            flush (bool, optional): Whether to flush the auth cache. (Default: ``True``)
        """
        if self.on_this_cluster():
            raise ValueError("Cannot toggle Den Auth live on the cluster.")
        else:
            self.den_auth = True
            self.call_client_method(
                "set_settings", {"den_auth": True, "flush_auth_cache": flush}
            )
        return self

    def disable_den_auth(self):
        if self.on_this_cluster():
            raise ValueError("Cannot toggle Den Auth live on the cluster.")
        else:
            self.den_auth = False
            self.call_client_method("set_settings", {"den_auth": False})
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
        if self._creds:
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
        from runhouse.resources.secrets.secret import Secret
        from runhouse.resources.secrets.utils import load_config

        creds = config.pop("creds", None) or config.pop("ssh_creds", None)

        if isinstance(creds, str):
            creds = Secret.from_config(config=load_config(name=creds))
        elif isinstance(creds, dict):
            creds = Secret.from_config(creds)
        config["creds"] = creds

        return config

    ##############################################
    # Send Cluster status to Den methods
    ##############################################
    def _disable_status_check(self):
        """Stop sending periodic status checks to Den."""
        if not self.den_auth:
            logger.error(
                "Cluster must have Den auth enabled to allow periodic status checks. "
                "Make sure you have a Den account and the cluster has `den_auth=True`."
            )
            return
        if self.on_this_cluster():
            obj_store.set_cluster_config_value("status_check_interval", -1)
        else:
            self.call_client_method("set_settings", {"status_check_interval": -1})

    def _enable_or_update_status_check(
        self, new_interval: int = DEFAULT_STATUS_CHECK_INTERVAL
    ):
        """
        Enables a periodic status check or updates the interval between cluster status checks.

        Args:
            new_interval (int): Updated number of minutes between status checks.
        """
        if not self.den_auth:
            logger.error(
                "Cluster must have Den auth enabled to update the interval for periodic status checks. "
                "Make sure you have a Den account and the cluster has `den_auth=True`."
            )
            return
        if self.on_this_cluster():
            obj_store.set_cluster_config_value("status_check_interval", new_interval)
        else:
            self.call_client_method(
                "set_settings", {"status_check_interval": new_interval}
            )

    ##############################################
    # Folder Operations
    ##############################################
    def _folder_ls(
        self, path: Union[str, Path], full_paths: bool = True, sort: bool = False
    ):
        return self.client.folder_ls(path=path, full_paths=full_paths, sort=sort)

    def _folder_get(
        self,
        path: Union[str, Path],
        mode: str = "rb",
        encoding: str = None,
    ):
        return self.client.folder_get(
            path=path,
            mode=mode,
            encoding=encoding,
        )

    def _folder_put(
        self,
        path: Union[str, Path],
        contents: Union[Dict[str, Any], Resource, List[Resource]],
        mode: str = "wb",
        overwrite: bool = False,
        serialization: str = None,
    ):
        return self.client.folder_put(
            path=path,
            contents=contents,
            mode=mode,
            overwrite=overwrite,
            serialization=serialization,
        )

    def _folder_rm(
        self,
        path: Union[str, Path],
        contents: List[str] = None,
        recursive: bool = False,
    ):
        return self.client.folder_rm(path=path, contents=contents, recursive=recursive)

    def _folder_mkdir(self, path: Union[str, Path]):
        return self.client.folder_mkdir(path=path)

    def _folder_mv(
        self,
        path: Union[str, Path],
        dest_path: Union[str, Path],
        overwrite: bool = True,
    ):
        return self.client.folder_mv(
            path=path, dest_path=dest_path, overwrite=overwrite
        )

    def _folder_exists(self, path: Union[str, Path]):
        return self.client.folder_exists(path=path)

    ###############################
    # Cluster list
    ###############################
    @classmethod
    def list(
        cls,
        show_all: bool = False,
        since: Optional[str] = None,
        status: Optional[Union[str, ClusterStatus]] = None,
        force: bool = False,
    ) -> Dict[str, List[Dict]]:
        """
        Loads Runhouse clusters saved in Den and locally via Sky. If filters are provided, only clusters that
        are matching the filters are returned. If no filters are provided, all running clusters will be returned.

        Args:
            show_all (bool, optional): Whether to list all clusters saved in Den. Maximum of 200 will be listed.
                (Default: False).
            since (str, optional): Clusters that were active in the specified time period will be returned.
                Value can be in seconds, minutes, hours or days.
            status (str or ClusterStatus, optional): Clusters with the provided status will be returned.
                Options include: ``running``, ``terminated``, ``initializing``, ``unknown``.
            force (bool, optional): Whether to force a status update for all relevant clusters, or load the latest
                values. (Default: False).

        Examples:
            >>> Cluster.list(since="75s")
            >>> Cluster.list(since="3m")
            >>> Cluster.list(since="2h", status="running")
            >>> Cluster.list(since="7d")
            >>> Cluster.list(show_all=True)
        """
        cluster_filters = (
            parse_filters(since=since, cluster_status=status)
            if not show_all
            else {"all": "all"}
        )

        # get clusters from den
        den_clusters_resp = get_clusters_from_den(
            cluster_filters=cluster_filters, force=force
        )
        if den_clusters_resp.status_code != 200:
            logger.error(f"Failed to load {rns_client.username}'s clusters from Den")
            den_clusters = []
        else:
            den_clusters = den_clusters_resp.json().get("data")

        try:
            # get sky live clusters
            sky_live_clusters = get_unsaved_live_clusters(den_clusters=den_clusters)
            sky_live_clusters = [
                {
                    "Name": sky_cluster.get("name"),
                    "Cluster Type": "OnDemandCluster (Sky)",
                    "Status": sky_cluster.get("status").value,
                }
                for sky_cluster in sky_live_clusters
            ]
        except Exception:
            logger.debug("Failed to load sky live clusters.")
            sky_live_clusters = []

        if not sky_live_clusters and not den_clusters:
            return {}

        # running_clusters: running clusters which are saved in Den
        # not running clusters: clusters that are terminated / unknown / down which are also saved in Den.
        (
            running_clusters,
            not_running_clusters,
        ) = get_running_and_not_running_clusters(clusters=den_clusters)
        all_clusters = running_clusters + not_running_clusters

        clusters = {
            "den_clusters": all_clusters,
            "sky_clusters": sky_live_clusters,
        }
        return clusters

    def list_processes(self):
        """List all workers on the cluster."""
        if self.on_this_cluster():
            return obj_store.list_processes()
        else:
            return self.client.list_processes()

    def create_process(
        self,
        name: str,
        env_vars: Optional[Dict] = None,
        compute: Optional[Dict] = None,
        runtime_env: Optional[Dict] = None,
    ) -> str:
        runtime_env = runtime_env or {}
        create_process_params = CreateProcessParams(
            name=name, compute=compute, runtime_env=runtime_env, env_vars=env_vars
        )

        # If it exists, but with the exact same args, then we're good, else raise an error
        existing_processes = self.list_processes()
        if (
            name in existing_processes
            and existing_processes[name] != create_process_params
        ):
            raise ValueError(
                f"Process {name} already exists and was started with different arguments."
            )

        if self.on_this_cluster():
            obj_store.get_servlet(
                env_name=name, create_process_params=create_process_params, create=True
            )
        else:
            self.client.create_process(params=create_process_params)

        return name

    def ensure_process_created(
        self,
        name: str,
        env_vars: Optional[Dict] = None,
        compute: Optional[Dict] = None,
        runtime_env: Optional[Dict] = None,
    ) -> str:
        existing_processes = self.list_processes()
        if name in existing_processes:
            return name

        self.create_process(
            name=name, env_vars=env_vars, compute=compute, runtime_env=runtime_env
        )
        return name

    def set_process_env_vars(self, name: str, env_vars: Dict):
        if self.on_this_cluster():
            return obj_store.set_process_env_vars(name, env_vars)
        else:
            return self.client.set_process_env_vars(
                process_name=name, env_vars=env_vars
            )

    def install_package(
        self, package: Union["Package", str], conda_env_name: Optional[str] = None
    ):
        from runhouse.resources.packages.package import Package

        if isinstance(package, str):
            package = Package.from_string(package)

        if self.on_this_cluster():
            obj_store.ainstall_package_in_all_nodes_and_processes(
                package, conda_env_name
            )
        else:
            package = package.to(self)
            self.client.install_package(package, conda_env_name)

    def install_package_over_ssh(
        self, package: Union["Package", str], node: str, conda_env_name: str
    ):
        from runhouse.resources.packages.package import Package

        if isinstance(package, str):
            package = Package.from_string(package)
            if package.install_method in ["reqs", "local"]:
                package = package.to(self)

        package._install(cluster=self, node=node, conda_env_name=conda_env_name)
