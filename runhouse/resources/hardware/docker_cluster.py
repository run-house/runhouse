import contextlib
import json
import subprocess
import time
import warnings
from pathlib import Path
from typing import Any, Dict

import requests

try:
    import docker

    client = docker.from_env()
except ImportError:
    pass

from runhouse.constants import (
    DEFAULT_HTTP_PORT,
    DEFAULT_HTTPS_PORT,
    DEFAULT_SERVER_PORT,
    DOCKER_LOGIN_ENV_VARS,
    LOCAL_HOSTS,
)

from runhouse.globals import configs, rns_client

from runhouse.logger import logger
from runhouse.resources.hardware.utils import ResourceServerStatus, ServerConnectionType

from .cluster import Cluster


class DockerCluster(Cluster):
    RESOURCE_TYPE = "cluster"
    RECONNECT_TIMEOUT = 5
    DEFAULT_KEYFILE = "~/.ssh/sky-key"

    def __init__(
        self,
        name,
        num_instances: int = None,
        default_env: "Env" = None,
        dryrun=False,
        autostop_mins=None,  # We should actually try to support this to help with testing
        image_id=None,
        memory=None,
        disk_size=None,
        open_ports=None,
        server_host: str = None,
        server_port: int = None,
        server_connection_type: str = None,
        den_auth: bool = False,
        run_kwargs: Dict = None,
        **kwargs,  # We have this here to ignore extra arguments when calling from from_config
    ):
        """
        On-demand `SkyPilot <https://github.com/skypilot-org/skypilot/>`_ Cluster.

        .. note::
            To build a cluster, please use the factory method :func:`cluster`.
        """
        super().__init__(
            name=name,
            default_env=default_env,
            server_host=server_host,
            server_port=server_port,
            server_connection_type=server_connection_type,
            den_auth=den_auth,
            dryrun=dryrun,
            **kwargs,
        )

        self.num_instances = num_instances
        self.autostop_mins = (
            autostop_mins
            if autostop_mins is not None
            else configs.get("default_autostop")
        )

        self.open_ports = open_ports
        if "docker:" in image_id:
            # To allow reuse of image_id from ondemand_cluster
            image_id = image_id.split("docker:")[1]
        self.image_id = image_id
        self.memory = memory
        self.disk_size = disk_size
        self.run_kwargs = run_kwargs or {}

        self.stable_internal_external_ips = kwargs.get(
            "stable_internal_external_ips", None
        )
        self._docker_user = None

        self._network = None
        self._image = None
        self._container = None

    @property
    def docker_user(self) -> str:
        if self._docker_user:
            return self._docker_user

        if not self.image_id:
            return None

        from runhouse.resources.hardware.sky_ssh_runner import get_docker_user

        if not self._creds:
            return
        self._docker_user = get_docker_user(self, self._creds.values)

        return self._docker_user

    def config(self, condensed=True):
        config = super().config(condensed)
        self.save_attrs_to_config(
            config,
            [
                "num_instances",
                "open_ports",
                "image_id",
                "stable_internal_external_ips",
                "memory",
                "disk_size",
                "run_kwargs",
            ],
        )
        config["autostop_mins"] = self.autostop_mins
        return config

    def endpoint(self, external=False):
        try:
            self.check_server()
        except ValueError:
            return None

        return super().endpoint(external)

    def set_connection_defaults(self):
        if self.server_connection_type in [
            ServerConnectionType.AWS_SSM,
        ]:
            raise ValueError(
                f"OnDemandCluster does not support server connection type {self.server_connection_type}"
            )

        if not self.server_connection_type:
            if self.ssl_keyfile or self.ssl_certfile:
                self.server_connection_type = ServerConnectionType.TLS
            else:
                self.server_connection_type = ServerConnectionType.SSH

        if self.server_port is None:
            if self.server_connection_type == ServerConnectionType.TLS:
                self.server_port = DEFAULT_HTTPS_PORT
            elif self.server_connection_type == ServerConnectionType.NONE:
                self.server_port = DEFAULT_HTTP_PORT
            else:
                self.server_port = DEFAULT_SERVER_PORT

        if (
            self.server_connection_type
            in [ServerConnectionType.TLS, ServerConnectionType.NONE]
            and self.server_host in LOCAL_HOSTS
        ):
            warnings.warn(
                f"Server connection type: {self.server_connection_type}, server host: {self.server_host}. "
                f"Note that this will require opening an SSH tunnel to forward traffic from"
                f" {self.server_host} to the server."
            )

        self.open_ports = (
            []
            if self.open_ports is None
            else [self.open_ports]
            if isinstance(self.open_ports, (int, str))
            else self.open_ports
        )

        if self.open_ports:
            self.open_ports = [str(p) for p in self.open_ports]
            if str(self.server_port) in self.open_ports:
                if (
                    self.server_connection_type
                    in [ServerConnectionType.TLS, ServerConnectionType.NONE]
                    and not self.den_auth
                ):
                    warnings.warn(
                        "Server is insecure and must be inside a VPC or have `den_auth` enabled to secure it."
                    )
            else:
                warnings.warn(
                    f"Server port {self.server_port} not included in open ports. Note you are responsible for opening "
                    f"the port or ensure you have access to it via a VPC."
                )
        else:
            # If using HTTP or HTTPS must enable traffic on the relevant port
            if self.server_connection_type in [
                ServerConnectionType.TLS,
                ServerConnectionType.NONE,
            ]:
                if self.server_port:
                    warnings.warn(
                        f"No open ports specified. Setting default port {self.server_port} to open."
                    )
                    self.open_ports = [str(self.server_port)]
                else:
                    warnings.warn(
                        f"No open ports specified. Make sure the relevant port is open. "
                        f"HTTPS default: {DEFAULT_HTTPS_PORT} and HTTP "
                        f"default: {DEFAULT_HTTP_PORT}."
                    )

    # ----------------- Launch/Lifecycle Methods -----------------

    def is_up(self) -> bool:
        """Whether the cluster is up.

        Example:
            >>> rh.ondemand_cluster("rh-cpu").is_up()
        """
        if self.on_this_cluster():
            return True
        self._update_from_sky_status(dryrun=False)
        return self.address is not None

    def _start_ray_workers(self, ray_port, env):
        # Find the internal IP corresponding to the public_head_ip and the rest are workers
        internal_head_ip = None
        worker_ips = []
        stable_internal_external_ips = self._sky_status()[
            "handle"
        ].stable_internal_external_ips
        for internal, external in stable_internal_external_ips:
            if external == self.address:
                internal_head_ip = internal
            else:
                # NOTE: Using external worker address here because we're running from local
                worker_ips.append(external)

        logger.debug(f"Internal head IP: {internal_head_ip}")

        for host in worker_ips:
            logger.info(
                f"Starting Ray on worker {host} with head node at {internal_head_ip}:{ray_port}."
            )
            self.run(
                commands=[
                    f"ray start --address={internal_head_ip}:{ray_port} --disable-usage-stats",
                ],
                node=host,
                env=env,
            )
        time.sleep(5)

    def _populate_connection_from_status_dict(self, cluster_dict: Dict[str, Any]):
        if cluster_dict and cluster_dict["status"].name in ["UP", "INIT"]:
            handle = cluster_dict["handle"]
            self.address = handle.head_ip
            self.stable_internal_external_ips = handle.stable_internal_external_ips
            if self.stable_internal_external_ips is None or self.address is None:
                raise ValueError(
                    "Sky's cluster status does not have the necessary information to connect to the cluster. Please check if the cluster is up via `sky status`. Consider bringing down the cluster with `sky down` if you are still having issues."
                )
            yaml_path = handle.cluster_yaml
            if Path(yaml_path).exists():
                ssh_values = backend_utils.ssh_credential_from_yaml(yaml_path)
                if not self.creds_values:
                    from runhouse.resources.secrets.utils import setup_cluster_creds

                    self._creds = setup_cluster_creds(ssh_values, self.name)

            # Add worker IPs if multi-node cluster - keep the head node as the first IP
            self.ips = [ext for _, ext in self.stable_internal_external_ips]
        else:
            self.address = None
            self._creds = None
            self.stable_internal_external_ips = None

    def _update_from_sky_status(self, dryrun: bool = False):
        # Try to get the cluster status from SkyDB
        if self.is_shared:
            # If the cluster is shared can ignore, since the sky data will only be saved on the machine where
            # the cluster was initially upped
            return

        cluster_dict = self._sky_status(refresh=not dryrun)
        self._populate_connection_from_status_dict(cluster_dict)

    def get_instance_type(self):
        if self.instance_type and "--" in self.instance_type:  # K8s specific syntax
            return self.instance_type
        elif (
            self.instance_type
            and ":" not in self.instance_type
            and "CPU" not in self.instance_type
        ):
            return self.instance_type

        return None

    def accelerators(self):
        if (
            self.instance_type
            and ":" in self.instance_type
            and "CPU" not in self.instance_type
        ):
            return self.instance_type

        return None

    def num_cpus(self):
        if (
            self.instance_type
            and ":" in self.instance_type
            and "CPU" in self.instance_type
        ):
            return self.instance_type.rsplit(":", 1)[1]

        return None

    def up(self):
        """Up the cluster.

        Example:
            >>> rh.ondemand_cluster("rh-cpu").up()
        """
        if self.on_this_cluster():
            return self

        dockerfile_path = None
        if Path(self.image_id).resolve().exists():
            dockerfile_path = Path(self.image_id).resolve()
            image_id = f"runhouse:{self.name}"
        else:
            image_id = self.image_id

        # Check if the container is already running, and if so, skip build and run
        containers = client.containers.list(
            all=True,
            filters={
                "ancestor": image_id,
                "status": ResourceServerStatus.running,
                "name": self.name,
            },
        )
        if not self._network:
            self._network = client.networks.create("runhouse-bridge", driver="bridge")

        if len(containers) > 0:
            if not self.detach:
                raise ValueError(
                    f"Container {self.name} already running, but detach=False"
                )
            else:
                logger.info(
                    f"Container {self.name} already running, skipping build and run."
                )
                self._container = containers[0]
                self._container.reload()
                self._network.connect(self._container)

        if not self._container:
            # The container is not running, so we need to pull/build and run it
            # Check if image exists locally before building or pulling
            images = client.images.list(filters={"reference": image_id})
            self._image = images[0] if images else None
            if not self._image or self.force_rebuild:
                if dockerfile_path:
                    self._image = client.images.build(
                        dockerfile_path,
                        tag=image_id,
                        pull=True,
                        rm=True,
                        forcerm=True,
                        secret_files=[
                            (self.creds_values["ssh_private_key"], "ssh_key")
                        ],
                    )
                else:
                    self._image = client.images.pull(image_id)

            self._container = client.containers.run(
                self._image,
                detach=True,
                name=self.name,
                network=self._network.name,
                environment=DOCKER_LOGIN_ENV_VARS,
                shm_size="5.04gb",
                rm=True,
            )
            self._container.reload()

        self.address = self._container.attrs["NetworkSettings"]["IPAddress"]
        self.restart_server()

        return self

    def keep_warm(self, mins: int = -1):
        """Keep the cluster warm for given number of minutes after inactivity.

        Args:
            mins (int): Amount of time (in min) to keep the cluster warm after inactivity.
                If set to -1, keep cluster warm indefinitely. (Default: `-1`)
        """
        self.autostop_mins = mins
        return self

    def teardown(self):
        """Teardown cluster.

        Example:
            >>> rh.ondemand_cluster("rh-cpu").teardown()
        """
        # TODO [SB]: remove the den_auth check once we will get status of clusters without den_auth as well.
        warning_msg = "Failed to update Den with cluster terminated status."
        if self.den_auth:
            try:
                cluster_status_data = self.status()
                status_data = {
                    "status": ResourceServerStatus.terminated,
                    "resource_type": self.__class__.__base__.__name__.lower(),
                    "data": cluster_status_data,
                }
                cluster_uri = rns_client.format_rns_address(self.rns_address)
                api_server_url = cluster_status_data.get("cluster_config").get(
                    "api_server_url", rns_client.api_server_url
                )
                post_status_data_resp = requests.post(
                    f"{api_server_url}/resource/{cluster_uri}/cluster/status",
                    data=json.dumps(status_data),
                    headers=rns_client.request_headers(),
                )
                if post_status_data_resp.status_code != 200:
                    post_status_data_resp = post_status_data_resp.json()
                    warning_msg = (
                        warning_msg
                        + f' Got {post_status_data_resp.status_code}: {post_status_data_resp.json()["detail"]}'
                    )
                    logger.warning(warning_msg)
            except Exception as e:
                warning_msg = warning_msg + f" Got {e}"
                logger.warning(warning_msg)

        # Stream logs
        sky.down(self.name)
        self.address = None

    def teardown_and_delete(self):
        """Teardown cluster and delete it from configs.

        Example:
            >>> rh.ondemand_cluster("rh-cpu").teardown_and_delete()
        """
        self.teardown()
        rns_client.delete_configs(resource=self)

    @contextlib.contextmanager
    def pause_autostop(self):
        """Context manager to temporarily pause autostop.

        Example:
            >>> with rh.ondemand_cluster.pause_autostop():
            >>>     rh.ondemand_cluster.run(["python train.py"])
        """
        sky.autostop(self.name, idle_minutes=-1)
        yield
        sky.autostop(self.name, idle_minutes=self._autostop_mins, down=True)

    # ----------------- SSH Methods ----------------- #

    @staticmethod
    def cluster_ssh_key(path_to_file):
        """Retrieve SSH key for the cluster.

        Example:
            >>> ssh_priv_key = rh.ondemand_cluster("rh-cpu").cluster_ssh_key("~/.ssh/id_rsa")
        """
        try:
            f = open(path_to_file, "r")
            private_key = f.read()
            return private_key
        except FileNotFoundError:
            raise Exception(f"File with ssh key not found in: {path_to_file}")

    def ssh(self, node: str = None):
        """SSH into the cluster. If no node is specified, will SSH onto the head node.

        Example:
            >>> rh.ondemand_cluster("rh-cpu").ssh()
            >>> rh.ondemand_cluster("rh-cpu", node="3.89.174.234").ssh()
        """
        if self.provider == "kubernetes":
            command = f"kubectl get pods | grep {self.name}"

            try:

                output = subprocess.check_output(command, shell=True, text=True)

                lines = output.strip().split("\n")
                if lines:
                    pod_name = lines[0].split()[0]
                else:
                    logger.info("No matching pods found.")
            except subprocess.CalledProcessError as e:
                raise Exception(f"Error: {e}")

            cmd = f"kubectl exec -it {pod_name} -- /bin/bash"
            subprocess.run(cmd, shell=True, check=True)

        else:
            # If SSHing onto a specific node, which requires the default sky public key for verification
            from runhouse.resources.hardware.sky_ssh_runner import SkySSHRunner, SshMode

            ssh_user = self.creds_values.get("ssh_user")
            sky_key = Path(
                self.creds_values.get("ssh_private_key", self.DEFAULT_KEYFILE)
            ).expanduser()

            if not sky_key.exists():
                raise FileNotFoundError(f"Expected default sky key in path: {sky_key}")

            runner = SkySSHRunner(
                ip=node or self.address,
                ssh_user=ssh_user,
                port=self.ssh_port,
                ssh_private_key=str(sky_key),
                docker_user=self.docker_user,
            )
            cmd = runner.run(
                cmd="bash --rcfile <(echo '. ~/.bashrc; conda deactivate')",
                ssh_mode=SshMode.INTERACTIVE,
                port_forward=None,
                return_cmd=True,
            )
            subprocess.run(cmd, shell=True)
