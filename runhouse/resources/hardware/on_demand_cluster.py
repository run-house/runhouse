import asyncio
import contextlib
import json
import subprocess
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Union

import requests

import rich.errors
import yaml

try:
    import sky
    from sky.backends import backend_utils
except ImportError:
    pass

from runhouse.constants import (
    DEFAULT_HTTP_PORT,
    DEFAULT_HTTPS_PORT,
    DEFAULT_SERVER_PORT,
    DOCKER_LOGIN_ENV_VARS,
    LOCAL_HOSTS,
)

from runhouse.globals import configs, obj_store, rns_client
from runhouse.logger import get_logger
from runhouse.resources.hardware.utils import (
    ResourceServerStatus,
    ServerConnectionType,
    up_cluster_helper,
)

from .cluster import Cluster

logger = get_logger(__name__)


class OnDemandCluster(Cluster):
    RESOURCE_TYPE = "cluster"
    RECONNECT_TIMEOUT = 5
    DEFAULT_KEYFILE = "~/.ssh/sky-key"

    def __init__(
        self,
        name,
        instance_type: str = None,
        num_instances: int = None,
        provider: str = None,
        default_env: "Env" = None,
        dryrun: bool = False,
        autostop_mins: int = None,
        use_spot: bool = False,
        image_id: str = None,
        memory: Union[int, str] = None,
        disk_size: Union[int, str] = None,
        open_ports: Union[int, str, List[int]] = None,
        server_host: int = None,
        server_port: int = None,
        server_connection_type: str = None,
        ssl_keyfile: str = None,
        ssl_certfile: str = None,
        domain: str = None,
        den_auth: bool = False,
        region: str = None,
        sky_kwargs: Dict = None,
        **kwargs,  # We have this here to ignore extra arguments when calling from from_config
    ):
        """
        On-demand `SkyPilot <https://github.com/skypilot-org/skypilot/>`__ Cluster.

        .. note::
            To build a cluster, please use the factory method :func:`cluster`.
        """
        super().__init__(
            name=name,
            default_env=default_env,
            server_host=server_host,
            server_port=server_port,
            server_connection_type=server_connection_type,
            ssl_keyfile=ssl_keyfile,
            ssl_certfile=ssl_certfile,
            domain=domain,
            den_auth=den_auth,
            dryrun=dryrun,
            **kwargs,
        )

        self.instance_type = instance_type
        self.num_instances = num_instances
        self.provider = provider or configs.get("default_provider")
        self._autostop_mins = (
            autostop_mins
            if autostop_mins is not None
            else configs.get("default_autostop")
        )

        self.open_ports = open_ports
        self.use_spot = use_spot if use_spot is not None else configs.get("use_spot")
        self.image_id = image_id
        self.region = region
        self.memory = memory
        self.disk_size = disk_size
        self.sky_kwargs = sky_kwargs or {}

        self.stable_internal_external_ips = kwargs.get(
            "stable_internal_external_ips", None
        )
        self.launched_properties = kwargs.get("launched_properties", {})
        self._docker_user = None

        # Checks if state info is in local sky db, populates if so.
        if not dryrun and not self.ips and not self.creds_values:
            # Cluster status is set to INIT in the Sky DB right after starting, so we need to refresh once
            self._update_from_sky_status(dryrun=False)

    @property
    def client(self):
        try:
            return super().client
        except ValueError as e:
            if not self.address:
                # Try loading in from local Sky DB
                self._update_from_sky_status(dryrun=True)
                if not self.address:
                    raise ValueError(
                        f"Could not determine address for ondemand cluster <{self.name}>. "
                        "Up the cluster with `cluster.up_if_not`."
                    )
                return super().client
            raise e

    @property
    def autostop_mins(self):
        return self._autostop_mins

    @autostop_mins.setter
    def autostop_mins(self, mins):
        self._autostop_mins = mins
        if self.on_this_cluster():
            obj_store.set_cluster_config_value("autostop_mins", mins)
        else:
            # if self.run_python(["import skypilot"])[0] != 0:
            #     raise ImportError(
            #         "Skypilot must be installed on the cluster in order to set autostop."
            #     )
            self.call_client_method("set_settings", {"autostop_mins": mins})
            sky.autostop(self.name, mins, down=True)

    @property
    def docker_user(self) -> str:
        if self._docker_user:
            return self._docker_user

        # TODO detect whether this is a k8s cluster properly, and handle the user setting / SSH properly
        #  (e.g. SkyPilot's new KubernetesCommandRunner)
        if not self.image_id or "docker:" not in self.image_id:
            return None

        if self.launched_properties["cloud"] == "kubernetes":
            return self.launched_properties.get(
                "docker_user", self.launched_properties.get("ssh_user", "root")
            )

        from runhouse.resources.hardware.sky_command_runner import get_docker_user

        if not self._creds:
            return
        self._docker_user = get_docker_user(self, self._creds.values)

        return self._docker_user

    def config(self, condensed=True):
        config = super().config(condensed)
        self.save_attrs_to_config(
            config,
            [
                "instance_type",
                "num_instances",
                "provider",
                "open_ports",
                "use_spot",
                "image_id",
                "region",
                "stable_internal_external_ips",
                "memory",
                "disk_size",
                "sky_kwargs",
                "launched_properties",
            ],
        )
        config["autostop_mins"] = self._autostop_mins
        return config

    def endpoint(self, external: bool = False):
        if not self.address or self.on_this_cluster():
            return None

        try:
            self.client.check_server()
        except ConnectionError:
            return None

        return super().endpoint(external)

    def _copy_sky_yaml_from_cluster(self, abs_yaml_path: str):
        if not Path(abs_yaml_path).exists():
            Path(abs_yaml_path).parent.mkdir(parents=True, exist_ok=True)
            self.rsync("~/.sky/sky_ray.yml", abs_yaml_path, up=False)

            # Save SSH info to the ~/.ssh/config
            ray_yaml = yaml.safe_load(open(abs_yaml_path, "r"))
            backend_utils.SSHConfigHelper.add_cluster(
                self.name, [self.address], ray_yaml["auth"]
            )

    @staticmethod
    def relative_yaml_path(yaml_path):
        if Path(yaml_path).is_absolute():
            yaml_path = "~/.sky/generated/" + Path(yaml_path).name
        return yaml_path

    def set_connection_defaults(self):
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
        return self._ping(retry=True)

    def _sky_status(self, refresh: bool = True, retry: bool = True):
        """
        Get status of Sky cluster.

        Return dict looks like:

        .. code-block::

            {'name': 'sky-cpunode-donny',
             'launched_at': 1662317201,
             'handle': ResourceHandle(
                          cluster_name=sky-cpunode-donny,
                          head_ip=54.211.97.164,
                          cluster_yaml=/Users/donny/.sky/generated/sky-cpunode-donny.yml,
                          launched_resources=1x AWS(m6i.2xlarge),
                          tpu_create_script=None,
                          tpu_delete_script=None),
             'last_use': 'sky cpunode',
             'status': <ClusterStatus.UP: 'UP'>,
             'autostop': -1,
             'metadata': {}}


        .. note:: For more information see SkyPilot's :code:`ResourceHandle` `class
        <https://github.com/skypilot-org/skypilot/blob/0c2b291b03abe486b521b40a3069195e56b62324/sky/backends/cloud_vm_ray_backend.py#L1457>`__.
        """
        if not sky.global_user_state.get_cluster_from_name(self.name):
            return None

        try:
            state = sky.status(cluster_names=[self.name], refresh=refresh)
        except rich.errors.LiveError as e:
            # We can't have more than one Live display at once, so if we've already launched one (e.g. the first
            # time we call status), we can retry without refreshing
            if not retry:
                raise e
            return self._sky_status(refresh=False, retry=False)

        # We still need to check if the cluster present in case the cluster went down and was removed from the DB
        if len(state) == 0:
            return None
        return state[0]

    @property
    def internal_ips(self):
        if not self.stable_internal_external_ips:
            self._update_from_sky_status()
        return [int_ip for int_ip, _ in self.stable_internal_external_ips]

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
                ssh_values = backend_utils.ssh_credential_from_yaml(
                    yaml_path, ssh_user=handle.ssh_user
                )
                if not self.creds_values:
                    from runhouse.resources.secrets.utils import setup_cluster_creds

                    self._creds = setup_cluster_creds(ssh_values, self.name)

            # Add worker IPs if multi-node cluster - keep the head node as the first IP
            self.ips = [ext for _, ext in self.stable_internal_external_ips]

            launched_resource = handle.launched_resources
            cloud = str(launched_resource.cloud).lower()
            instance_type = launched_resource.instance_type
            region = launched_resource.region
            cost_per_hr = launched_resource.get_cost(60 * 60)
            disk_size = launched_resource.disk_size
            num_cpus = launched_resource.cpus

            self.launched_properties = {
                "cloud": cloud,
                "instance_type": instance_type,
                "region": region,
                "cost_per_hour": str(cost_per_hr),
                "disk_size": disk_size,
                "num_cpus": num_cpus,
            }
            if launched_resource.accelerators:
                self.launched_properties[
                    "accelerators"
                ] = launched_resource.accelerators
            if handle.ssh_user:
                self.launched_properties["ssh_user"] = handle.ssh_user
            if handle.docker_user:
                self.launched_properties["docker_user"] = handle.docker_user
            if cloud == "kubernetes":
                try:
                    import kubernetes

                    _, context = kubernetes.config.list_kube_config_contexts()
                    if "namespace" in context["context"]:
                        namespace = context["context"]["namespace"]
                    else:
                        namespace = "default"
                except:
                    namespace = "default"
                pod_name = f"{handle.cluster_name_on_cloud}-head"
                self.launched_properties["namespace"] = namespace
                self.launched_properties["pod_name"] = pod_name
        else:
            self.address = None
            self._creds = None
            self.stable_internal_external_ips = None
            self.launched_properties = {}

    def _update_from_sky_status(self, dryrun: bool = False):
        # Try to get the cluster status from SkyDB
        if self.is_shared:
            # If the cluster is shared can ignore, since the sky data will only be saved on the machine where
            # the cluster was initially upped
            return

        cluster_dict = self._sky_status(refresh=not dryrun)
        self._populate_connection_from_status_dict(cluster_dict)

    def get_instance_type(self):
        """Returns instance type of the cluster."""
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
        """Returns the acclerator type, or None if is a CPU."""
        if (
            self.instance_type
            and ":" in self.instance_type
            and "CPU" not in self.instance_type
        ):
            return self.instance_type

        return None

    def num_cpus(self):
        """Return the number of CPUs for a CPU cluster."""
        if (
            self.instance_type
            and ":" in self.instance_type
            and "CPU" in self.instance_type
        ):
            return self.instance_type.rsplit(":", 1)[1]

        return None

    async def a_up(self, capture_output: Union[bool, str] = True):
        """Up the cluster async in another process, so it can be parallelized and logs can be captured sanely.

        capture_output: If True, supress the output of the cluster creation process. If False, print the output
        normally. If a string, write the output to the file at that path.
        """

        with ProcessPoolExecutor() as executor:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                executor, up_cluster_helper, self, capture_output
            )
        return self

    async def a_up_if_not(self, capture_output: Union[bool, str] = True):
        if not self.is_up():
            # Don't store stale IPs
            self.ips = None
            await self.a_up(capture_output=capture_output)
        return self

    def up(self):
        """Up the cluster.

        Example:
            >>> rh.ondemand_cluster("rh-cpu").up()
        """
        if self.on_this_cluster():
            return self

        supported_providers = ["cheapest"] + list(sky.clouds.CLOUD_REGISTRY)
        if self.provider not in supported_providers:
            raise ValueError(
                f"Cluster provider {self.provider} not supported. Must be one {supported_providers} supported by SkyPilot."
            )

        task = sky.Task(num_nodes=self.num_instances)
        cloud_provider = (
            sky.clouds.CLOUD_REGISTRY.from_str(self.provider)
            if self.provider != "cheapest"
            else None
        )
        try:
            task.set_resources(
                sky.Resources(
                    # TODO: confirm if passing instance type in old way (without --) works when provider is k8s
                    cloud=cloud_provider,
                    instance_type=self.get_instance_type(),
                    accelerators=self.accelerators(),
                    cpus=self.num_cpus(),
                    memory=self.memory,
                    region=self.region or configs.get("default_region"),
                    disk_size=self.disk_size,
                    ports=self.open_ports,
                    image_id=self.image_id,
                    use_spot=self.use_spot,
                    **self.sky_kwargs.get("resources", {}),
                )
            )
            if self.image_id:
                import os

                docker_env_vars = {}
                for env_var in DOCKER_LOGIN_ENV_VARS:
                    if os.getenv(env_var):
                        docker_env_vars[env_var] = os.getenv(env_var)
                if docker_env_vars:
                    task.update_envs(docker_env_vars)
            sky.launch(
                task,
                cluster_name=self.name,
                idle_minutes_to_autostop=self._autostop_mins,
                down=True,
                **self.sky_kwargs.get("launch", {}),
            )
        # Make sure no args are passed both in sky_kwargs and as explicit args
        except TypeError as e:
            if "got multiple values for keyword argument" in str(e):
                raise TypeError(
                    f"{str(e)}. If argument is in `sky_kwargs`, it may need to be passed directly through the "
                    f"ondemand_cluster constructor (see `ondemand_cluster docs "
                    f"<https://www.run.house/docs/api/python/cluster#runhouse.ondemand_cluster>`__)."
                )
            raise e

        self._update_from_sky_status()

        if self.domain:
            logger.info(
                f"Cluster has been launched with the custom domain '{self.domain}'. "
                "Please add an A record to your DNS provider to point this domain to the cluster's "
                f"public IP address ({self.address}) to ensure successful requests."
            )

        self.restart_server()

        if rns_client.autosave_resources():
            self.save()

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
        try:
            cluster_status_data = self.status()
            status_data = {
                "status": ResourceServerStatus.terminated,
                "resource_type": self.__class__.__base__.__name__.lower(),
                "data": cluster_status_data,
            }
            cluster_uri = rns_client.format_rns_address(self.rns_address)
            api_server_url = rns_client.api_server_url
            status_resp = requests.post(
                f"{api_server_url}/resource/{cluster_uri}/cluster/status",
                data=json.dumps(status_data),
                headers=rns_client.request_headers(),
            )
            # 404 means that the cluster is not saved in den, it is fine that the status is not updated.
            if status_resp.status_code not in [200, 404]:
                logger.warning("Failed to update Den with terminated cluster status")
        except Exception as e:
            logger.warning(e)

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
    def cluster_ssh_key(path_to_file: Path):
        """Retrieve SSH key for the cluster.

        Args:
            path_to_file (Path): Path of the private key associated with the cluster.

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
        """SSH into the cluster.

        Args:
            node: Node to SSH into. If no node is specified, will SSH onto the head node.
                (Default: ``None``)

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
            from runhouse.resources.hardware.sky_command_runner import SshMode

            sky_key = Path(
                self.creds_values.get("ssh_private_key", self.DEFAULT_KEYFILE)
            ).expanduser()

            if not sky_key.exists():
                raise FileNotFoundError(f"Expected default sky key in path: {sky_key}")

            runner = self._command_runner(node=node)
            cmd = runner.run(
                cmd="bash --rcfile <(echo '. ~/.bashrc; conda deactivate')",
                ssh_mode=SshMode.INTERACTIVE,
                port_forward=None,
                return_cmd=True,
            )
            subprocess.run(cmd, shell=True)

    def _ping(self, timeout=5, retry=False):
        if super()._ping(timeout=timeout, retry=False):
            return True

        if retry:
            self._update_from_sky_status(dryrun=False)
            return super()._ping(timeout=timeout, retry=False)
        return False
