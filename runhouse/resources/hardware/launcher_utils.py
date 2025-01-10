import ast
import logging
import sys
from typing import Any, Optional

import requests

import runhouse as rh
from runhouse.globals import configs, rns_client
from runhouse.logger import get_logger
from runhouse.resources.hardware.utils import (
    _cluster_set_autostop_command,
    ClusterStatus,
    SSEClient,
)
from runhouse.rns.utils.api import generate_ssh_keys, load_resp_content, read_resp_data
from runhouse.utils import ClusterLogsFormatter, Spinner

logger = get_logger(__name__)


class LogProcessor:
    """Dedicated logger for handling streamed cluster logs"""

    def __init__(self, cluster_name: str):
        self.cluster_name = cluster_name
        self.cluster_logger = self._init_cluster_logger()

    def _init_cluster_logger(self):
        cluster_logger = logging.getLogger("cluster_logs")
        log_formatter = ClusterLogsFormatter(system=self.cluster_name)

        cluster_logger.setLevel(logging.DEBUG)

        # Add a handler with a simple formatter if not already set
        if not cluster_logger.handlers:
            system_color, reset_color = log_formatter.format_launcher_log()

            class PrependColorFormatter(logging.Formatter):
                """Custom formatter to add colors to log messages."""

                def __init__(self, system_color, reset_color, fmt="%(message)s"):
                    super().__init__(fmt)
                    self.system_color = system_color
                    self.reset_color = reset_color

                def format(self, record):
                    message = super().format(record)
                    return f"{self.system_color}{message}{self.reset_color}"

            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(PrependColorFormatter(system_color, reset_color))
            cluster_logger.addHandler(handler)

        # Prevent propagation to root_logger
        cluster_logger.propagate = False

        return cluster_logger

    def log_event(self, event):
        """Log the event at the appropriate level using the cluster logger."""
        # Note: we need to remove logger prefix from the local logger to just include the log in Den
        data = ast.literal_eval(event.data)
        log_level = data["log_level"].upper()
        log_message = data["log"]

        if log_level == "INFO":
            self.cluster_logger.info(log_message)
        elif log_level == "WARNING":
            self.cluster_logger.warning(log_message)
        elif log_level == "ERROR":
            self.cluster_logger.error(log_message)
        elif log_level == "EXCEPTION":
            self.cluster_logger.exception(log_message)
        elif log_level == "DEBUG":
            self.cluster_logger.debug(log_message)
        else:
            self.cluster_logger.info(log_message)


class Launcher:
    @classmethod
    def log_processor(cls, cluster_name: str):
        return LogProcessor(cluster_name)

    @classmethod
    def up(cls, cluster, verbose: bool = True):
        """Abstract method for launching a cluster."""
        raise NotImplementedError

    @classmethod
    def teardown(cls, cluster, verbose: bool = True):
        """Abstract method for tearing down a cluster."""
        raise NotImplementedError

    @classmethod
    def keep_warm(cls, cluster, mins: int):
        """Abstract method for keeping a cluster warm."""
        raise NotImplementedError

    @staticmethod
    def supported_providers():
        """Return the base list of Sky supported providers."""
        import sky

        return list(sky.clouds.CLOUD_REGISTRY)

    @classmethod
    def sky_secret(cls):
        from runhouse.constants import SSH_SKY_SECRET_NAME

        try:
            sky_secret = rh.secret(SSH_SKY_SECRET_NAME)
        except ValueError:
            # Create a new default key pair required for the Den launcher and save it to Den
            from runhouse import provider_secret

            default_ssh_path, _ = generate_ssh_keys()
            logger.info(f"Saved new SSH key to path: {default_ssh_path} ")
            sky_secret = provider_secret(
                provider="sky", path=default_ssh_path, name=SSH_SKY_SECRET_NAME
            )
            sky_secret.save()

        secret_values = sky_secret.values
        if (
            not secret_values
            or "public_key" not in secret_values
            or "private_key" not in secret_values
        ):
            raise ValueError(
                f"Public key and private key values not found in secret {secrets_name}"
            )
        return sky_secret

    @classmethod
    def run_verbose(
        cls,
        base_url: str,
        cluster_name: str,
        payload: dict = None,
    ) -> Any:
        """Call a specified Den API while streaming logs back using an SSE client."""
        resp = requests.post(
            base_url,
            json=payload,
            headers=rns_client.request_headers(),
            stream=True,
        )

        if resp.status_code != 200:
            raise Exception(
                f"Received [{resp.status_code}] from Den POST '{base_url}': {load_resp_content(resp)}"
            )

        client = SSEClient(resp)
        spinner: Optional[Spinner] = None
        log_processor = cls.log_processor(cluster_name)
        data = {}

        for event in client.events():
            # Stream through data events
            if spinner:
                spinner.stop()
                spinner = None

            if event.event == "log_spinner":
                spinner = Spinner(logger=logger, desc=str(event.data))
                spinner.start()

            if event.event == "info_spinner":
                logger.info(event.data)
                spinner = Spinner(logger=logger, desc=str(event.data))
                spinner.start()

            if event.event == "info":
                logger.info(event.data)

            if event.event == "log":
                log_processor.log_event(event)

            if event.event == "error":
                event_data = ast.literal_eval(event.data)
                raise Exception(
                    f"Received [{event_data.get('code')}] from Den POST '{base_url}': {event_data.get('detail')}"
                )

            if event.event == "end":
                # End returns data for continuing this method
                logger.info("Successfully ran cluster operation via Den")
                data = ast.literal_eval(event.data)
                break

        return data


class DenLauncher(Launcher):
    """Launcher APIs for operations handled remotely via Den."""

    LAUNCH_URL = f"{rns_client.api_server_url}/cluster/up"
    TEARDOWN_URL = f"{rns_client.api_server_url}/cluster/teardown"
    AUTOSTOP_URL = f"{rns_client.api_server_url}/cluster/autostop"

    @classmethod
    def _update_from_den_response(cls, cluster, config: dict):
        """Updates cluster with config from Den. Only add fields if found."""
        if not config:
            return

        for attribute in [
            "compute_properties",
            "ssh_properties",
            "client_port",
        ]:
            value = config.get(attribute)
            if value:
                setattr(cluster, attribute, value)

        creds = config.get("creds")
        if not cluster._creds and creds:
            cluster._setup_creds(creds)

    @classmethod
    def keep_warm(cls, cluster, mins: int):
        """Keeping a Den launched cluster warm."""
        payload = {
            "cluster_name": cluster.rns_address or cluster.name,
            "autostop_mins": mins,
        }

        # Blocking call with no streaming
        resp = requests.post(
            cls.AUTOSTOP_URL,
            json=payload,
            headers=rns_client.request_headers(),
        )
        if resp.status_code != 200:
            raise Exception(
                f"Received [{resp.status_code}] from Den POST '{cls.AUTOSTOP_URL}': Failed to "
                f"update cluster autostop: {load_resp_content(resp)}"
            )
        logger.info("Successfully updated cluster autostop")

    @classmethod
    def up(cls, cluster, verbose: bool = True, force: bool = False):
        """Launch the cluster via Den."""
        sky_secret = cls.sky_secret()
        cluster._setup_creds(sky_secret)
        cluster.save()

        cluster_config = cluster.config()

        payload = {
            "cluster_config": {
                **cluster_config,
                "ssh_creds": sky_secret.rns_address,
            },
            "force": force,
            "verbose": verbose,
            "observability": configs.observability_enabled,
        }

        if verbose:
            data = cls.run_verbose(
                base_url=cls.LAUNCH_URL,
                payload=payload,
                cluster_name=cluster_config.get("name"),
            )
            cluster._cluster_status = ClusterStatus.RUNNING
            cls._update_from_den_response(cluster=cluster, config=data)
            logger.info(
                f"To view this cluster in Den, visit https://run.house/resources/{rns_client.format_rns_address(cluster.rns_address)}"
            )
            return

        # Blocking call with no streaming
        resp = requests.post(
            cls.LAUNCH_URL,
            json=payload,
            headers=rns_client.request_headers(),
        )
        if resp.status_code != 200:
            raise Exception(
                f"Received [{resp.status_code}] from Den POST '{cls.LAUNCH_URL}': Failed to "
                f"launch cluster: {load_resp_content(resp)}"
            )
        data = read_resp_data(resp)
        logger.info("Successfully launched cluster")
        logger.info(
            f"To view this cluster in Den, visit https://run.house/resources/{rns_client.format_rns_address(cluster.rns_address)}"
        )
        cluster._cluster_status = ClusterStatus.RUNNING
        cls._update_from_den_response(cluster=cluster, config=data)

    @classmethod
    def teardown(cls, cluster, verbose: bool = True):
        """Tearing down a cluster via Den."""
        from runhouse.resources.secrets import Secret

        ssh_creds = cluster._creds
        if isinstance(ssh_creds, Secret):
            ssh_creds = ssh_creds.rns_address

        cluster_name = cluster.rns_address or cluster.name

        payload = {
            "cluster_name": cluster_name,
            "delete_from_den": False,
            "ssh_creds": ssh_creds,
            "verbose": verbose,
        }

        if verbose:
            cls.run_verbose(
                base_url=cls.TEARDOWN_URL,
                cluster_name=cluster_name,
                payload=payload,
            )
            cluster._cluster_status = ClusterStatus.TERMINATED
            return

        # Run blocking call, with no streaming
        resp = requests.post(
            cls.TEARDOWN_URL,
            json=payload,
            headers=rns_client.request_headers(),
        )
        if resp.status_code != 200:
            raise Exception(
                f"Received [{resp.status_code}] from Den POST '{cls.TEARDOWN_URL}': Failed to "
                f"teardown cluster: {load_resp_content(resp)}"
            )
        cluster._cluster_status = ClusterStatus.TERMINATED


class LocalLauncher(Launcher):
    """Launcher APIs for operations handled locally via Sky."""

    @classmethod
    def _validate_provider(cls, cluster):
        """Check if LocalLauncher supports the provided cloud provider."""
        supported_providers = ["cheapest"] + cls.supported_providers()
        if cluster.provider not in supported_providers:
            raise ValueError(
                f"Cluster provider {cluster.provider} not supported. "
                f"Must be one of {supported_providers} supported by SkyPilot."
            )

    @classmethod
    def up(cls, cluster, verbose: bool = True):
        """Launch the cluster locally."""
        import sky

        cls._validate_provider(cluster)

        task = sky.Task(num_nodes=cluster.num_nodes)
        cloud_provider = (
            sky.clouds.CLOUD_REGISTRY.from_str(cluster.provider)
            if cluster.provider != "cheapest"
            else None
        )

        try:
            task.set_resources(
                sky.Resources(
                    cloud=cloud_provider,
                    instance_type=cluster.get_instance_type(),
                    accelerators=cluster.gpus(),
                    cpus=cluster.num_cpus(),
                    memory=cluster.memory,
                    region=cluster.region or configs.get("default_region"),
                    disk_size=cluster.disk_size,
                    ports=cluster.open_ports,
                    image_id=cluster.image_id,
                    use_spot=cluster.use_spot,
                    **cluster.sky_kwargs.get("resources", {}),
                )
            )
            if cluster.image_id:
                cls._set_docker_env_vars(cluster.image, task)

            sky.launch(
                task,
                cluster_name=cluster.name,
                idle_minutes_to_autostop=cluster._autostop_mins,
                down=True,
                **cluster.sky_kwargs.get("launch", {}),
            )

            cluster._update_from_sky_status()
            if cluster.domain:
                logger.info(
                    f"Cluster has been launched with the custom domain '{cluster.domain}'. "
                    "Please add an A record to your DNS provider to point this domain to the cluster's "
                    f"public IP address ({cluster.head_ip}) to ensure successful requests."
                )

            if rns_client.autosave_resources():
                logger.debug("Saving cluster to Den")
                cluster.save()

        except TypeError as e:
            if "got multiple values for keyword argument" in str(e):
                raise TypeError(
                    f"{str(e)}. If argument is in `sky_kwargs`, it may need to be passed directly through the "
                    f"ondemand_cluster constructor (see `ondemand_cluster docs "
                    f"<https://www.run.house/docs/api/python/cluster#runhouse.ondemand_cluster>`__)."
                )
            raise e

    @classmethod
    def teardown(cls, cluster, verbose: bool = True):
        """Tearing down a cluster locally via Sky."""
        import sky

        sky.down(cluster.name)
        cluster._cluster_status = ClusterStatus.TERMINATED
        cluster._http_client = None

        # Save to Den with updated null IPs
        if rns_client.autosave_resources():
            cluster.save()

        logger.info("Successfully terminated cluster")

    @classmethod
    def keep_warm(cls, cluster, mins: int):
        """Keeping a locally launched cluster warm."""
        try:
            import sky

            sky.autostop(cluster.name, mins, down=True)
        except ImportError:
            set_cluster_autostop_cmd = _cluster_set_autostop_command(mins)
            cluster.run_bash_over_ssh([set_cluster_autostop_cmd], node=cluster.head_ip)

    @staticmethod
    def _set_docker_env_vars(image, task):
        """Helper method to set Docker login environment variables."""
        docker_secret = image.docker_secret if image else None
        if docker_secret:
            if isinstance(image.docker_secret, str):
                from runhouse.resources.secrets.secret import Secret

                docker_secret = Secret.from_name(image.docker_secret)
            docker_env_vars = image.docker_secret._map_env_vars()
        else:
            try:
                docker_env_vars = rh.provider_secret("docker")._map_env_vars()
            except ValueError:
                docker_env_vars = {}

        if docker_env_vars:
            task.update_envs(docker_env_vars)
