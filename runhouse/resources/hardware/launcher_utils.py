import ast

import requests

import runhouse as rh
from runhouse.constants import DOCKER_LOGIN_ENV_VARS
from runhouse.globals import configs, rns_client
from runhouse.logger import get_logger
from runhouse.resources.hardware.utils import SSEClient
from runhouse.rns.utils.api import load_resp_content

logger = get_logger(__name__)


class SSEHandler:
    def __init__(self, event):
        self.event = event


class Launcher:
    def __init__(self, cluster):
        self.cluster = cluster

    def up(self, verbose: bool = True):
        """Abstract method for launching a cluster."""
        raise NotImplementedError

    def teardown(self, verbose: bool = True):
        """Abstract method for tearing down a cluster."""
        raise NotImplementedError

    @staticmethod
    def supported_providers():
        """Return the base list of Sky supported providers."""
        import sky

        return list(sky.clouds.CLOUD_REGISTRY)

    def run_verbose(self, base_url: str, payload: dict = None):
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

        data = {}
        for event in client.events():
            # Stream through data events
            # if event.event == "data":
            #     print(event.data)

            # TODO: Properly send, recieve, and format logs
            print(event.data)

            if event.event == "error":
                event_data = ast.literal_eval(event.data)
                raise Exception(
                    f"Received [{event_data.get('code')}] from Den POST '{base_url}': {event_data.get('detail')}"
                )

            # End returns data for continuing this method
            if event.event == "end":
                logger.info("Successfully ran cluster operation via Den")
                data = ast.literal_eval(event.data)
                break

        return data


class DenLauncher(Launcher):
    """Launcher APIs for operations handled remotely via Den."""

    LAUNCH_URL = f"{rns_client.api_server_url}/cluster/up"
    TEARDOWN_URL = f"{rns_client.api_server_url}/cluster/down"

    def __init__(self, cluster, force: bool = False):
        super().__init__(cluster)
        self._validate_provider()
        self.force = force

    def _validate_provider(self):
        """Ensure that the provider is supported."""
        if self.cluster.provider == "cheapest":
            raise ValueError(
                "Cheapest not currently supported for Den launcher. Please specify a cloud provider."
            )

        supported_providers = self.supported_providers()
        if self.cluster.provider not in supported_providers:
            raise ValueError(
                f"Cluster provider {self.cluster.provider} not supported. "
                f"Must be one of {supported_providers} supported by SkyPilot."
            )

    def up(self, verbose: bool = True):
        """Launch the cluster via Den."""
        # TODO: is this guaranteed to be the default SkySecret name?
        secrets_name = "ssh-sky-key"
        try:
            sky_secret = rh.secret(secrets_name)
        except ValueError:
            raise ValueError(f"No cluster secret found in Den with name {secrets_name}")

        secret_values = sky_secret.values
        if (
            not secret_values
            or "public_key" not in secret_values
            or "private_key" not in secret_values
        ):
            raise ValueError(
                f"Public key and private key values not found in secret {secrets_name}"
            )

        payload = {
            "cluster_config": {
                **self.cluster.config(),
                # TODO: update once secrets are updated to include pub + private key values (should have only one field)
                "ssh_creds": "/mkandler/aws-cpu-matt-ssh-secret",
                "sky_creds": sky_secret.rns_address,
            },
            "force": self.force,
        }

        if verbose:
            data = self.run_verbose(base_url=self.LAUNCH_URL, payload=payload)
            logger.info("Successfully launched cluster.")
            return data

        # Blocking call with no streaming
        resp = requests.post(
            self.LAUNCH_URL,
            json=payload,
            headers=rns_client.request_headers(),
        )
        if resp.status_code != 200:
            raise Exception(
                f"Received [{resp.status_code}] from Den POST '{self.LAUNCH_URL}': Failed to "
                f"launch cluster: {load_resp_content(resp)}"
            )
        data = resp.json()
        logger.info("Successfully launched cluster.")
        return data

    def teardown(self, verbose: bool = True):
        """Tearing down a cluster via Den."""
        # TODO [MK] what payload do we need here?
        if verbose:
            self.run_verbose(base_url=self.TEARDOWN_URL)
            return

        # Run blocking call, with no streaming
        resp = requests.post(
            self.TEARDOWN_URL,
            json=self.cluster.config(),
            headers=rns_client.request_headers(),
        )
        if resp.status_code != 200:
            raise Exception(
                f"Received [{resp.status_code}] from Den POST '{self.TEARDOWN_URL}': Failed to "
                f"teardown cluster: {load_resp_content(resp)}"
            )


class LocalLauncher(Launcher):
    """Launcher APIs for operations handled locally via Sky."""

    def __init__(self, cluster):
        super().__init__(cluster)
        self._validate_provider()

    def _validate_provider(self):
        """Check if LocalLauncher supports the provided cloud provider."""
        supported_providers = ["cheapest"] + self.supported_providers()
        if self.cluster.provider not in supported_providers:
            raise ValueError(
                f"Cluster provider {self.cluster.provider} not supported. "
                f"Must be one of {supported_providers} supported by SkyPilot."
            )

    def up(self, verbose: bool = True):
        """Launch the cluster locally."""
        import sky

        task = sky.Task(num_nodes=self.cluster.num_instances)
        cloud_provider = (
            sky.clouds.CLOUD_REGISTRY.from_str(self.cluster.provider)
            if self.cluster.provider != "cheapest"
            else None
        )

        try:
            task.set_resources(
                sky.Resources(
                    cloud=cloud_provider,
                    instance_type=self.cluster.get_instance_type(),
                    accelerators=self.cluster.accelerators(),
                    cpus=self.cluster.num_cpus(),
                    memory=self.cluster.memory,
                    region=self.cluster.region or configs.get("default_region"),
                    disk_size=self.cluster.disk_size,
                    ports=self.cluster.open_ports,
                    image_id=self.cluster.image_id,
                    use_spot=self.cluster.use_spot,
                    **self.cluster.sky_kwargs.get("resources", {}),
                )
            )
            if self.cluster.image_id:
                self._set_docker_env_vars(task)

            sky.launch(
                task,
                cluster_name=self.cluster.name,
                idle_minutes_to_autostop=self.cluster._autostop_mins,
                down=True,
                **self.cluster.sky_kwargs.get("launch", {}),
            )
        except TypeError as e:
            if "got multiple values for keyword argument" in str(e):
                raise TypeError(
                    f"{str(e)}. If argument is in `sky_kwargs`, it may need to be passed directly through the "
                    f"ondemand_cluster constructor (see `ondemand_cluster docs "
                    f"<https://www.run.house/docs/api/python/cluster#runhouse.ondemand_cluster>`__)."
                )
            raise e

    def teardown(self, verbose: bool = True):
        """Tearing down a cluster locally via Sky."""
        import sky

        sky.down(self.cluster.name)

    @staticmethod
    def _set_docker_env_vars(task):
        """Helper method to set Docker login environment variables."""
        import os

        docker_env_vars = {}
        for env_var in DOCKER_LOGIN_ENV_VARS:
            if os.getenv(env_var):
                docker_env_vars[env_var] = os.getenv(env_var)

        if docker_env_vars:
            task.update_envs(docker_env_vars)
