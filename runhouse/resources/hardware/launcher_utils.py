import json
from asyncio import Event

import requests

from runhouse.constants import DOCKER_LOGIN_ENV_VARS
from runhouse.globals import configs, rns_client
from runhouse.logger import get_logger
from runhouse.rns.utils.api import load_resp_content

logger = get_logger(__name__)


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

    def run_verbose(self, base_url: str, logs_url: str, payload: dict = None):
        """Call a specified Den API while streaming logs back from a separate logs endpoint."""
        from asyncio import Event
        from concurrent.futures import ThreadPoolExecutor

        # Create an event to stop the log streaming thread when finished
        stop_event = Event()

        with ThreadPoolExecutor() as executor:
            # Send request to the Den URL specified
            resp = requests.post(
                base_url,
                json=payload or self.cluster.config(),
                headers=rns_client.request_headers(),
            )

            if resp.status_code != 200:
                # Stop the log thread on failure
                stop_event.set()
                raise Exception(
                    f"Received [{resp.status_code}] from Den POST '{base_url}': {load_resp_content(resp)}"
                )

            resp_data = resp.json().get("data", {})
            launch_id = resp_data.get("launch_id")
            temp_dir = resp_data.get("temp_dir")

            # Start the log streaming in a separate thread
            executor.submit(
                self.load_logs_in_thread, stop_event, logs_url, temp_dir, launch_id
            )

            # Stop the log streaming once the request is done
            stop_event.set()

    def load_logs_in_thread(
        self, stop_event: Event, url: str, temp_dir: str, launch_id: str
    ):
        import asyncio

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(
            self.stream_logs_from_url(stop_event, url, temp_dir, launch_id)
        )

    @staticmethod
    async def stream_logs_from_url(
        stop_event: Event, url: str, temp_dir: str, launch_id: str
    ):
        """Load logs returned from the specified Den URL. The response should be a stream of JSON logs."""
        import httpx

        client = httpx.AsyncClient(timeout=None)

        async with client.stream(
            "POST",
            url,
            json={"temp_dir": temp_dir, "launch_id": launch_id},
            headers=rns_client.request_headers(),
        ) as res:
            if res.status_code != 200:
                error_resp = await res.aread()
                raise ValueError(f"Error calling Den logs API: {error_resp.decode()}")

            async for response_json in res.aiter_lines():
                if stop_event.is_set():
                    break
                resp = json.loads(response_json)

                # TODO [JL] any formatting to do here?
                print(resp)

        await client.aclose()


class DenLauncher(Launcher):
    """Launcher APIs for operations handled remotely via Den."""

    LAUNCH_URL = f"{rns_client.api_server_url}/cluster/up"
    TEARDOWN_URL = f"{rns_client.api_server_url}/cluster/down"

    # TODO update these URLs
    LAUNCH_LOGS_URL = f"{rns_client.api_server_url}/cluster/logs/up"
    TEARDOWN_LOGS_URL = f"{rns_client.api_server_url}/cluster/logs/teardown"

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
        if verbose:
            config = self.cluster.config()
            config["force"] = self.force
            self.run_verbose(
                base_url=self.LAUNCH_URL, logs_url=self.LAUNCH_LOGS_URL, payload=config
            )
        else:
            # Blocking call with no streaming
            resp = requests.post(
                self.LAUNCH_URL,
                json={**self.cluster.config(), "force": self.force},
                headers=rns_client.request_headers(),
            )
            if resp.status_code != 200:
                raise Exception(
                    f"Received [{resp.status_code}] from Den POST '{self.LAUNCH_URL}': Failed to "
                    f"launch cluster: {load_resp_content(resp)}"
                )

        logger.info("Successfully launched cluster.")

    def teardown(self, verbose: bool = True):
        """Tearing down a cluster via Den."""
        # TODO [MK] what payload do we need here?
        if verbose is False:
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
            return

        self.run_verbose(base_url=self.TEARDOWN_URL, logs_url=self.TEARDOWN_LOGS_URL)


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

        self.cluster._update_from_sky_status()

        if self.cluster.domain:
            logger.info(
                f"Cluster has been launched with the custom domain '{self.cluster.domain}'. "
                "Please add an A record to your DNS provider to point this domain to the cluster's "
                f"public IP address ({self.cluster.address}) to ensure successful requests."
            )

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
