import logging
import os
import platform
import subprocess
import tarfile
import time
import urllib
from builtins import bool
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import psutil
import requests
import yaml

from runhouse.constants import (
    TELEMETRY_AGENT_GRPC_PORT,
    TELEMETRY_AGENT_HEALTH_CHECK_PORT,
    TELEMETRY_AGENT_HTTP_PORT,
    TELEMETRY_COLLECTOR_ENDPOINT,
    TELEMETRY_COLLECTOR_HOST,
    TELEMETRY_COLLECTOR_STATUS_URL,
)
from runhouse.globals import configs

from runhouse.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TelemetryAgentConfig:
    http_port: int = TELEMETRY_AGENT_HTTP_PORT
    grpc_port: int = TELEMETRY_AGENT_GRPC_PORT
    health_check_port: int = TELEMETRY_AGENT_HEALTH_CHECK_PORT
    log_level: str = field(
        default_factory=lambda: logging.getLevelName(logger.getEffectiveLevel())
    )


@dataclass
class TelemetryCollectorConfig:
    endpoint: str = TELEMETRY_COLLECTOR_ENDPOINT
    status_url: str = TELEMETRY_COLLECTOR_STATUS_URL


class TelemetryAgentExporter:
    """Runs a local OpenTelemetry receiver instance for telemetry collection

    Key actions:
    - Installs the OpenTelemetry Collector binary (if not already present)
    - Creates a config file for the agent
    - Starts the agent exporter background process on localhost
    - Listens for incoming telemetry data on specified ports
    - Exports telemetry data to the backend collector
    """

    BASE_DIR = Path.home() / ".otel"

    def __init__(
        self,
        agent_config: TelemetryAgentConfig = None,
        collector_config: TelemetryCollectorConfig = None,
    ):
        self._agent_process = None
        self.agent_config = agent_config or TelemetryAgentConfig()
        self.collector_config = collector_config or TelemetryCollectorConfig()

        self._setup_directories()

    @property
    def bin_dir(self) -> Path:
        return self.BASE_DIR / "bin"

    @property
    def config_dir(self) -> Path:
        return self.BASE_DIR / "config"

    @property
    def log_dir(self) -> Path:
        return self.BASE_DIR / "logs"

    @property
    def executable_path(self) -> str:
        """Path to the otel binary."""
        return f"{self.bin_dir}/otelcol-contrib"

    @property
    def local_config_path(self) -> str:
        """Path to the agent config file."""
        return f"{self.config_dir}/config.yaml"

    @property
    def agent_status_url(self) -> str:
        """Health check URL of the local agent."""
        return f"http://localhost:{self.agent_config.health_check_port}"

    @classmethod
    def auth_token(cls):
        # Use the token saved in the local config file (~/.rh/config.yaml)
        # Note: this will be the cluster token for the cluster owner
        cluster_token = configs.token
        if cluster_token is None:
            raise ValueError(
                "No cluster token found, cannot configure telemetry agent auth"
            )
        return cluster_token

    @classmethod
    def request_headers(cls):
        return {"authorization": f"Bearer {cls.auth_token()}"}

    def _setup_directories(self):
        # Note: use paths that are local to the user's home directory and won't necessitate root access on the cluster
        for dir_name in ["bin", "config", "logs"]:
            (self.BASE_DIR / dir_name).mkdir(parents=True, exist_ok=True)

    def _get_verbosity(self) -> str:
        log_level = self.agent_config.log_level.lower()
        if log_level in ["debug", "trace"]:
            return "detailed"
        elif log_level in ["info", "warn", "warning"]:
            return "normal"
        else:  # 'error', 'critical', etc.
            return "basic"

    def _create_default_config(self):
        """Base config for the local agent, which forwards the collected telemetry data to the collector backend."""
        collector_endpoint = self.collector_config.endpoint

        # Use insecure connection if the collector is not secured with HTTPS (ex: running on localhost)
        insecure = False if TELEMETRY_COLLECTOR_HOST in collector_endpoint else True
        service_extensions = ["health_check"]

        # Note: the receiver does not have any auth enabled, as the agent will be running on localhost
        # The auth is configured on the "exporter", since auth is required to send data to the collector
        auth_extension = {}
        if not insecure:
            auth_extension = {
                "bearertokenauth/withscheme": {
                    "scheme": "Bearer",
                    "token": TelemetryAgentExporter.auth_token(),
                }
            }
            service_extensions.append("bearertokenauth/withscheme")

        otel_config = {
            "extensions": {
                "health_check": {
                    "endpoint": f"0.0.0.0:{self.agent_config.health_check_port}"
                },
                **auth_extension,
            },
            "receivers": {
                "otlp": {
                    "protocols": {
                        "grpc": {"endpoint": f"0.0.0.0:{self.agent_config.grpc_port}"},
                        "http": {"endpoint": f"0.0.0.0:{self.agent_config.http_port}"},
                    }
                }
            },
            "processors": {"batch": {}},
            "exporters": {
                "debug": {"verbosity": self._get_verbosity()},
                "otlp/grpc": {
                    "endpoint": collector_endpoint,
                    "tls": {"insecure": insecure},
                    **(
                        {"auth": {"authenticator": "bearertokenauth/withscheme"}}
                        if not insecure
                        else {}
                    ),
                },
            },
            "service": {
                "extensions": service_extensions,
                "pipelines": {
                    "traces": {
                        "receivers": ["otlp"],
                        "processors": ["batch"],
                        "exporters": ["debug", "otlp/grpc"],
                    },
                    "metrics": {
                        "receivers": ["otlp"],
                        "processors": ["batch"],
                        "exporters": ["debug", "otlp/grpc"],
                    },
                    "logs": {
                        "receivers": ["otlp"],
                        "processors": ["batch"],
                        "exporters": ["debug", "otlp/grpc"],
                    },
                },
            },
        }

        with open(self.local_config_path, "w") as f:
            yaml.dump(otel_config, f, default_flow_style=False)

    def _generate_install_url(self):
        """Generate the download URL for the agent binary based on the host system."""
        # https://opentelemetry.io/docs/collector/installation/
        system = platform.system().lower()
        arch = platform.machine()
        if system == "linux":
            # https://opentelemetry.io/docs/collector/installation/#linux
            if arch in ["x86_64", "amd64"]:
                arch = "amd64"
            elif arch in ["aarch64", "arm64"]:
                arch = "arm64"
            else:
                raise ValueError(
                    f"Unsupported Linux architecture for OpenTelemetry: {arch}"
                )
        elif system == "darwin":
            if arch == "x86_64":
                arch = "amd64"
            elif arch == "arm64":
                arch = "arm64"
            else:
                raise ValueError(f"Unsupported macOS architecture: {arch}")
        else:
            raise ValueError(f"Unsupported system: {system}")

        # Note: version used by agent must match the collector version
        binary_url = f"https://github.com/open-telemetry/opentelemetry-collector-releases/releases/download/v0.108.0/otelcol-contrib_0.108.0_{system}_{arch}.tar.gz"

        return binary_url

    def _find_existing_agent_process(self):
        """Finds the running agent process by name"""
        for proc in psutil.process_iter(["pid", "name"]):
            try:
                if proc.info["name"] == "otelcol":
                    return proc
            except (
                psutil.NoSuchProcess,
                psutil.AccessDenied,
                psutil.TimeoutExpired,
                psutil.ZombieProcess,
            ):
                pass

    def collector_health_check(self) -> int:
        """Ping the health status of the backend collector - should return a status code of 200 if healthy."""
        try:
            resp = requests.get(self.collector_config.status_url)
            return resp.status_code
        except requests.ConnectionError:
            # Service unavailable - likely not up
            return 503

    def agent_health_check(self) -> int:
        """Ping the health status of the local agent - should return a status code of 200 if healthy."""
        try:
            resp = requests.get(self.agent_status_url)
            return resp.status_code
        except requests.ConnectionError:
            # Service unavailable - likely not up
            return 503

    def is_up(self) -> bool:
        """
        Checks if the local agent process is running, and that the agent health check returns a status of 200.

        Accounts for the python object holding a reference to a process it started, or an existing process
        that was previously started that needs to be found.

        In both cases, we maintain a ref to the running process.

        Returns:
            bool: True if the agent process is running, False otherwise.
        """
        agent_status_code = self.agent_health_check()
        if agent_status_code != 200:
            return False

        if self._agent_process:
            return psutil.pid_exists(self._agent_process.pid)

        proc = self._find_existing_agent_process()
        if not proc:
            return False

        self._agent_process = proc
        return True

    def is_installed(self) -> bool:
        """Check if the binary path exists on the host machine."""
        return Path(self.executable_path).exists()

    def install(self):
        """Install the binary for the telemetry agent."""
        logger.debug("Installing OTel agent")
        try:
            install_url = self._load_install_url()
            logger.debug(f"Downloading OTel agent from url: {install_url}")

            # Download and extract
            tar_path = "/tmp/otelcol.tar.gz"
            urllib.request.urlretrieve(install_url, tar_path)

            local_agent_dir = self.bin_dir

            with tarfile.open(tar_path, "r:gz") as tar:
                tar.extractall(path=local_agent_dir)

            # Verify installation
            if not self.is_installed():
                raise FileExistsError(
                    f"No OTel binary found in path: {self.executable_path}"
                )

            logger.info("OTel agent installed successfully")

        except Exception as e:
            raise RuntimeError(f"Failed to install OTel agent: {e}")

    def start(self, force_reinstall=False, reload_config=False, timeout=10):
        """
        Start the local telemetry agent as a background process.

        Args:
            force_reinstall (Optional[bool]): If True, reinstall the agent even if it's already installed.
                (Default: ``False``)
            reload_config (Optional[bool]): If True, reload the agent config even if it exists. (Default: ``False``)
            timeout (Optional[int]): Maximum time to wait for the agent to start, in seconds. (Default: 30)

        Returns:
            bool: True if the agent was successfully started, False otherwise.
        """
        if force_reinstall or not self.is_installed():
            self.install()

        if self.is_up() and not reload_config:
            logger.debug("Otel agent is already running")
            return True

        try:
            config_path = self.local_config_path
            if reload_config or not Path(config_path).exists():
                self._create_default_config()

            log_file = os.path.join(self.log_dir, "agent.log")

            logger.debug(f"Starting OpenTelemetry agent at {datetime.now()}\n")

            # Overwrite the file each time the agent is restarted
            with open(log_file, "w") as out_file:
                self._agent_process = subprocess.Popen(
                    [self.executable_path, "--config", config_path],
                    stdout=out_file,
                    stderr=subprocess.STDOUT,
                    text=True,
                )

            # Wait for the process to start
            start_time = datetime.now()
            while (datetime.now() - start_time).total_seconds() < timeout:
                if self.is_up():
                    logger.info(
                        f"Successfully started Otel agent (pid={self._agent_process.pid})"
                    )
                    return True

                time.sleep(0.5)
            raise TimeoutError(
                f"Otel agent failed to start within the specified timeout ({timeout} seconds)"
            )

        except (PermissionError, subprocess.CalledProcessError, TimeoutError) as e:
            logger.error(f"Failed to start Otel agent: {e}")
            return False

    def stop(self):
        """
        Stop the local telemetry agent.

        Returns:
            bool: True if a process was stopped, False otherwise.
        """
        process_to_stop = self._agent_process or self._find_existing_agent_process()
        if not process_to_stop:
            logger.info("No running Otel agent found")
            return False

        try:
            process_to_stop.terminate()
            # Wait for up to 5 seconds for the process to terminate
            process_to_stop.wait(timeout=5)
            logger.info(f"Stopped Otel agent (pid={process_to_stop.pid})")
            self._agent_process = None  # Clear the reference
            return True

        except psutil.NoSuchProcess:
            logger.info("Otel agent no longer running")
            return True

        except psutil.TimeoutExpired:
            process_to_stop.kill()  # Force kill if it doesn't terminate
            logger.info("Stopped the Otel agent")
            return True

        except Exception as e:
            logger.error(f"Error stopping Otel agent: {e}")
            return False
