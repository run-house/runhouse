import json

from enum import Enum
from pathlib import Path
from typing import Dict

from runhouse.constants import (
    CLUSTER_CONFIG_PATH,
    EMPTY_DEFAULT_ENV_NAME,
    RESERVED_SYSTEM_NAMES,
)
from runhouse.resources.envs.utils import _get_env_from, run_setup_command


class ServerConnectionType(str, Enum):
    """Manage the type of connection Runhouse will make with the API server started on the cluster.
    ``ssh``: Use port forwarding to connect to the server via SSH, by default on port 32300.
    ``tls``: Do not use port forwarding and start the server with HTTPS (using custom or fresh TLS certs), by default
        on port 443.
    ``none``: Do not use port forwarding, and start the server with HTTP, by default on port 80.
    ``aws_ssm``: Use AWS SSM to connect to the server, by default on port 32300.
    """

    SSH = "ssh"
    TLS = "tls"
    NONE = "none"
    AWS_SSM = "aws_ssm"


class ResourceServerStatus(str, Enum):
    running = "running"
    terminated = "terminated"
    unauthorized = "unauthorized"
    unknown = "unknown"
    internal_server_error = "internal_server_error"
    runhouse_daemon_down = "runhouse_daemon_down"
    invalid_url = "invalid_url"


def cluster_config_file_exists() -> bool:
    return Path(CLUSTER_CONFIG_PATH).expanduser().exists()


def load_cluster_config_from_file() -> Dict:
    if cluster_config_file_exists():
        with open(Path(CLUSTER_CONFIG_PATH).expanduser()) as f:
            cluster_config = json.load(f)
        return cluster_config
    else:
        return {}


def _current_cluster(key="config"):
    """Retrieve key value from the current cluster config.
    If key is "config", returns entire config."""
    from runhouse.globals import obj_store

    cluster_config = obj_store.get_cluster_config()
    cluster_config.pop("creds", None)
    if cluster_config:
        # This could be a local cluster started via runhouse start,
        # in which case it would have no Name.
        if key in ["cluster_name", "name"] and "name" not in cluster_config:
            return None
        if key == "config":
            return cluster_config
        if key == "cluster_name":
            return cluster_config["name"].rsplit("/", 1)[-1]
        return cluster_config[key]
    else:
        return None


def _default_env_if_on_cluster():
    from runhouse import Env

    config = _current_cluster()
    return (
        _get_env_from(
            config.get(
                "default_env",
                Env(name=EMPTY_DEFAULT_ENV_NAME),
            )
        )
        if config
        else None
    )


def _get_cluster_from(system, dryrun=False):
    from .cluster import Cluster

    if isinstance(system, Cluster):
        return system
    if system in RESERVED_SYSTEM_NAMES:
        return system

    if isinstance(system, Dict):
        return Cluster.from_config(system, dryrun)

    if isinstance(system, str):
        config = _current_cluster(key="config")
        if config and system == config.get("name"):
            return Cluster.from_config(config, dryrun)
        try:
            system = Cluster.from_name(name=system, dryrun=dryrun)
        except ValueError:
            # Name not found in RNS. Doing the lookup this way saves us a hop to RNS
            pass

    return system


def _unnamed_default_env_name(cluster_name):
    return f"{cluster_name}_default_env"


def detect_cuda_version_or_cpu(cluster: "Cluster" = None):
    """Return the CUDA version on the cluster. If we are on a CPU-only cluster return 'cpu'.

    Note: A cpu-only machine may have the CUDA toolkit installed, which means nvcc will still return
    a valid version. Also check if the NVIDIA driver is installed to confirm we are on a GPU."""

    status_codes = run_setup_command("nvcc --version", cluster=cluster)
    if not status_codes[0] == 0:
        return "cpu"
    cuda_version = status_codes[1].split("release ")[1].split(",")[0]

    if run_setup_command("nvidia-smi", cluster=cluster)[0] == 0:
        return cuda_version
    return "cpu"
