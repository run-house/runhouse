import hashlib
import json
import subprocess

from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union

from runhouse.constants import (
    CLUSTER_CONFIG_PATH,
    EMPTY_DEFAULT_ENV_NAME,
    RESERVED_SYSTEM_NAMES,
)

from runhouse.logger import get_logger
from runhouse.resources.envs.utils import _get_env_from, run_setup_command
from runhouse.resources.hardware.sky.command_runner import (
    _HASH_MAX_LENGTH,
    ssh_options_list,
    SshMode,
)

logger = get_logger(__name__)


class ServerConnectionType(str, Enum):
    """Manage the type of connection Runhouse will make with the API server started on the cluster.
    ``ssh``: Use port forwarding to connect to the server via SSH, by default on port 32300.
    ``tls``: Do not use port forwarding and start the server with HTTPS (using custom or fresh TLS certs), by default
        on port 443.
    ``none``: Do not use port forwarding, and start the server with HTTP, by default on port 80.
    """

    SSH = "ssh"
    TLS = "tls"
    NONE = "none"


class ResourceServerStatus(str, Enum):
    running = "running"
    terminated = "terminated"
    unauthorized = "unauthorized"
    unknown = "unknown"
    internal_server_error = "internal_server_error"
    runhouse_daemon_down = "runhouse_daemon_down"
    invalid_url = "invalid_url"
    local_cluster = "local_cluster"


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


def detect_cuda_version_or_cpu(cluster: "Cluster" = None, node: Optional[str] = None):
    """Return the CUDA version on the cluster. If we are on a CPU-only cluster return 'cpu'.

    Note: A cpu-only machine may have the CUDA toolkit installed, which means nvcc will still return
    a valid version. Also check if the NVIDIA driver is installed to confirm we are on a GPU."""

    status_codes = run_setup_command("nvcc --version", cluster=cluster, node=node)
    if not status_codes[0] == 0:
        return "cpu"
    cuda_version = status_codes[1].split("release ")[1].split(",")[0]

    if run_setup_command("nvidia-smi", cluster=cluster, node=node)[0] == 0:
        return cuda_version
    return "cpu"


def _run_ssh_command(
    address: str,
    ssh_user: str,
    ssh_port: int,
    ssh_private_key: str,
    ssh_proxy_command: str,
    docker_user: str,
):
    from runhouse.resources.hardware.sky_command_runner import SkySSHRunner

    runner = SkySSHRunner(
        (address, ssh_port),
        ssh_user=ssh_user,
        ssh_private_key=ssh_private_key,
        ssh_proxy_command=ssh_proxy_command,
        docker_user=docker_user,
    )
    ssh_command = runner._ssh_base_command(
        ssh_mode=SshMode.INTERACTIVE, port_forward=None
    )
    subprocess.run(ssh_command)


def _docker_ssh_proxy_command(
    address: str,
    ssh_user: str,
    ssh_private_key: str,
):
    return lambda ssh: " ".join(
        ssh
        + ssh_options_list(ssh_private_key, None)
        + ["-W", "%h:%p", f"{ssh_user}@{address}"]
    )


# Adapted from SkyPilot Command Runner
def _ssh_base_command(
    address: str,
    ssh_user: str,
    ssh_private_key: str,
    ssh_control_name: Optional[str] = "__default__",
    ssh_proxy_command: Optional[str] = None,
    ssh_port: int = 22,
    docker_ssh_proxy_command: Optional[str] = None,
    disable_control_master: Optional[bool] = False,
    ssh_mode: SshMode = SshMode.INTERACTIVE,
    port_forward: Optional[List[int]] = None,
    connect_timeout: Optional[int] = 30,
):
    ssh = ["ssh"]
    if ssh_mode == SshMode.NON_INTERACTIVE:
        # Disable pseudo-terminal allocation. Otherwise, the output of
        # ssh will be corrupted by the user's input.
        ssh += ["-T"]
    else:
        # Force pseudo-terminal allocation for interactive/login mode.
        ssh += ["-tt"]

    if port_forward is not None:
        # RH MODIFIED: Accept port int (to forward same port) or pair of ports
        for fwd in port_forward:
            if isinstance(fwd, int):
                local, remote = fwd, fwd
            else:
                local, remote = fwd
            logger.debug(f"Forwarding port {local} to port {remote} on localhost.")
            ssh += ["-L", f"{local}:localhost:{remote}"]

    return (
        ssh
        + ssh_options_list(
            ssh_private_key,
            ssh_control_name,
            ssh_proxy_command=ssh_proxy_command,
            docker_ssh_proxy_command=docker_ssh_proxy_command,
            # TODO change to None like before?
            port=ssh_port,
            connect_timeout=connect_timeout,
            disable_control_master=disable_control_master,
        )
        + [f"{ssh_user}@{address}"]
    )


def _generate_ssh_control_hash(ssh_control_name):
    return hashlib.md5(ssh_control_name.encode()).hexdigest()[:_HASH_MAX_LENGTH]


def up_cluster_helper(cluster, capture_output: Union[bool, str] = True):
    from runhouse.utils import SuppressStd

    if capture_output:
        try:
            with SuppressStd() as outfile:
                cluster.up()
        except Exception as e:
            if isinstance(capture_output, str):
                logger.error(
                    f"Error starting cluster {cluster.name}, logs written to {capture_output}"
                )
            raise e
        finally:
            if isinstance(capture_output, str):
                with open(capture_output, "w") as f:
                    f.write(outfile.output)
    else:
        cluster.up()
