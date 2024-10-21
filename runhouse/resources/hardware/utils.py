import copy
import datetime
import hashlib
import json
import os
import re
import subprocess
from asyncio import Event

from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from runhouse.constants import (
    CLUSTER_CONFIG_PATH,
    EMPTY_DEFAULT_ENV_NAME,
    LAST_ACTIVE_AT_TIMEFRAME,
    RESERVED_SYSTEM_NAMES,
    TIME_UNITS,
)
from runhouse.globals import rns_client

from runhouse.logger import get_logger
from runhouse.resources.envs.utils import _get_env_from
from runhouse.resources.hardware.sky.command_runner import (
    _HASH_MAX_LENGTH,
    ssh_options_list,
    SshMode,
)
from runhouse.utils import run_setup_command

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


class LauncherType(str, Enum):
    LOCAL = "local"
    DEN = "den"


class ResourceServerStatus(str, Enum):
    running = "running"
    terminated = "terminated"
    unauthorized = "unauthorized"
    unknown = "unknown"
    internal_server_error = "internal_server_error"
    runhouse_daemon_down = "runhouse_daemon_down"
    invalid_url = "invalid_url"


class ClustersListStatus(str, Enum):
    running = "running"
    terminated = "terminated"
    down = "down"


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
        # This could be a local cluster started via runhouse server start,
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


def _setup_default_creds(cluster_type: str):
    from runhouse.resources.secrets import Secret

    default_ssh_key = rns_client.default_ssh_key
    if cluster_type == "OnDemandCluster":
        try:
            sky_secret = Secret.from_name("sky")
            return sky_secret
        except ValueError:
            if default_ssh_key:
                # copy over default key to sky-key for launching use
                default_secret = Secret.from_name(default_ssh_key)
                sky_secret = default_secret._write_to_file("~/.ssh/sky-key")
                return sky_secret
            else:
                return None
    elif default_ssh_key:
        return Secret.from_name(default_ssh_key)
    return None


def _setup_creds_from_dict(ssh_creds: Dict, cluster_name: str):
    from runhouse.resources.secrets import Secret
    from runhouse.resources.secrets.provider_secrets.sky_secret import SkySecret
    from runhouse.resources.secrets.provider_secrets.ssh_secret import SSHSecret

    creds = copy.copy(ssh_creds)
    ssh_properties = {}
    cluster_secret = None

    private_key_path = creds["ssh_private_key"] if "ssh_private_key" in creds else None
    password = creds.pop("password") if "password" in creds else None

    if private_key_path:
        key = os.path.basename(private_key_path)

        if password:
            # extract ssh values and create ssh secret
            values = (
                SSHSecret.extract_secrets_from_path(private_key_path)
                if private_key_path
                else {}
            )
            values["password"] = password
            cluster_secret = SSHSecret(
                name=f"{cluster_name}-ssh-secret",
                provider="ssh",
                key=key,
                path=private_key_path,
                values=values,
            )
        else:
            # set as standard SSH secret
            constructor = SkySecret if key == "sky-key" else SSHSecret
            cluster_secret = constructor(
                name=f"ssh-{key}", key=key, path=private_key_path
            )
    elif password:
        cluster_secret = Secret(
            name=f"{cluster_name}-ssh-secret", values={"password": password}
        )

    # keep track of non secret/password values in ssh_properties
    if isinstance(creds, Dict):
        ssh_properties = creds

    return cluster_secret, ssh_properties


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


###################################
# Cluster list helping methods
###################################


def parse_time_duration(duration: str):
    # A simple parser for duration like "15m", "2h", "3d"
    try:
        unit = duration[-1]
        value = int(duration[:-1])

        time_filter_match = re.match(r"(\d+)([smhd])$", duration)

        if time_filter_match:

            # if the user provides a "--since" time filter than is less than a minute, set it by default to one minute.
            if unit == "s" and value < 60:
                value = 60

            time_duration = value * TIME_UNITS.get(unit)

        else:
            logger.warning(
                f"Filter is not applied, invalid time unit provided ({unit}). Setting filter to default (24h)."
            )
            time_duration = LAST_ACTIVE_AT_TIMEFRAME

        return time_duration
    except Exception:
        return LAST_ACTIVE_AT_TIMEFRAME


def parse_filters(since: str, cluster_status: str):
    cluster_filters = {}

    if since:
        last_active_in: int = parse_time_duration(
            duration=since
        )  # return in representing the "since" filter in seconds
        if last_active_in:
            cluster_filters["since"] = last_active_in

    if cluster_status:

        if cluster_status.lower() == ClustersListStatus.down:
            cluster_status = ResourceServerStatus.runhouse_daemon_down

        cluster_filters["cluster_status"] = cluster_status

    return cluster_filters


def get_clusters_from_den(cluster_filters: dict):
    get_clusters_params = {"resource_type": "cluster", "folder": rns_client.username}

    # send den request with filters if the user specifies filters.
    # If "all" filter is specified - get all clusters (no filters are added to get_clusters_params)
    if cluster_filters and "all" not in cluster_filters.keys():
        get_clusters_params.update(cluster_filters)

    # If not filters are specified, get only running clusters.
    elif not cluster_filters:
        get_clusters_params.update(
            {"cluster_status": "running", "since": LAST_ACTIVE_AT_TIMEFRAME}
        )

    clusters_in_den_resp = rns_client.session.get(
        f"{rns_client.api_server_url}/resource",
        params=get_clusters_params,
        headers=rns_client.request_headers(),
    )

    return clusters_in_den_resp


def get_unsaved_live_clusters(den_clusters: list[Dict]):
    import sky

    try:
        sky_live_clusters: list = sky.status()
        den_clusters_names = [c.get("name") for c in den_clusters]

        # getting the on-demand clusters that are not saved in den.
        if sky_live_clusters:
            return [
                cluster
                for cluster in sky_live_clusters
                if f'/{rns_client.username}/{cluster.get("name")}'
                not in den_clusters_names
            ]
        else:
            return []
    except Exception as e:
        logger.debug(f"Failed to get unsaved sky live clusters: {e}")
        return []


def get_all_sky_clusters():
    import sky

    try:
        sky_live_clusters: list = sky.status()

        # getting the on-demand clusters that are not saved in den.
        if sky_live_clusters:
            return [cluster.get("name") for cluster in sky_live_clusters]
        else:
            return []
    except Exception as e:
        logger.debug(f"Failed to get sky live clusters: {e}")
        return []


def cluster_last_active_from_datetime_to_str(clusters: List[Dict[str, Any]]):
    for cluster in clusters:
        cluster["Last Active (UTC)"] = cluster.get("Last Active (UTC)").strftime(
            "%m/%d/%Y, %H:%M:%S"
        )
    return clusters


def get_running_and_not_running_clusters(clusters: list):
    running_clusters, not_running_clusters = [], []

    for den_cluster in clusters:
        # get just name, not full rns address. reset is used so the name will be printed all in white.
        cluster_name = den_cluster.get("name").split("/")[-1]
        cluster_type = den_cluster.get("data").get("resource_subtype")
        cluster_status = (
            den_cluster.get("status") if den_cluster.get("status") else "unknown"
        )

        # currently relying on status pings to den as a sign of cluster activity.
        # The split is required to remove milliseconds and the offset (according to UTC) from the timestamp.
        # (status_last_checked is in the following format: YYYY-MM-DD HH:MM:SS.ssssss±HH:MM)

        last_active_at = den_cluster.get("status_last_checked")
        last_active_at = (
            datetime.datetime.fromisoformat(last_active_at.split(".")[0])
            if isinstance(last_active_at, str)
            else datetime.datetime(1970, 1, 1)
        )  # Convert to datetime
        last_active_at = last_active_at.replace(tzinfo=datetime.timezone.utc)

        if cluster_status == "running" and not last_active_at:
            # For BC, in case there are clusters that were saved and created before we introduced sending cluster status to den.
            cluster_status = "unknown"

        cluster_info = {
            "Name": cluster_name,
            "Cluster Type": cluster_type,
            "Status": cluster_status,
            "Last Active (UTC)": last_active_at,
        }
        running_clusters.append(
            cluster_info
        ) if cluster_status == "running" else not_running_clusters.append(cluster_info)

    # Sort clusters by the 'Last Active (UTC)' and 'Status' column
    not_running_clusters = sorted(
        not_running_clusters,
        key=lambda x: (x["Last Active (UTC)"], x["Status"]),
        reverse=True,
    )

    not_running_clusters = cluster_last_active_from_datetime_to_str(
        clusters=not_running_clusters
    )

    running_clusters = sorted(
        running_clusters, key=lambda x: x["Last Active (UTC)"], reverse=True
    )

    running_clusters = cluster_last_active_from_datetime_to_str(
        clusters=running_clusters
    )

    return running_clusters, not_running_clusters


###################################
# CLUSTER LOGS
###################################
def get_saved_logs_from_den(rns_address: str):
    """
    get the latest cluster logs saved in den.
    """
    cluster_uri = rns_client.resource_uri(rns_address)
    clusters_in_den_resp = rns_client.session.get(
        f"{rns_client.api_server_url}/resource/{cluster_uri}/logs",
        headers=rns_client.request_headers(),
    )
    return clusters_in_den_resp


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


def load_logs_in_thread(stop_event: Event, url: str, temp_dir: str, launch_id: str):
    import asyncio

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(stream_logs_from_url(stop_event, url, temp_dir, launch_id))
