import copy
import datetime
import hashlib
import json
import logging
import os
import re
import shlex
import subprocess
from asyncio import Event

from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from runhouse.constants import (
    CLUSTER_CONFIG_PATH,
    LAST_ACTIVE_AT_TIMEFRAME,
    RESERVED_SYSTEM_NAMES,
    SKY_VENV,
    TIME_UNITS,
)
from runhouse.globals import configs, rns_client

from runhouse.logger import get_logger
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


class RunhouseDaemonStatus(str, Enum):
    RUNNING = "running"
    TERMINATED = "terminated"
    UNAUTHORIZED = "unauthorized"
    UNKNOWN = "unknown"
    INTERNAL_SERVER_ERROR = "internal_server_error"
    RUNHOUSE_DAEMON_DOWN = "runhouse_daemon_down"
    INVALID_URL = "invalid_url"


class ClusterStatus(str, Enum):
    RUNNING = "running"
    TERMINATED = "terminated"
    INITIALIZING = "initializing"
    UNKNOWN = "unknown"


def cluster_config_file_exists() -> bool:
    return Path(CLUSTER_CONFIG_PATH).expanduser().exists()


def load_cluster_config_from_file() -> Dict:
    if cluster_config_file_exists():
        with open(Path(CLUSTER_CONFIG_PATH).expanduser()) as f:
            cluster_config = json.load(f)
        return cluster_config
    else:
        return {}


def _config_and_args_mismatches(config, alt_options):
    """Overload by child resources to compare their config with the alt_options. If the user specifies alternate
    options, compare the config with the options. It's generally up to the child class to decide how to handle the
    options, but default behavior is provided. The default behavior simply checks if any of the alt_options are
    present in the config (with awareness of resources), and if their values differ, return None.

    If the child class returns None, it's deciding to override the config
    with the options. If the child class returns a config, it's deciding to use the config and ignore the options
    (or somehow incorporate them, rarely). Note that if alt_options are provided and the config is not found,
    no error is raised, while if alt_options are not provided and the config is not found, an error is raised.
    """
    from runhouse.resources.images.image import Image
    from runhouse.resources.resource import Resource

    def alt_option_to_repr(val):
        if isinstance(val, dict):
            # This can either be a sub-resource which hasn't been converted to a resource yet, or an
            # actual user-provided dict
            if "rns_address" in val:
                return val["rns_address"]
            if "name" in val:
                # convert a user-provided name to an rns_address
                return rns_client.resolve_rns_path(val["name"])
            else:
                return val
        elif isinstance(val, list):
            val = [str(item) if isinstance(item, int) else item for item in val]
        elif isinstance(val, int) or isinstance(val, float):
            val = str(val)
        elif isinstance(val, Image):
            val = val.config()
        elif isinstance(val, Resource) and (
            val.config().get("name") or val.config().get("rns_address")
        ):
            return alt_option_to_repr(val.config())
        return val

    mismatches = {}
    for key, value in alt_options.items():
        if key in config:
            if alt_option_to_repr(value) != alt_option_to_repr(config[key]):
                mismatches[key] = value
        else:
            mismatches[key] = value
    return mismatches


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
            # Name not found in Den. Doing the lookup this way saves us a hop to Den
            pass

    return system


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


def is_gpu_cluster(cluster: "Cluster" = None, node: Optional[str] = None):
    if run_setup_command("nvidia-smi", cluster=cluster, node=node)[0] == 0:
        return True
    return False


def detect_cuda_version_or_cpu(cluster: "Cluster" = None, node: Optional[str] = None):
    """Return the CUDA version on the cluster. If we are on a CPU-only cluster return 'cpu'.

    Note: A cpu-only machine may have the CUDA toolkit installed, which means nvcc will still return
    a valid version. Also check if the NVIDIA driver is installed to confirm we are on a GPU."""

    if not is_gpu_cluster(cluster=cluster, node=node):
        return "cpu"

    status_codes = run_setup_command("nvcc --version", cluster=cluster, node=node)
    if not status_codes[0] == 0:
        status_codes = run_setup_command(
            "/usr/local/cuda/bin/nvcc --version", cluster=cluster, node=node
        )
        if not status_codes[0] == 0:
            raise RuntimeError(
                "Could not determine CUDA version on GPU cluster for installing the correct torch version. "
                "Please install nvcc on the cluster to enable automatic CUDA version detection, or include "
                "the exact version to install for the package, e.g. 'torch==1.13.1+cu117' or by including "
                "the --index-url or --extra-index-url"
            )
    cuda_version = status_codes[1].split("release ")[1].split(",")[0]

    return cuda_version


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


def _cluster_set_autostop_command(autostop_mins: int):
    sky_set_autostop_cmd = shlex.quote(
        f"from sky.skylet.autostop_lib import set_autostop; "
        f'set_autostop({autostop_mins}, "cloudvmray", True)'
    )
    return f"{SKY_VENV}/bin/python -c {sky_set_autostop_cmd}"


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


def parse_filters(since: str, cluster_status: Union[str, ClusterStatus]):
    cluster_filters = {}

    if since:
        last_active_in: int = parse_time_duration(
            duration=since
        )  # return in representing the "since" filter in seconds
        if last_active_in:
            cluster_filters["since"] = last_active_in

    if cluster_status:
        cluster_filters["cluster_status"] = cluster_status

    return cluster_filters


def get_clusters_from_den(cluster_filters: dict, force: bool):
    get_clusters_params = {"resource_type": "cluster", "folder": rns_client.username}

    if (
        "cluster_status" in cluster_filters
        and cluster_filters["cluster_status"] == ClusterStatus.TERMINATED
    ):
        # Include the relevant daemon status for the filter
        cluster_filters["daemon_status"] = RunhouseDaemonStatus.TERMINATED

    # If "all" filter is specified load all clusters (no filters are added to get_clusters_params)
    if cluster_filters and "all" not in cluster_filters.keys():
        get_clusters_params.update(cluster_filters)

    # If not filters are specified, get only running clusters.
    elif not cluster_filters:
        # For ondemand clusters, use cluster status (via Sky). for other clusters, use the status of the daemon
        get_clusters_params.update(
            {
                "cluster_status": ClusterStatus.RUNNING,
                "daemon_status": RunhouseDaemonStatus.RUNNING,
                "since": LAST_ACTIVE_AT_TIMEFRAME,
                "force": force,
            }
        )

    clusters_in_den_resp = rns_client.session.get(
        f"{rns_client.api_server_url}/resource",
        params=get_clusters_params,
        headers=rns_client.request_headers(),
    )

    return clusters_in_den_resp


def get_unsaved_live_clusters(den_clusters: List[Dict]):
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


def cast_last_active_timestamp(clusters: List[Dict[str, Any]]):
    for cluster in clusters:
        timestamp = cluster.get("Last Active (UTC)")
        timestamp = (
            timestamp.strftime("%m/%d/%Y, %H:%M:%S")
            if isinstance(timestamp, datetime.datetime)
            else timestamp
        )
        timestamp = timestamp if timestamp != "01/01/1970, 00:00:00" else None
        cluster["Last Active (UTC)"] = timestamp
    return clusters


def get_running_and_not_running_clusters(clusters: list):
    up_clusters, down_clusters = [], []
    for den_cluster in clusters:
        # Display the name instead of the full Den address
        cluster_name = den_cluster.get("name").split("/")[-1]
        cluster_type = den_cluster.get("data").get("resource_subtype")
        cluster_status = den_cluster.get("cluster_status")
        cluster_status = cluster_status or ClusterStatus.UNKNOWN.value

        # The split is required to remove milliseconds and the offset (according to UTC) from the timestamp.
        # (cluster_status_last_checked is in the following format: YYYY-MM-DD HH:MM:SS.ssssss±HH:MM)
        last_active_at = den_cluster.get("cluster_status_last_checked")

        # Convert to datetime
        last_active_at = (
            datetime.datetime.fromisoformat(last_active_at.split(".")[0])
            if isinstance(last_active_at, str)
            else datetime.datetime(1970, 1, 1)
        )

        last_active_at = last_active_at.replace(tzinfo=datetime.timezone.utc)
        if cluster_status == ClusterStatus.RUNNING and not last_active_at:
            # For BC, in case there are clusters that were saved and created before sending cluster status to Den
            cluster_status = ClusterStatus.UNKNOWN.value

        cluster_info = {
            "Name": cluster_name,
            "Cluster Type": cluster_type,
            "Status": cluster_status,
            "Autostop": den_cluster.get("data", {}).get("autostop_mins"),
            "Last Active (UTC)": last_active_at,
        }

        if cluster_status == ClusterStatus.RUNNING:
            up_clusters.append(cluster_info)
        else:
            down_clusters.append(cluster_info)

    # Sort clusters by the 'Last Active (UTC)' and 'Status' column
    down_clusters = sorted(
        down_clusters,
        key=lambda x: (x["Last Active (UTC)"], x["Status"]),
        reverse=True,
    )

    down_clusters = cast_last_active_timestamp(clusters=down_clusters)
    up_clusters = sorted(
        up_clusters, key=lambda x: x["Last Active (UTC)"], reverse=True
    )

    return cast_last_active_timestamp(clusters=up_clusters), cast_last_active_timestamp(
        clusters=down_clusters
    )


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


###################################
# SERVER-SIDE EVENTS
###################################
_FIELD_SEPARATOR = ":"


class SSEClient(object):
    """Implementation of a SSE client.

    See http://www.w3.org/TR/2009/WD-eventsource-20091029/ for the
    specification.
    """

    def __init__(self, event_source, char_enc="utf-8"):
        """Initialize the SSE client over an existing, ready to consume
        event source.

        The event source is expected to be a binary stream and have a close()
        method. That would usually be something that implements
        io.BinaryIOBase, like an httplib or urllib3 HTTPResponse object.
        """
        self._logger = logging.getLogger(self.__class__.__module__)
        self._logger.debug("Initialized SSE client from event source %s", event_source)
        self._event_source = event_source
        self._char_enc = char_enc

    def _read(self):
        """Read the incoming event source stream and yield event chunks.

        Unfortunately it is possible for some servers to decide to break an
        event into multiple HTTP chunks in the response. It is thus necessary
        to correctly stitch together consecutive response chunks and find the
        SSE delimiter (empty new line) to yield full, correct event chunks."""
        data = b""
        for chunk in self._event_source:
            for line in chunk.splitlines(True):
                data += line
                if data.endswith((b"\r\r", b"\n\n", b"\r\n\r\n")):
                    yield data
                    data = b""
        if data:
            yield data

    def events(self):
        for chunk in self._read():
            event = Event()
            # Split before decoding so splitlines() only uses \r and \n
            for line in chunk.splitlines():
                # Decode the line.
                line = line.decode(self._char_enc)

                # Lines starting with a separator are comments and are to be
                # ignored.
                if not line.strip() or line.startswith(_FIELD_SEPARATOR):
                    continue

                data = line.split(_FIELD_SEPARATOR, 1)
                field = data[0]

                # Ignore unknown fields.
                if field not in event.__dict__:
                    self._logger.debug(
                        "Saw invalid field %s while parsing " "Server Side Event", field
                    )
                    continue

                if len(data) > 1:
                    # From the spec:
                    # "If value starts with a single U+0020 SPACE character,
                    # remove it from value."
                    if data[1].startswith(" "):
                        value = data[1][1:]
                    else:
                        value = data[1]
                else:
                    # If no value is present after the separator,
                    # assume an empty value.
                    value = ""

                # The data field may come over multiple lines and their values
                # are concatenated with each other.
                if field == "data":
                    event.__dict__[field] += value + "\n"
                else:
                    event.__dict__[field] = value

            # Events with no data are not dispatched.
            if not event.data:
                continue

            # If the data field ends with a newline, remove it.
            if event.data.endswith("\n"):
                event.data = event.data[0:-1]

            # Empty event names default to 'message'
            event.event = event.event or "message"

            # Dispatch the event
            self._logger.debug("Dispatching %s...", event)
            yield event

    def close(self):
        """Manually close the event source stream."""
        self._event_source.close()


class Event(object):
    """Representation of an event from the event stream."""

    def __init__(self, id=None, event="message", data="", retry=None):
        self.id = id
        self.event = event
        self.data = data
        self.retry = retry

    def __str__(self):
        s = "{0} event".format(self.event)
        if self.id:
            s += " #{0}".format(self.id)
        if self.data:
            s += ", {0} byte{1}".format(len(self.data), "s" if len(self.data) else "")
        else:
            s += ", no data"
        if self.retry:
            s += ", retry in {0}ms".format(self.retry)
        return s


###################################
# KUBERNETES SETUP
###################################
def setup_kubernetes(
    kube_namespace: Optional[str] = None,
    kube_config_path: Optional[str] = None,
    kube_context: Optional[str] = None,
    **kwargs,
):
    if kwargs.get("provider") and not kwargs.get("provider") == "kubernetes":
        raise ValueError(
            f"Received non kubernetes provider {kwargs.get('provider')} with kubernetes specific "
            "cluster arguments."
        )

    if (
        kwargs.get("server_connection_type")
        and kwargs.get("server_connection_type") != ServerConnectionType.SSH
    ):
        raise ValueError(
            "Runhouse K8s Cluster server connection type must be set to `ssh`. "
            f"You passed {kwargs.get('server_connection_type')}."
        )

    if kube_context and kube_namespace:
        logger.warning(
            "You passed both a context and a namespace. Ensure your namespace matches the one in your context.",
        )

    launcher = kwargs.get("launcher") or configs.launcher
    if launcher == "local":
        if kube_context:
            # check if user passed a user-defined context
            try:
                cmd = f"kubectl config use-context {kube_context}"  # set user-defined context as current context
                subprocess.run(cmd, shell=True, check=True)
                logger.info(f"Kubernetes context has been set to: {kube_context}")
            except subprocess.CalledProcessError as e:
                logger.error(f"Error setting context {kube_context}: {e}")

        if kube_namespace:
            # Set the context only if launching locally
            # check if user passed a user-defined namespace
            cmd = f"kubectl config set-context --current --namespace={kube_namespace}"
            try:
                process = subprocess.run(
                    cmd,
                    shell=True,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                logger.debug(process.stdout)
                logger.info(f"Kubernetes namespace set to {kube_namespace}")

            except subprocess.CalledProcessError as e:
                logger.info(f"Error: {e}")

        if kube_config_path:  # check if user passed a user-defined kube_config_path
            kube_config_dir = os.path.expanduser("~/.kube")
            kube_config_path_rl = os.path.join(kube_config_dir, "config")

            if not os.path.exists(
                kube_config_dir
            ):  # check if ~/.kube directory exists on local machine
                try:
                    os.makedirs(
                        kube_config_dir
                    )  # create ~/.kube directory if it doesn't exist
                    logger.info(f"Created directory: {kube_config_dir}")
                except OSError as e:
                    logger.info(f"Error creating directory: {e}")

            if os.path.exists(kube_config_path_rl):
                raise Exception(
                    "A kubeconfig file already exists in ~/.kube directory. Aborting."
                )

            try:
                cmd = f"cp {kube_config_path} {kube_config_path_rl}"  # copy user-defined kube_config to ~/.kube/config
                subprocess.run(cmd, shell=True, check=True)
                logger.info(f"Copied kubeconfig to: {kube_config_path}")
            except subprocess.CalledProcessError as e:
                logger.info(f"Error copying kubeconfig: {e}")
