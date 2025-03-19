import importlib
import math
import subprocess
import time

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List

import requests

import rich
import typer
from rich.table import Table

import runhouse as rh

from runhouse.constants import (
    BULLET_UNICODE,
    CALLABLE_RESOURCE_TYPES,
    DEFAULT_PROCESS_NAME,
    DOUBLE_SPACE_UNICODE,
    HOUR,
    LAST_ACTIVE_AT_TIMEFRAME,
    MAX_CLUSTERS_DISPLAY,
    SERVER_START_CMD,
    START_NOHUP_CMD,
    START_SCREEN_CMD,
)

from runhouse.logger import get_logger

from runhouse.resources.hardware.utils import ClusterStatus, LauncherType
from runhouse.servers.obj_store import ObjStoreError

logger = get_logger(__name__)


####################################################################################################
# Cluster list utils
####################################################################################################
class ClusterStatusColors(str, Enum):
    RUNNING = "[green]Running[/green]"
    INITIALIZING = "[yellow]Initializing[/yellow]"
    TERMINATED = "[red]Terminated[/red]"
    UNKNOWN = "Unknown"

    @classmethod
    def get_status_color(cls, status: str):
        try:
            return getattr(cls, status.upper()).value
        except AttributeError:
            return cls.UNKNOWN.value


def create_output_table(
    total_clusters: int,
    running_clusters: int,
    displayed_clusters: int,
    filters_requested: bool,
):
    """The cluster list is printed as a table, this method creates it."""
    from runhouse.globals import rns_client

    displayed_running_clusters = (
        running_clusters
        if running_clusters < displayed_clusters
        else displayed_clusters
    )
    table_title = (
        f"[bold cyan]Clusters for {rns_client.username} "
        f"(Running: {displayed_running_clusters}/{running_clusters}, "
        f"Total Displayed: {displayed_clusters}/{total_clusters})[/bold cyan]"
    )

    table = Table(title=table_title)

    if not filters_requested:
        table.caption = (
            f"[reset]Showing clusters that were active in the "
            f"last {int(LAST_ACTIVE_AT_TIMEFRAME / HOUR)} hours."
        )
        table.caption_justify = "left"

    if displayed_clusters == MAX_CLUSTERS_DISPLAY:
        link_to_clusters_in_den = "[reset]The full list of clusters can be viewed at https://www.run.house/resources?type=cluster"
        if table.caption:
            table.caption += f"\n{link_to_clusters_in_den}"
        else:
            table.caption = link_to_clusters_in_den
        table.caption_justify = "left"

    # Add columns to the table
    table.add_column("Name", justify="left", no_wrap=True)
    table.add_column("Cluster Type", justify="left", no_wrap=True)
    table.add_column("Status", justify="left")
    table.add_column("Last Active (UTC)", justify="left")
    table.add_column("Autostop (Mins)", justify="left")

    return table


def add_cluster_as_table_row(table: Table, rh_cluster: dict):
    """Adding an info of a single cluster to the output table."""
    table.add_row(
        rh_cluster.get("Name"),
        rh_cluster.get("Cluster Type"),
        rh_cluster.get("Status"),
        rh_cluster.get("Last Active (UTC)"),
        rh_cluster.get("Autostop (Mins)"),
    )

    return table


def add_clusters_to_output_table(table: Table, clusters: List[Dict]):
    """Adding clusters info to the output table."""
    for rh_cluster in clusters:
        last_active_at = rh_cluster.get("Last Active (UTC)")
        if not last_active_at:  # case when last_active_at == None
            last_active_at_no_offset = last_active_at
        else:
            last_active_at_no_offset = str(last_active_at).split("+")[
                0
            ]  # The split is required to remove the offset (according to UTC)
        rh_cluster["Last Active (UTC)"] = last_active_at_no_offset
        rh_cluster["Status"] = ClusterStatusColors.get_status_color(
            rh_cluster.get("Status")
        )
        table = add_cluster_as_table_row(table, rh_cluster)


def condense_resource_type(resource_type: str):
    """
    status helping function. transforms a str form runhouse.resources.{X.Y...}.resource_type to runhouse.resource_type
    """
    try:
        resource_type = resource_type.split(".")[-1]
        getattr(importlib.import_module("runhouse"), resource_type)
        return f"runhouse.{resource_type}"
    except AttributeError:
        return resource_type


####################################################################################################
# Cluster status utils
####################################################################################################
class StatusType(str, Enum):
    server = "server"
    cluster = "cluster"


# The user will be able to pass the node argument, which represents the specific cluster node they would like to get
# the status of. The value of the node argument could be either the node IP or its index in the IPs list.
class NodeFilterType(str, Enum):
    ip = "ip"
    node_index = "node_index"


def print_cluster_config(cluster_config: Dict, status_type: str = StatusType.cluster):
    """
    Helping function to the `_print_status` which prints the relevant info from the cluster config.
    """
    from runhouse.main import console

    top_level_config = [
        "server_port",
        "server_connection_type",
    ]

    backend_config = ["server_host"]
    if status_type == StatusType.cluster:
        backend_config = backend_config + ["resource_subtype", "ips"]

    for key in top_level_config:
        console.print(
            f"[reset]{BULLET_UNICODE} {key.replace('_', ' ')}: {cluster_config[key]}"
        )

    if status_type == StatusType.cluster:
        console.print(f"{BULLET_UNICODE} backend config:")

    for key in backend_config:
        if key == "ips" and cluster_config.get("resource_subtype") == "OnDemandCluster":
            val = cluster_config.get("compute_properties", {}).get("ips", [])
        else:
            val = cluster_config.get(key, None)

        # don't print keys whose values are None
        if val is None:
            continue

        console.print(
            f"[reset]{DOUBLE_SPACE_UNICODE}{BULLET_UNICODE} {key.replace('_', ' ')}: {val}"
        ) if status_type == StatusType.cluster else console.print(
            f"[reset]{BULLET_UNICODE} {key.replace('_', ' ')}: {val}"
        )


def print_processes_info(servlet_processes: Dict[str, Dict[str, Any]], node_index: int):
    """
    Prints info about the processes in the current_cluster: resources in each process, the CPU usage and GPU usage of
    the process (if exists)
    """
    from runhouse.main import console

    # Print headline
    processes_in_cluster_headline = "Serving 🍦 :"
    console.print(processes_in_cluster_headline, style="bold turquoise4")

    process_resource_mapping = {
        process: servlet_processes[process].get("process_resource_mapping", {})
        for process in servlet_processes
    }

    if len(process_resource_mapping) == 0:
        console.print("This cluster has no processes nor resources.")

    first_processes_to_print = []

    # First: if the default process does not have resources, print it.
    default_process_name = DEFAULT_PROCESS_NAME
    if len(process_resource_mapping[default_process_name]) == 0:
        # case where the default process doesn't hve any other resources, apart from the default process itself.
        console.print(
            f"{BULLET_UNICODE} {default_process_name}",
            style="bold dark_cyan turquoise4",
        )
        console.print(
            f"{DOUBLE_SPACE_UNICODE}This process has only python packages installed, if provided. No "
            "resources were found."
        )

    else:
        # if the default process has other resources make sure it gets printed first
        first_processes_to_print = [default_process_name]

    # Make sure to print process with no resources first.
    first_processes_to_print = first_processes_to_print + [
        process_name
        for process_name in process_resource_mapping
        if (
            len(process_resource_mapping[process_name]) == 0
            and process_name != default_process_name
            and process_resource_mapping[process_name]
        )
    ]

    # Now, print the processes.
    # If the process has no resource associated, we'll print that it contain only installed packages (if such exist).
    # Otherwise, we will print the resources (rh.function, th.module) associated with the process.
    processes_to_print = first_processes_to_print + [
        process_name
        for process_name in process_resource_mapping
        if process_name not in first_processes_to_print + [default_process_name]
    ]

    for process_name in processes_to_print:
        resources_in_process = process_resource_mapping[process_name]
        process_info = servlet_processes[process_name]

        pid, node_name, process_node_index = (
            process_info.get("pid", None),
            process_info.get("node_name", None),
            process_info.get("node_index", None),
        )

        # if there is no info about the process, don't print it in the status output
        if not pid or not node_name or node_index != process_node_index:
            continue

        process_name_txt = f"[bold turquoise4]{BULLET_UNICODE} {process_name}"
        console.print(process_name_txt)

        # Print CPU info
        process_cpu_info = process_info.get("process_cpu_usage")
        if process_cpu_info:

            # convert bytes to GB
            memory_usage_gb = round(
                int(process_cpu_info["used_memory"]) / (1024**3),
                3,
            )
            total_cluster_memory = math.ceil(
                int(process_cpu_info["total_memory"]) / (1024**3)
            )
            cpu_memory_usage_percent = round(
                float(
                    process_cpu_info["used_memory"] / process_cpu_info["total_memory"]
                )
                * 100,
                3,
            )
            cpu_usage_percent = round(float(process_cpu_info["utilization_percent"]), 3)

            cpu_usage_summary = f"[reset]{DOUBLE_SPACE_UNICODE}CPU: {cpu_usage_percent}% | Memory: {memory_usage_gb} / {total_cluster_memory} Gb ({cpu_memory_usage_percent}%)"

        else:
            cpu_usage_summary = (
                f"{DOUBLE_SPACE_UNICODE}CPU: This process did not use CPU memory."
            )

        console.print(cpu_usage_summary)

        # Print GPU info
        process_gpu_info = process_info.get("process_gpu_usage")

        # sometimes the cluster has no GPU, therefore the process_gpu_info is an empty dictionary.
        if process_gpu_info:
            # get the gpu usage info, and convert it to GB.
            total_gpu_memory = math.ceil(
                float(process_gpu_info.get("total_memory")) / (1024**3)
            )
            used_gpu_memory = round(
                float(process_gpu_info.get("used_memory")) / (1024**3), 3
            )
            gpu_memory_usage_percent = round(
                float(used_gpu_memory / total_gpu_memory) * 100, 3
            )
            gpu_usage_summery = f"[reset]{DOUBLE_SPACE_UNICODE}GPU Memory: {used_gpu_memory} / {total_gpu_memory} Gb ({gpu_memory_usage_percent}%)"
            console.print(gpu_usage_summery)

        resources_in_process = [
            {resource: resources_in_process[resource]}
            for resource in resources_in_process
            if resource is not process_name
        ]

        if len(resources_in_process) == 0:
            # No resources were found in the process, only the associated installed python reqs were installed.
            console.print(
                f"{DOUBLE_SPACE_UNICODE}No objects are stored in this process."
            )

        else:
            for resource in resources_in_process:
                for resource_name, resource_info in resource.items():
                    resource_type = condense_resource_type(
                        resource_info.get("resource_type")
                    )

                    if resource_type == "runhouse.Env":
                        resource_type = None

                    active_function_calls = resource_info.get("active_function_calls")
                    resource_info_str = (
                        f"{DOUBLE_SPACE_UNICODE}{BULLET_UNICODE} {resource_name}"
                    )
                    if resource_type:
                        resource_info_str = resource_info_str + f" ({resource_type})"

                    if (
                        resource_type in CALLABLE_RESOURCE_TYPES
                        and active_function_calls
                    ):
                        func_start_time_utc = active_function_calls[0].get(
                            "start_time", None
                        )

                        # casting func_start_time_utc to datetime format
                        func_start_time_utc = datetime.fromtimestamp(
                            func_start_time_utc, tz=timezone.utc
                        )

                        # func_end_time_utc = current time. Making sure it is in the same format as func_start_time_utc,
                        # so we could calculate function's running time.
                        func_end_time_utc = datetime.fromtimestamp(
                            time.time(), tz=timezone.utc
                        )

                        func_running_time = (
                            func_end_time_utc - func_start_time_utc
                        ).total_seconds()

                        is_func_running: str = (
                            f" [italic dark_sea_green]Running for {func_running_time} "
                            f"seconds[/italic dark_sea_green]"
                        )

                    elif (
                        resource_type in CALLABLE_RESOURCE_TYPES
                        and not active_function_calls
                    ):
                        is_func_running: str = " [italic light_goldenrod3]Currently not running[/italic light_goldenrod3]"

                    else:
                        is_func_running: str = ""

                    resource_info_str = resource_info_str + is_func_running

                    console.print(resource_info_str)


def print_cloud_properties(cluster_config: dict):
    from runhouse.main import console

    cloud_properties = cluster_config.get("compute_properties", None)
    if not cloud_properties:
        return

    cloud = cloud_properties.get("cloud")
    instance_type = cloud_properties.get("instance_type")
    region = cloud_properties.get("region")
    cost_per_hour = cloud_properties.get("cost_per_hour")

    is_gpu = cluster_config.get("is_gpu", False)
    cost_emoji = "💰" if is_gpu else "💸"

    num_of_cpus = cloud_properties.get("num_cpus") or len(cluster_config.get("ips"))
    num_of_gpus = 0
    cluster_gpus = cloud_properties.get("gpus", None)
    gpu_types = set()
    if cluster_gpus:
        for k, v in cluster_gpus.items():
            num_of_gpus = num_of_gpus + int(v)
            gpu_types.add(k)

    console.print(
        f"[reset]🤖 {cloud} {instance_type} cluster | 🌍 {region} | {cost_emoji} ${cost_per_hour}/hr"
    )
    cpus_gpus_info_str = f"CPUs: {num_of_cpus}"
    if num_of_gpus > 0:
        gpu_types_str = gpu_types.pop()
        for gpu_type in gpu_types:
            gpu_types_str = gpu_types_str + f", {gpu_type}"
        cpus_gpus_info_str = (
            cpus_gpus_info_str + f" | GPUs: {num_of_gpus} (Type(s): {gpu_types_str})"
        )
    console.print(f"[reset]{cpus_gpus_info_str}")


# returns the worker cpu+gpu util info + it's index
def get_node_status_data(
    status_data: dict, ip_or_index: NodeFilterType = None, node: str = None
):
    workers = status_data.get("workers")

    # if node is not provided, we print the data of the head node
    if not node:
        return workers[0], 0

    if ip_or_index == NodeFilterType.ip:
        return next(
            (
                (worker, worker_index)
                for worker_index, worker in enumerate(workers)
                if worker.get("ip") == node
            ),
            ({}, None),
        )
    if ip_or_index == NodeFilterType.node_index:
        worker_index = int(node)
        return (
            (workers[worker_index], worker_index)
            if worker_index < len(workers)
            else ({}, None)
        )


def print_node_status(
    node_status_data: dict,
    worker_index: int,
    servlet_processes: dict,
    is_gpu: bool,
    current_cluster,
):
    from runhouse.main import console

    # print general cpu and gpu utilization
    cluster_gpu_usage: dict = node_status_data.get("server_gpu_usage", None)
    # Note: GPU utilization can be none, even if the cluster has GPU, if the cluster was not using its GPU when cluster.status() was invoked
    cluster_gpu_utilization_percent: float = (
        cluster_gpu_usage.get("utilization_percent") if cluster_gpu_usage else 0
    )

    cluster_cpu_usage: dict = node_status_data.get("server_cpu_usage")
    cluster_cpu_utilization_percent = cluster_cpu_usage.get("utilization_percent")
    node_name = f"worker {worker_index}" if worker_index > 0 else "head node"
    node_ip = node_status_data.get("ip")

    node_util_info = (
        f"[reset][bold light_sea_green]{node_name} | IP: {node_ip} | CPU Utilization: {round(cluster_cpu_utilization_percent, 3)}% | GPU Utilization: {round(cluster_gpu_utilization_percent, 3)}%"
        if is_gpu
        else f"[reset][bold light_sea_green]{node_name} | IP: {node_ip} | CPU Utilization: {round(cluster_cpu_utilization_percent, 3)}%"
    )
    console.print(node_util_info)

    # print the processes in the cluster, and the resources associated with each process.
    print_processes_info(servlet_processes, worker_index)


def print_status(status_data: dict, current_cluster, node: str = None) -> None:
    """Prints the status of the cluster to the console"""
    from runhouse.globals import rns_client
    from runhouse.main import console

    cluster_config = status_data.get("cluster_config")
    servlet_processes = status_data.get("processes")

    node_status_data, worker_index = None, None
    if node:
        node = "0" if node.lower() == "head" else node
        ip_or_index = (
            NodeFilterType.node_index if node.isnumeric() else NodeFilterType.ip
        )
        node_status_data, worker_index = get_node_status_data(
            status_data=status_data, ip_or_index=ip_or_index, node=node
        )
        if not node_status_data:
            console.print(
                f"[reset][bold italic red]Invalid node provided: {node}. Please provide correct node IP or node index."
            )
            return

    is_gpu = cluster_config.get("is_gpu", False)
    cluster_name = cluster_config.get("name", None)
    if cluster_name:
        cluster_uri = rns_client.format_rns_address(cluster_name)
        cluster_link_in_den_ui = f"https://www.run.house/resources/{cluster_uri}"
        cluster_name_hyperlink = rich.markdown.Text(
            cluster_name, style=f"link {cluster_link_in_den_ui} white"
        )
        console.print(cluster_name_hyperlink)

    # print headline
    daemon_headline_txt = (
        "\N{smiling face with horns} Runhouse server is running \N{Runner}"
    )
    console.print(daemon_headline_txt, style="bold royal_blue1")

    console.print(f"[reset]Runhouse v{status_data.get('runhouse_version')}")
    print_cloud_properties(cluster_config)
    console.print(f"[reset]server pid: {status_data.get('server_pid')}")

    # Print relevant info from cluster config.
    print_cluster_config(cluster_config)

    # Print processes information. If a node is provided, we'll print only the processes running on the specified node.
    # If no node is provided, we'll print information about all cluster nodes.
    if node:
        print_node_status(
            node_status_data, worker_index, servlet_processes, is_gpu, current_cluster
        )
    else:
        nodes_status_data = status_data.get("workers")
        for worker_index in range(len(nodes_status_data)):
            node_status_data = nodes_status_data[worker_index]
            print_node_status(
                node_status_data,
                worker_index,
                servlet_processes,
                is_gpu,
                current_cluster,
            )


def get_local_or_remote_cluster(cluster_name: str = None, exit_on_error: bool = True):
    from runhouse.main import console

    if cluster_name:
        try:
            current_cluster = rh.cluster(name=cluster_name, dryrun=True)
        except ValueError as e:
            console.print("Cluster not found in Den.")
            if exit_on_error:
                raise typer.Exit(1)
            raise e

        if isinstance(current_cluster, rh.OnDemandCluster):

            # in case we called current_cluster.up() on a local cluster, we need to update the cluster_status,
            # because its being updated properly only if we call up_if_not()
            if (
                current_cluster.launcher == LauncherType.LOCAL
                and current_cluster.cluster_status != ClusterStatus.RUNNING
            ):
                current_cluster._fetch_sky_status_and_update_cluster_status(
                    refresh=True
                )

            if current_cluster.cluster_status == ClusterStatus.INITIALIZING:
                console.print(
                    f"[reset]{cluster_name} is being initialized. Please wait for it to finish, or run [reset][bold italic]`runhouse cluster up {cluster_name} -f`[/bold italic] to abort the initialization and relaunch."
                )
                raise typer.Exit(0)

        if not current_cluster.is_up():
            console.print(
                f"Cluster [reset]{cluster_name} is not up. If it's an on-demand cluster, you can run "
                f"[reset][bold italic]`runhouse cluster up {cluster_name}`[/bold italic] to bring it up automatically."
            )
            if exit_on_error:
                raise typer.Exit(1)
        try:
            if current_cluster._http_client:
                current_cluster._http_client.check_server()
        except requests.exceptions.ConnectionError:
            console.print(
                f"Could not connect to the server on cluster [reset]{cluster_name}. Check that the server is up with "
                f"[reset][bold italic]`runhouse cluster status {cluster_name}`[/bold italic] or"
                f" [bold italic]`sky status -r`[/bold italic] for locally launched on-demand clusters."
            )
            if exit_on_error:
                raise typer.Exit(1)
        return current_cluster

    try:
        cluster_or_local = rh.here
    except ObjStoreError:
        console.print("Could not connect to Runhouse server. Is it up?")
        if exit_on_error:
            raise typer.Exit(1)

    if cluster_or_local == "file":
        # If running outside the cluster must specify a cluster name
        console.print(
            "Please specify a `cluster_name` or run [reset][bold italic]`runhouse server start`[/bold italic] to start "
            "a Runhouse server locally."
        )
        if exit_on_error:
            raise typer.Exit(1)
    elif not cluster_or_local:
        console.print(
            "\N{smiling face with horns} Runhouse Daemon is not running... \N{No Entry} \N{Runner}. "
            "Start it with [reset][bold italic]`runhouse server restart`[/bold italic] or specify a remote "
            "cluster to poll with [reset][bold italic]`runhouse cluster status <cluster_name>`[/bold italic]."
        )
        if exit_on_error:
            raise typer.Exit(1)

    else:
        # we are inside the cluster
        current_cluster = cluster_or_local  # cluster_or_local = rh.here

    return current_cluster


####################################################################################################
# General utils
####################################################################################################
class LogsSince(str, Enum):
    # Note: All options are represented in minutes
    one = 1
    five = 5
    ten = 10
    sixty = 60  # one hour
    one_eighty = int(3 * (HOUR / 60))  # three hours
    three_sixty = int(6 * (HOUR / 60))  # six hours
    day = int(24 * (HOUR / 60))  # one day


def is_command_available(cmd: str) -> bool:
    """Checks if a command is available on the system."""
    cmd_check = subprocess.run(
        f"command -v {cmd}", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    available = cmd_check.returncode == 0
    if not available:
        logger.info(f"{cmd} is not available on the system.")
    return available


def get_wrapped_server_start_cmd(flags: List[str], screen: bool, nohup: bool):
    """Add flags to the base server start command"""
    if screen:
        wrapped_cmd = START_SCREEN_CMD
    elif nohup:
        wrapped_cmd = START_NOHUP_CMD
    else:
        wrapped_cmd = SERVER_START_CMD

    if flags:
        flags_str = "".join(flags)
        wrapped_cmd = wrapped_cmd.replace(
            SERVER_START_CMD, SERVER_START_CMD + flags_str
        )

    return wrapped_cmd


def check_ray_installation():
    try:
        import ray  # noqa

    except ImportError:
        raise ImportError(
            "Ray is required for this command. "
            'You can install Ray and other server dependencies using `pip install "runhouse[server]"`'
        )
