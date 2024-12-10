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


class AutoStopColors:
    @classmethod
    def format_autostop(cls, autostop_value: int = None):
        if autostop_value is None:
            # Leave blank if autostop is not set or not relevant for the cluster type
            return None

        if autostop_value == -1:
            return "[red]Suspended[/red]"
        else:
            return f"[yellow]{autostop_value}[/yellow]"


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
        link_to_clusters_in_den = (
            f"[reset]The full list of clusters can be viewed "
            f"at https://www.run.house/resources?folder={rns_client.username}&type=cluster."
        )
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
        autostop_value = rh_cluster.get("Autostop")
        rh_cluster["Autostop (Mins)"] = AutoStopColors.format_autostop(
            int(autostop_value) if autostop_value else None
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


def print_cluster_config(cluster_config: Dict, status_type: str = StatusType.cluster):
    """
    Helping function to the `_print_status` which prints the relevant info from the cluster config.
    """
    from runhouse.main import console

    top_level_config = [
        "server_port",
        "den_auth",
        "server_connection_type",
    ]

    backend_config = ["domain", "server_host"]
    if status_type == StatusType.cluster:
        backend_config = backend_config + ["resource_subtype", "ips"]

        if cluster_config.get("resource_subtype") != "Cluster":
            backend_config.append("autostop_mins")

    for key in top_level_config:
        console.print(
            f"{BULLET_UNICODE} {key.replace('_', ' ')}: {cluster_config[key]}"
        )

    if status_type == StatusType.cluster:
        console.print(f"{BULLET_UNICODE} backend config:")

    for key in backend_config:
        if key == "autostop_mins" and cluster_config[key] == -1:
            console.print(
                f"{DOUBLE_SPACE_UNICODE}{BULLET_UNICODE} {key.replace('_', ' ')}: autostop disabled"
            )
        else:
            if (
                key == "ips"
                and cluster_config.get("resource_subtype") == "OnDemandCluster"
            ):
                val = cluster_config["compute_properties"].get("ips")
            else:
                val = cluster_config[key]

            console.print(
                f"{DOUBLE_SPACE_UNICODE}{BULLET_UNICODE} {key.replace('_', ' ')}: {val}"
            ) if status_type == StatusType.cluster else console.print(
                f"{BULLET_UNICODE} {key.replace('_', ' ')}: {val}"
            )


def print_envs_info(servlet_processes: Dict[str, Dict[str, Any]], current_cluster):
    """
    Prints info about the envs in the current_cluster: resources in each env, the CPU usage and GPU usage of the env
    (if exists)
    """
    from runhouse.main import console

    # Print headline
    envs_in_cluster_headline = "Serving 🍦 :"
    console.print(envs_in_cluster_headline)

    env_resource_mapping = {
        env: servlet_processes[env]["env_resource_mapping"] for env in servlet_processes
    }

    if len(env_resource_mapping) == 0:
        console.print("This cluster has no environment nor resources.")

    first_envs_to_print = []

    # First: if the default env does not have resources, print it.
    default_process_name = DEFAULT_PROCESS_NAME
    if len(env_resource_mapping[default_process_name]) <= 1:
        # case where the default env doesn't hve any other resources, apart from the default env itself.
        console.print(f"{BULLET_UNICODE} {default_process_name}")
        console.print(
            f"{DOUBLE_SPACE_UNICODE}This environment has only python packages installed, if provided. No "
            "resources were found."
        )

    else:
        # if the default env has other resources make sure it gets printed first
        first_envs_to_print = [default_process_name]

    # Make sure to print envs with no resources first.
    # (the only resource they have is a runhouse.env, which is the env itself).
    first_envs_to_print = first_envs_to_print + [
        env_name
        for env_name in env_resource_mapping
        if (
            len(env_resource_mapping[env_name]) <= 1
            and env_name != default_process_name
            and env_resource_mapping[env_name]
        )
    ]

    # Now, print the envs.
    # If the env have packages installed, that means that it contains an env resource. In that case:
    # * If the env contains only itself, we will print that the env contains only the installed packages.
    # * Else, we will print the resources (rh.function, th.module) associated with the env.
    envs_to_print = first_envs_to_print + [
        env_name
        for env_name in env_resource_mapping
        if env_name not in first_envs_to_print + [default_process_name]
    ]

    for env_name in envs_to_print:
        resources_in_env = env_resource_mapping[env_name]
        env_process_info = servlet_processes[env_name]

        env_name_txt = f"{BULLET_UNICODE} {env_name} | pid: {env_process_info['pid']} | node: {env_process_info['node_name']}"
        console.print(env_name_txt)

        # Print CPU info
        env_cpu_info = env_process_info.get("env_cpu_usage")
        if env_cpu_info:

            # convert bytes to GB
            memory_usage_gb = round(
                int(env_cpu_info["used_memory"]) / (1024**3),
                2,
            )
            total_cluster_memory = math.ceil(
                int(env_cpu_info["total_memory"]) / (1024**3)
            )
            cpu_memory_usage_percent = round(
                float(env_cpu_info["used_memory"] / env_cpu_info["total_memory"]),
                2,
            )
            cpu_usage_percent = round(float(env_cpu_info["utilization_percent"]), 2)

            cpu_usage_summary = f"{DOUBLE_SPACE_UNICODE}CPU: {cpu_usage_percent}% | Memory: {memory_usage_gb} / {total_cluster_memory} Gb ({cpu_memory_usage_percent}%)"

        else:
            cpu_usage_summary = (
                f"{DOUBLE_SPACE_UNICODE}CPU: This process did not use CPU memory."
            )

        console.print(cpu_usage_summary)

        # Print GPU info
        env_gpu_info = env_process_info.get("env_gpu_usage")

        # sometimes the cluster has no GPU, therefore the env_gpu_info is an empty dictionary.
        if env_gpu_info:
            # get the gpu usage info, and convert it to GB.
            total_gpu_memory = math.ceil(
                float(env_gpu_info.get("total_memory")) / (1024**3)
            )
            used_gpu_memory = round(
                float(env_gpu_info.get("used_memory")) / (1024**3), 2
            )
            gpu_memory_usage_percent = round(
                float(used_gpu_memory / total_gpu_memory) * 100, 2
            )
            gpu_usage_summery = f"{DOUBLE_SPACE_UNICODE}GPU Memory: {used_gpu_memory} / {total_gpu_memory} Gb ({gpu_memory_usage_percent}%)"
            console.print(gpu_usage_summery)

        resources_in_env = [
            {resource: resources_in_env[resource]}
            for resource in resources_in_env
            if resource is not env_name
        ]

        if len(resources_in_env) == 0:
            # No resources were found in the env, only the associated installed python reqs were installed.
            console.print(
                f"{DOUBLE_SPACE_UNICODE}This environment has only python packages installed, if provided. "
                f"No resources were found."
            )

        else:
            for resource in resources_in_env:
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

                    if resource_type == "runhouse.Function" and active_function_calls:
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
                            f" [italic bright_green]Running for {func_running_time} "
                            f"seconds[/italic bright_green]"
                        )

                    elif (
                        resource_type == "runhouse.Function"
                        and not active_function_calls
                    ):
                        is_func_running: str = " [italic bright_yellow]Currently not running[/italic bright_yellow]"

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

    has_cuda = cluster_config.get("has_cuda", False)
    cost_emoji = "💰" if has_cuda else "💸"

    num_of_cpus = cloud_properties.get("num_cpus") or len(cluster_config.get("ips"))
    num_of_gpus = 0
    cluster_accelerators = cloud_properties.get("accelerators", None)
    gpu_types = set()
    if cluster_accelerators:
        for k, v in cluster_accelerators.items():
            num_of_gpus = num_of_gpus + int(v)
            gpu_types.add(k)

    console.print(
        f"[reset]🤖 {cloud} {instance_type} cluster | 🌍 {region} | {cost_emoji} ${cost_per_hour}/hr"
    )
    cpus_gpus_info_str = f"CPUs: {int(float(num_of_cpus))}"
    if num_of_gpus > 0:
        gpu_types_str = gpu_types.pop()
        for gpu_type in gpu_types:
            gpu_types_str = gpu_types_str + f", {gpu_type}"
        cpus_gpus_info_str = (
            cpus_gpus_info_str + f" | GPUs: {num_of_gpus} (Type(s): {gpu_types_str})"
        )
    console.print(f"[reset]{cpus_gpus_info_str}")


def print_status(status_data: dict, current_cluster) -> None:
    """Prints the status of the cluster to the console"""
    from runhouse.globals import rns_client
    from runhouse.main import console

    cluster_config = status_data.get("cluster_config")
    servlet_processes = status_data.get("env_servlet_processes")

    cluster_name = cluster_config.get("name", None)
    if cluster_name:
        cluster_uri = rns_client.format_rns_address(cluster_name)
        cluster_link_in_den_ui = f"https://www.run.house/resources/{cluster_uri}"
        cluster_name_hyperlink = rich.markdown.Text(
            cluster_name, style=f"link {cluster_link_in_den_ui} white"
        )
        console.print(cluster_name_hyperlink)

    has_cuda: bool = cluster_config.get("has_cuda")

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

    # print general cpu and gpu utilization
    cluster_gpu_utilization: float = status_data.get("server_gpu_utilization")

    # Note: GPU utilization can be none if the cluster was not using its GPU when cluster.status() was invoked
    if cluster_gpu_utilization is None and has_cuda:
        cluster_gpu_utilization: float = 0.0

    cluster_cpu_utilization: float = status_data.get("server_cpu_utilization")

    server_util_info = (
        f"CPU Utilization: {round(cluster_cpu_utilization, 2)}% | GPU Utilization: {round(cluster_gpu_utilization, 2)}%"
        if has_cuda
        else f"CPU Utilization: {round(cluster_cpu_utilization, 2)}%"
    )
    console.print(server_util_info)

    # print the environments in the cluster, and the resources associated with each environment.
    print_envs_info(servlet_processes, current_cluster)


def print_bring_cluster_up_msg(
    cluster_name: str, msg_prefix="Can't execute the command"
):
    from runhouse.main import console

    console.print(
        f"{msg_prefix} because [reset]{cluster_name} is not up. To bring it up, "
        f"run [bold italic]`runhouse cluster up {cluster_name}`[/bold italic]."
    )


def get_cluster_or_local(cluster_name: str = None):
    from runhouse.main import console

    if cluster_name:
        try:
            current_cluster = rh.cluster(name=cluster_name, dryrun=True)
        except ValueError:
            console.print("Cluster not found in Den.")
            raise typer.Exit(1)

        if not current_cluster.is_up():
            console.print(
                f"Cluster [reset]{cluster_name} is not up. If it's an on-demand cluster, you can run "
                f"[reset][bold italic]`runhouse cluster up {cluster_name}`[/bold italic] to bring it up automatically."
            )
            raise typer.Exit(1)
        try:
            if current_cluster._http_client:
                current_cluster._http_client.check_server()
        except requests.exceptions.ConnectionError:
            console.print(
                f"Could not connect to the server on cluster [reset]{cluster_name}. Check that the server is up with "
                f"[reset][bold italic]`runhouse cluster status {cluster_name}`[/bold italic] or"
                f" [bold italic]`sky status -r`[/bold italic] for on-demand clusters."
            )
            raise typer.Exit(1)
        return current_cluster

    try:
        cluster_or_local = rh.here
    except ObjStoreError:
        console.print("Could not connect to Runhouse server. Is it up?")
        raise typer.Exit(1)

    if cluster_or_local == "file":
        # If running outside the cluster must specify a cluster name
        console.print(
            "Please specify a `cluster_name` or run [reset][bold italic]`runhouse server start`[/bold italic] to start "
            "a Runhouse server locally."
        )
        raise typer.Exit(1)
    elif not cluster_or_local:
        console.print(
            "\N{smiling face with horns} Runhouse Daemon is not running... \N{No Entry} \N{Runner}. "
            "Start it with [reset][bold italic]`runhouse server restart`[/bold italic] or specify a remote "
            "cluster to poll with [reset][bold italic]`runhouse cluster status <cluster_name>`[/bold italic]."
        )
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
