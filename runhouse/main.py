import datetime
import importlib
import logging
import math
import shlex
import subprocess
import time
import webbrowser
from pathlib import Path
from typing import Any, Dict, List, Optional

import ray

import requests
import rich.markdown

import typer
import yaml
from rich.console import Console

import runhouse as rh

import runhouse.rns.login

from runhouse import __version__, cluster, Cluster, configs
from runhouse.constants import (
    BULLET_UNICODE,
    DOUBLE_SPACE_UNICODE,
    RAY_KILL_CMD,
    RAY_START_CMD,
    SERVER_LOGFILE,
    SERVER_START_CMD,
    SERVER_STOP_CMD,
    START_NOHUP_CMD,
    START_SCREEN_CMD,
)
from runhouse.globals import rns_client
from runhouse.logger import get_logger
from runhouse.resources.hardware.ray_utils import (
    check_for_existing_ray_instance,
    kill_actors,
)

# create an explicit Typer application
app = typer.Typer(add_completion=False)

# For printing with typer
console = Console()

logger = get_logger(__name__)


@app.command()
def login(
    token: Optional[str] = typer.Argument(None, help="Your Runhouse API token"),
    sync_secrets: Optional[bool] = typer.Option(
        False,
        "--sync-secrets",
        help="Whether to sync secrets. You will be prompted whether to upload local secrets or download saved secrets",
    ),
    yes: Optional[bool] = typer.Option(
        False, "--yes", "-y", help="Sets any confirmations to 'yes' automatically."
    ),
):
    """Login to Runhouse. Validates token provided, with options to upload or download stored secrets or config between
    local environment and Runhouse / Vault.
    """
    valid_token: str = (
        runhouse.rns.login.login(
            token=token,
            download_config=True,
            upload_config=True,
            download_secrets=True,
            upload_secrets=True,
            from_cli=True,
        )
        if yes
        else runhouse.rns.login.login(
            token=token,
            interactive=True,
            ret_token=True,
            from_cli=True,
            sync_secrets=sync_secrets,
        )
    )

    if valid_token:
        webbrowser.open(f"{configs.get('dashboard_url')}/dashboard?token={valid_token}")
        raise typer.Exit()
    else:
        raise typer.Exit(code=1)


@app.command()
def logout():
    """Logout of Runhouse. Provides options to delete locally configured secrets and local Runhouse configs"""
    runhouse.rns.login.logout(interactive=True)
    raise typer.Exit()


@app.command()
def notebook(
    cluster_name: str, up: bool = typer.Option(False, help="Start the cluster")
):
    """Open a Jupyter notebook on a cluster."""
    c = cluster(name=cluster_name)
    if up:
        c.up_if_not()
    if not c.is_up():
        console.print(
            f"Cluster {cluster_name} is not up. Please run `runhouse notebook {cluster_name} --up` to bring "
            f"it up if it is an on-demand cluster."
        )
        raise typer.Exit(1)
    c.notebook()


@app.command()
def ssh(cluster_name: str, up: bool = typer.Option(False, help="Start the cluster")):
    """SSH into a cluster created elsewhere (so `ssh cluster` doesn't work out of the box) or not yet up."""
    try:
        c = cluster(name=cluster_name)

        if not c.is_shared:
            if up:
                try:
                    c.up_if_not()
                except NotImplementedError:
                    console.print(
                        f"Cluster {cluster_name} is not an on-demand cluster, so it can't be brought up automatically."
                        f"Please start it manually and re-save the cluster with the new connection info in Python."
                    )
                    raise typer.Exit(1)
            elif not c.is_up():
                console.print(
                    f"Cluster {cluster_name} is not up. Please run `runhouse ssh {cluster_name} --up`."
                )
                raise typer.Exit(1)

        c.ssh()
    except ValueError:
        import sky

        from runhouse import OnDemandCluster
        from runhouse.constants import DEFAULT_SSH_PORT
        from runhouse.resources.hardware.utils import _run_ssh_command

        state = sky.status(cluster_names=[cluster_name], refresh=False)

        if len(state) == 0:
            console.print(
                f"Could not load cluster called {cluster_name}. Cluster must either be saved to Den, "
                "or be an ondemand cluster that is currently up."
            )

            raise typer.Exit(1)

        resource_handle = state[0].get("handle", {})

        sky_ssh_config = Path(f"~/.sky/generated/{cluster_name}.yml").expanduser()

        if sky_ssh_config.exists():
            with open(sky_ssh_config, "r", encoding="utf-8") as f:
                ssh_config = yaml.safe_load(f)
            ssh_proxy_command = ssh_config["auth"].get("ssh_proxy_command", None)
            if ssh_proxy_command:
                ssh_proxy_command = ssh_proxy_command.replace(
                    "skypilot:ssh_user", resource_handle.ssh_user
                )

        else:
            ssh_proxy_command = None

        _run_ssh_command(
            address=resource_handle.head_ip,
            ssh_user=resource_handle.ssh_user or "ubuntu",
            ssh_port=resource_handle.stable_ssh_ports[0] or DEFAULT_SSH_PORT,
            ssh_private_key=OnDemandCluster.DEFAULT_KEYFILE,
            docker_user=resource_handle.docker_user,
            ssh_proxy_command=ssh_proxy_command,
        )


###############################
# Status helping functions
###############################


def _adjust_resource_type(resource_type: str):
    """
    status helping function. transforms a str form runhouse.resources.{X.Y...}.resource_type to runhouse.resource_type
    """
    try:
        resource_type = resource_type.split(".")[-1]
        getattr(importlib.import_module("runhouse"), resource_type)
        return f"runhouse.{resource_type}"
    except AttributeError:
        return resource_type


def _resource_name_to_rns(name: str):
    """
    If possible, transform the resource name to a rns address.
    If not, return the name as is (it is the key in the object store).
    """
    resource_config = rns_client.load_config(name)
    if resource_config and resource_config.get("resource_type") != "env":
        return resource_config.get("name")
    else:
        return name


def _print_cluster_config(cluster_config: Dict):
    """
    Helping function to the `_print_status` which prints the relevant info from the cluster config.
    """
    # TODO [SB]: need to modify printing format (colour palette etc).

    top_level_config = [
        "server_port",
        "den_auth",
        "server_connection_type",
    ]

    backend_config = ["resource_subtype", "domain", "server_host", "ips"]

    if cluster_config.get("resource_subtype") != "Cluster":
        backend_config.append("autostop_mins")

    if cluster_config.get("default_env") and isinstance(
        cluster_config.get("default_env"), Dict
    ):
        cluster_config["default_env"] = cluster_config["default_env"]["name"]

    for key in top_level_config:
        console.print(
            f"{BULLET_UNICODE} {key.replace('_', ' ')}: {cluster_config[key]}"
        )

    console.print(f"{BULLET_UNICODE} backend config:")
    for key in backend_config:
        if key == "autostop_mins" and cluster_config[key] == -1:
            console.print(
                f"{DOUBLE_SPACE_UNICODE}{BULLET_UNICODE} {key.replace('_', ' ')}: autostop disabled"
            )
        else:
            console.print(
                f"{DOUBLE_SPACE_UNICODE}{BULLET_UNICODE} {key.replace('_', ' ')}: {cluster_config[key]}"
            )


def _print_envs_info(
    env_servlet_processes: Dict[str, Dict[str, Any]], current_cluster: Cluster
):
    """
    Prints info about the envs in the current_cluster: resources in each env, the CPU usage and GPU usage of the env
    (if exists)
    """

    # Print headline
    envs_in_cluster_headline = "Serving 🍦 :"
    console.print(envs_in_cluster_headline)

    env_resource_mapping = {
        env: env_servlet_processes[env]["env_resource_mapping"]
        for env in env_servlet_processes
    }

    if len(env_resource_mapping) == 0:
        console.print("This cluster has no environment nor resources.")

    first_envs_to_print = []

    # First: if the default env does not have resources, print it.
    default_env_name = current_cluster.default_env.name
    if len(env_resource_mapping[default_env_name]) <= 1:
        # case where the default env doesn't hve any other resources, apart from the default env itself.
        console.print(f"{BULLET_UNICODE} {default_env_name} (runhouse.Env)")
        console.print(
            f"{DOUBLE_SPACE_UNICODE}This environment has only python packages installed, if provided. No "
            "resources were found."
        )

    else:
        # case where the default env have other resources. We make sure that our of all the envs which have resources,
        # the default_env will be printed first.
        first_envs_to_print = [default_env_name]

    # Make sure to print envs with no resources first.
    # (the only resource they have is a runhouse.env, which is the env itself).
    first_envs_to_print = first_envs_to_print + [
        env_name
        for env_name in env_resource_mapping
        if (
            len(env_resource_mapping[env_name]) <= 1
            and env_name != default_env_name
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
        if env_name not in first_envs_to_print + [default_env_name]
        and env_resource_mapping[env_name]
    ]

    for env_name in envs_to_print:
        resources_in_env = env_resource_mapping[env_name]
        env_process_info = env_servlet_processes[env_name]

        # sometimes the env itself is not a resource (key) inside the env's servlet.
        if len(resources_in_env) == 0:
            env_type = "runhouse.Env"
        else:
            env_type = _adjust_resource_type(
                resources_in_env[env_name]["resource_type"]
            )

        env_name_txt = f"{BULLET_UNICODE} {env_name} ({env_type}) | pid: {env_process_info['pid']} | node: {env_process_info['node_name']}"
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

            cpu_usage_summery = f"{DOUBLE_SPACE_UNICODE}CPU: {cpu_usage_percent}% | Memory: {memory_usage_gb} / {total_cluster_memory} Gb ({cpu_memory_usage_percent}%)"

        else:
            cpu_usage_summery = (
                f"{DOUBLE_SPACE_UNICODE}CPU: This process did not use CPU memory."
            )

        console.print(cpu_usage_summery)

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
                f"{DOUBLE_SPACE_UNICODE}This environment has only python packages installed, if provided. No resources were "
                "found."
            )

        else:
            for resource in resources_in_env:
                for resource_name, resource_info in resource.items():
                    resource_type = _adjust_resource_type(
                        resource_info.get("resource_type")
                    )

                    active_function_calls = resource_info.get("active_function_calls")
                    resource_info_str = f"{DOUBLE_SPACE_UNICODE}{BULLET_UNICODE} {resource_name} ({resource_type})"

                    if resource_type == "runhouse.Function" and active_function_calls:
                        func_start_time_utc = active_function_calls[0].get(
                            "start_time", None
                        )

                        # casting func_start_time_utc to datetime format
                        func_start_time_utc = datetime.datetime.fromtimestamp(
                            func_start_time_utc, tz=datetime.timezone.utc
                        )

                        # func_end_time_utc = current time. Making sure it is in the same format as func_start_time_utc,
                        # so we could calculate function's running time.
                        func_end_time_utc = datetime.datetime.fromtimestamp(
                            time.time(), tz=datetime.timezone.utc
                        )

                        func_running_time = (
                            func_end_time_utc - func_start_time_utc
                        ).total_seconds()

                        is_func_running: str = f" [italic bright_green]Running for {func_running_time} seconds[/italic bright_green]"

                    elif (
                        resource_type == "runhouse.Function"
                        and not active_function_calls
                    ):
                        is_func_running: str = " [italic bright_yellow]Currently not running[/italic bright_yellow]"

                    else:
                        is_func_running: str = ""

                    resource_info_str = resource_info_str + is_func_running

                    console.print(resource_info_str)


def _print_cloud_properties(cluster_config: dict):
    cloud_properties = cluster_config.get("launched_properties", None)
    if not cloud_properties:
        return
    cloud = cloud_properties.get("cloud")
    instance_type = cloud_properties.get("instance_type")
    region = cloud_properties.get("region")
    cost_per_hour = cloud_properties.get("cost_per_hour")

    has_cuda = cluster_config.get("has_cuda", False)
    cost_emoji = "💰" if has_cuda else "💸"

    num_of_instances = len(cluster_config.get("ips"))
    num_of_instances_str = f"{num_of_instances}x " if num_of_instances > 1 else ""

    print(
        f"🤖 {num_of_instances_str}{cloud} {instance_type} cluster | 🌍 {region} | {cost_emoji} ${cost_per_hour}/hr"
    )


def _get_resource_link_in_den_ui(cluster_name: str, api_server_url: str):
    cluster_uri = rns_client.format_rns_address(cluster_name)
    link_to_den_dashboard = f"{api_server_url}/resources/{cluster_uri}"
    return link_to_den_dashboard


def _print_status(status_data: dict, current_cluster: Cluster) -> None:
    """Prints the status of the cluster to the console"""
    cluster_config = status_data.get("cluster_config")
    env_servlet_processes = status_data.get("env_servlet_processes")

    cluster_name = cluster_config.get("name", None)

    if cluster_name:
        api_server_url = cluster_config.get(
            "api_server_url", rh.configs.get("api_server_url")
        )
        api_server_url = api_server_url.replace(
            "api", "www"
        )  # convert the api link to the ui link.
        cluster_link_in_den_ui = _get_resource_link_in_den_ui(
            cluster_name=cluster_name, api_server_url=api_server_url
        )
        cluster_name_hyperlink = rich.markdown.Text(
            cluster_name, style=f"link {cluster_link_in_den_ui} white"
        )
        console.print(cluster_name_hyperlink)

    has_cuda: bool = cluster_config.get("has_cuda")

    # print headline
    daemon_headline_txt = (
        "\N{smiling face with horns} Runhouse Daemon is running \N{Runner}"
    )
    console.print(daemon_headline_txt, style="bold royal_blue1")

    console.print(f"Runhouse v{status_data.get('runhouse_version')}")
    _print_cloud_properties(cluster_config)
    console.print(f"server pid: {status_data.get('server_pid')}")

    # Print relevant info from cluster config.
    _print_cluster_config(cluster_config)

    # print general cpu and gpu utilization
    cluster_gpu_utilization: float = status_data.get("server_gpu_utilization")

    # cluster_gpu_utilization can be none, if the cluster was not using its GPU at the moment cluster.status() was invoked.
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
    _print_envs_info(env_servlet_processes, current_cluster)


@app.command()
def status(
    cluster_name: str = typer.Argument(
        None,
        help="Name of cluster to check. If not specified will check the local cluster.",
    ),
    send_to_den: bool = typer.Option(
        default=False,
        help="Whether to update Den with the status",
    ),
):
    """Load the status of the Runhouse daemon running on a cluster."""
    cluster_or_local = rh.here

    if cluster_or_local == "file" and not cluster_name:
        # If running outside the cluster must specify a cluster name
        console.print("Missing argument `cluster_name`.")
        return

    elif cluster_name:
        current_cluster = cluster(name=cluster_name)
        if not current_cluster.is_up():
            console.print(
                f"Cluster {cluster_name} is not up. If it's an on-demand cluster, you can run "
                f"`runhouse ssh --up {cluster_name}` to bring it up automatically."
            )
            raise typer.Exit(1)
        try:
            if current_cluster._http_client:
                current_cluster._http_client.check_server()
        except requests.exceptions.ConnectionError:
            console.print(
                f"Could not connect to the server on cluster {cluster_name}. Check that the server is up with "
                f"`runhouse ssh {cluster_name}` or `sky status -r` for on-demand clusters."
            )
            raise typer.Exit(1)

    elif not cluster_or_local:
        console.print(
            "\N{smiling face with horns} Runhouse Daemon is not running... \N{No Entry} \N{Runner}. "
            "Start it with `runhouse restart` or specify a remote "
            "cluster to poll with `runhouse status <cluster_name>`."
        )
        raise typer.Exit(1)

    else:
        # we are inside the cluster
        current_cluster = cluster_or_local  # cluster_or_local = rh.here

    try:
        cluster_status = current_cluster.status(send_to_den=send_to_den)

    except ValueError:
        console.print("Failed to load status for cluster.")
        return
    except requests.exceptions.ConnectionError:
        console.print(
            "\N{smiling face with horns} Runhouse Daemon is not running... \N{No Entry} \N{Runner}"
        )
        return

    _print_status(cluster_status, current_cluster)


def load_cluster(cluster_name: str):
    """Load a cluster from RNS into the local environment, e.g. to be able to ssh."""
    c = cluster(name=cluster_name)
    if not c.address:
        c._update_from_sky_status(dryrun=True)


def _check_if_command_exists(cmd: str):
    cmd_check = subprocess.run(
        f"command -v {cmd}", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    available = cmd_check.returncode == 0
    if not available:
        logger.info(f"{cmd} is not available on the system.")
    return available


def _get_wrapped_server_start_cmd(flags: List[str], screen: bool, nohup: bool):
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


def _start_server(
    restart,
    restart_ray,
    screen,
    nohup,
    create_logfile=True,
    host=None,
    port=None,
    use_https=False,
    den_auth=False,
    ssl_keyfile=None,
    ssl_certfile=None,
    restart_proxy=False,
    use_caddy=False,
    domain=None,
    certs_address=None,
    api_server_url=None,
    default_env_name=None,
    conda_env=None,
    from_python=None,
):
    ############################################
    # Build CLI commands to start the server
    ############################################
    cmds = []
    if restart:
        cmds.append(SERVER_STOP_CMD)

    # We have to `ray start` not within screen/nohup
    existing_ray_instance = check_for_existing_ray_instance()
    if not existing_ray_instance or restart_ray:
        cmds.append(RAY_KILL_CMD)
        cmds.append(RAY_START_CMD)

    # Collect flags
    flags = []

    den_auth_flag = " --use-den-auth" if den_auth else ""
    if den_auth_flag:
        logger.info("Starting server with Den auth.")
        flags.append(den_auth_flag)

    restart_proxy_flag = " --restart-proxy" if restart_proxy else ""
    if restart_proxy_flag:
        logger.info("Reinstalling server configs.")
        flags.append(restart_proxy_flag)

    use_caddy_flag = " --use-caddy" if use_caddy else ""
    if use_caddy_flag:
        logger.info("Configuring Caddy on the cluster.")
        flags.append(use_caddy_flag)

    ssl_keyfile_flag = f" --ssl-keyfile {ssl_keyfile}" if ssl_keyfile else ""
    if ssl_keyfile_flag:
        logger.info(f"Using SSL keyfile in path: {ssl_keyfile}")
        flags.append(ssl_keyfile_flag)

    ssl_certfile_flag = f" --ssl-certfile {ssl_certfile}" if ssl_certfile else ""
    if ssl_certfile_flag:
        logger.info(f"Using SSL certfile in path: {ssl_certfile}")
        flags.append(ssl_certfile_flag)

    domain = f" --domain {domain}" if domain else ""
    if domain:
        logger.info(f"Using domain: {domain}")
        flags.append(domain)

    # Use HTTPS if explicitly specified or if SSL cert or keyfile path are provided
    https_flag = " --use-https" if use_https or (ssl_keyfile or ssl_certfile) else ""
    if https_flag:
        logger.info("Starting server with HTTPS.")
        flags.append(https_flag)

    host_flag = f" --host {host}" if host else ""
    if host_flag:
        logger.info(f"Using host: {host}.")
        flags.append(host_flag)

    port_flag = f" --port {port}" if port else ""
    if port_flag:
        logger.info(f"Using port: {port}.")
        flags.append(port_flag)

    address_flag = f" --certs-address {certs_address}" if certs_address else ""
    if address_flag:
        logger.info(f"Server public IP address: {certs_address}.")
        flags.append(address_flag)

    api_server_url_flag = (
        f" --api-server-url {api_server_url}" if api_server_url else ""
    )
    if api_server_url_flag:
        logger.info(f"Setting api_server url to {api_server_url}")
        flags.append(api_server_url_flag)

    default_env_flag = (
        f" --default-env-name {default_env_name}" if default_env_name else ""
    )
    if default_env_flag:
        logger.info(f"Starting server in default env named: {default_env_name}")
        flags.append(default_env_flag)

    conda_env_flag = f" --conda-env {conda_env}" if conda_env else ""
    if conda_env_flag:
        logger.info(f"Creating runtime env for conda env: {conda_env}")
        flags.append(conda_env_flag)

    flags.append(" --from-python" if from_python else "")

    # Check if screen or nohup are available
    screen = screen and _check_if_command_exists("screen")
    nohup = not screen and nohup and _check_if_command_exists("nohup")

    # Create logfile if we are using backgrounding
    if (screen or nohup) and create_logfile and not Path(SERVER_LOGFILE).exists():
        Path(SERVER_LOGFILE).parent.mkdir(parents=True, exist_ok=True)
        Path(SERVER_LOGFILE).touch()

    # Add flags to the server start command
    cmds.append(_get_wrapped_server_start_cmd(flags, screen, nohup))
    logger.info(f"Starting API server using the following command: {cmds[-1]}.")

    try:
        # Open and read the lines of the server logfile so we only print the most recent lines after starting
        f = None
        if (screen or nohup) and Path(SERVER_LOGFILE).exists():
            f = open(SERVER_LOGFILE, "r")
            f.readlines()  # Discard these, they're from the previous times the server was started

        # We do these one by one so it's more obvious where the error is if there is one
        for i, cmd in enumerate(cmds):
            console.print(f"Executing `{cmd}`")
            if (
                i == len(cmds) - 1
            ):  # last cmd is not being parsed correctly when ran with shlex.split
                result = subprocess.run(cmd, shell=True, check=True)
            else:
                result = subprocess.run(shlex.split(cmd), text=True)

            if result.returncode != 0:
                # We don't want to raise an error if the server kill fails, as it may simply not be running
                if "pkill" in cmd:
                    continue

                # Retry ray start in case pkill process did not complete in time, up to 10s
                if cmd == RAY_START_CMD:
                    console.print("Retrying:")
                    attempt = 0
                    while result.returncode != 0 and attempt < 10:
                        attempt += 1
                        time.sleep(1)
                        result = subprocess.run(
                            shlex.split(cmd), text=True, capture_output=True
                        )
                        if result.stderr and "ConnectionError" not in result.stderr:
                            break
                    if result.returncode == 0:
                        continue

                console.print(f"Error while executing `{cmd}`")
                raise typer.Exit(1)

        server_started_str = "Uvicorn running on"
        # Read and print the server logs until the
        if screen or nohup:
            while not Path(SERVER_LOGFILE).exists():
                time.sleep(1)
            f = f or open(SERVER_LOGFILE, "r")
            start_time = time.time()
            # Wait for input for 60 seconds max (for Caddy to download and set up)
            while time.time() - start_time < 60:
                for line in f:
                    if server_started_str in line:
                        console.print(line)
                        f.close()
                        return
                    else:
                        console.print(line, end="")
                time.sleep(1)
            f.close()

    except FileNotFoundError:
        console.print(
            "python3 command was not found. Make sure you have python3 installed."
        )
        raise typer.Exit(1)


@app.command()
def start(
    restart_ray: bool = typer.Option(True, help="Restart the Ray runtime"),
    screen: bool = typer.Option(False, help="Start the server in a screen"),
    nohup: bool = typer.Option(
        False, help="Start the server in a nohup if screen is not available"
    ),
    host: Optional[str] = typer.Option(
        None, help="Custom server host address. Default is `0.0.0.0`."
    ),
    port: Optional[str] = typer.Option(
        None, help="Port for server. If not specified will start on 32300"
    ),
    use_https: bool = typer.Option(
        False, help="Start an HTTPS server with TLS verification"
    ),
    use_den_auth: bool = typer.Option(
        False, help="Whether to authenticate requests with a Runhouse token"
    ),
    use_caddy: bool = typer.Option(
        False,
        help="Whether to configure Caddy on the cluster as a reverse proxy.",
    ),
    domain: str = typer.Option(
        None,
        help="Server domain. Relevant if using Caddy to automate generating CA verified certs.",
    ),
    certs_address: Optional[str] = typer.Option(
        None,
        help="Public IP address of the server. Required for generating self-signed certs and enabling HTTPS",
    ),
    default_env_name: str = typer.Option(
        None, help="Default env to start the server on."
    ),
    conda_env: str = typer.Option(
        None, help="Name of conda env corresponding to default env if it is a CondaEnv."
    ),
):
    """Start the HTTP or HTTPS server on the cluster."""
    _start_server(
        restart=False,
        restart_ray=restart_ray,
        screen=screen,
        nohup=nohup,
        create_logfile=True,
        host=host,
        port=port,
        use_https=use_https,
        den_auth=use_den_auth,
        use_caddy=use_caddy,
        domain=domain,
        certs_address=certs_address,
        default_env_name=default_env_name,
        conda_env=conda_env,
    )


@app.command()
def restart(
    name: str = typer.Option(None, help="A *saved* remote cluster object to restart."),
    restart_ray: bool = typer.Option(True, help="Restart the Ray runtime"),
    screen: bool = typer.Option(
        True,
        help="Start the server in a screen. Only relevant when restarting locally.",
    ),
    nohup: bool = typer.Option(
        True,
        help="Start the server in a nohup if screen is not available. Only relevant when restarting locally.",
    ),
    resync_rh: bool = typer.Option(
        False,
        help="Resync the Runhouse package. Only relevant when restarting remotely.",
    ),
    host: Optional[str] = typer.Option(
        None, help="Custom server host address. Default is `0.0.0.0`."
    ),
    port: Optional[str] = typer.Option(
        None, help="Port for server. If not specified will start on 32300"
    ),
    use_https: bool = typer.Option(
        False, help="Start an HTTPS server with TLS verification"
    ),
    use_den_auth: bool = typer.Option(
        False, help="Whether to authenticate requests with a Runhouse token"
    ),
    ssl_keyfile: Optional[str] = typer.Option(
        None, help="Path to custom SSL key file to use for enabling HTTPS"
    ),
    ssl_certfile: Optional[str] = typer.Option(
        None, help="Path to custom SSL cert file to use for enabling HTTPS"
    ),
    restart_proxy: bool = typer.Option(
        False, help="Whether to reinstall server configs on the cluster"
    ),
    use_caddy: bool = typer.Option(
        False,
        help="Whether to configure Caddy on the cluster as a reverse proxy.",
    ),
    domain: str = typer.Option(
        None,
        help="Server domain. Relevant if using Caddy to automate generating CA verified certs.",
    ),
    certs_address: Optional[str] = typer.Option(
        None,
        help="Public IP address of the server. Required for generating self-signed certs and enabling HTTPS",
    ),
    api_server_url: str = typer.Option(
        default="https://api.run.house",
        help="URL of Runhouse Den",
    ),
    default_env_name: str = typer.Option(
        None, help="Default env to start the server on."
    ),
    conda_env: str = typer.Option(
        None, help="Name of conda env corresponding to default env if it is a CondaEnv."
    ),
    from_python: bool = typer.Option(
        False,
        help="Whether HTTP server started from inside a Python call rather than CLI.",
    ),
):
    """Restart the HTTP server on the cluster."""
    if name:
        c = cluster(name=name)
        c.restart_server(resync_rh=resync_rh, restart_ray=restart_ray)
        return

    _start_server(
        restart=True,
        restart_ray=restart_ray,
        screen=screen,
        nohup=nohup,
        create_logfile=True,
        host=host,
        port=port,
        use_https=use_https,
        den_auth=use_den_auth,
        ssl_keyfile=ssl_keyfile,
        ssl_certfile=ssl_certfile,
        restart_proxy=restart_proxy,
        use_caddy=use_caddy,
        domain=domain,
        certs_address=certs_address,
        api_server_url=api_server_url,
        default_env_name=default_env_name,
        conda_env=conda_env,
        from_python=from_python,
    )


@app.command()
def stop(
    stop_ray: bool = typer.Option(False, help="Stop the Ray runtime"),
    cleanup_actors: bool = typer.Option(True, help="Kill all Ray actors"),
):
    logger.info("Stopping the server.")
    subprocess.run(SERVER_STOP_CMD, shell=True)

    if cleanup_actors:
        ray.init(
            address="auto",
            ignore_reinit_error=True,
            logging_level=logging.ERROR,
            namespace="runhouse",
        )

        kill_actors(namespace="runhouse", gracefully=False)

    if stop_ray:
        logger.info("Stopping Ray.")
        subprocess.run(RAY_KILL_CMD, shell=True)


@app.callback(invoke_without_command=True, help="Runhouse CLI")
def main(
    ctx: typer.Context,
    version: bool = typer.Option(
        None, "--version", "-v", help="Show the version and exit."
    ),
):
    """
    Runhouse CLI
    """
    if version:
        print(f"{__version__}")
    elif ctx.invoked_subcommand is None:
        subprocess.run("runhouse --help", shell=True)
