import logging
import subprocess
import time
import webbrowser
from pathlib import Path
from typing import Optional

import requests
import typer
from rich.console import Console

import runhouse as rh
import runhouse.rns.login

from runhouse import __version__, cluster, Cluster, configs

from runhouse.cli_utils import (
    add_clusters_to_output_table,
    check_ray_installation,
    create_output_table,
    get_local_or_remote_cluster,
    get_wrapped_server_start_cmd,
    is_command_available,
    LogsSince,
    print_cluster_config,
    print_status,
    StatusType,
)

from runhouse.constants import (
    BULLET_UNICODE,
    ITALIC_BOLD,
    MAX_CLUSTERS_DISPLAY,
    RAY_KILL_CMD,
    RAY_START_CMD,
    RESET_FORMAT,
    SERVER_LOGFILE,
    SERVER_STOP_CMD,
)
from runhouse.globals import rns_client

from runhouse.logger import get_logger

from runhouse.resources.hardware import (
    check_for_existing_ray_instance,
    get_all_sky_clusters,
    kill_actors,
)

from runhouse.resources.hardware.utils import cast_node_to_ip, ClusterStatus

from runhouse.servers.obj_store import ObjStoreError

SKY_LIVE_CLUSTERS_MSG = (
    "Live on-demand clusters created via Sky may exist that are not saved in Den. "
    "For more information, please run [bold italic]`sky status -r`[/bold italic]."
)

# create an explicit Typer application
app = typer.Typer(add_completion=False)

###############################
# Resources Command Groups
###############################

# creating a cluster app to enable subcommands of cluster (ex: runhouse cluster list).
# Register it with the main runhouse application
cluster_app = typer.Typer(help="Cluster related CLI commands.")
app.add_typer(cluster_app, name="cluster")

# creating a server app to enable subcommands of server (ex: runhouse server status).
# Register it with the main runhouse application
server_app = typer.Typer(help="Runhouse server related CLI commands.")
app.add_typer(server_app, name="server")

config_app = typer.Typer(
    help="Runhouse config related CLI commands", invoke_without_command=True
)
app.add_typer(config_app, name="config")

# For printing with typer
console = Console()

logger = get_logger(__name__)


###############################
# General Runhouse CLI commands
###############################
@app.command()
def login(
    token: Optional[str] = typer.Argument(None, help="Your Runhouse API token"),
    sync_secrets: Optional[bool] = typer.Option(
        False,
        "--sync-secrets",
        help="Whether to sync secrets. You will be prompted whether to upload local secrets or download saved secrets.",
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


###############################
# Cluster CLI commands
###############################
@cluster_app.command("ssh")
def cluster_ssh(
    cluster_name: str = typer.Argument(
        ...,
        help="Name of the cluster to SSH into.",
    ),
    node: Optional[str] = typer.Option(
        None,
        "-n",
        "--node",
        help="Specify the node by its public IP, an integer index, or specify 'head' to indicate the head node.",
    ),
):
    """SSH into a remote cluster.

    Example:
        ``$ runhouse cluster ssh rh-basic-cpu``

    """
    try:
        c = cluster(name=cluster_name)
        if isinstance(c, rh.OnDemandCluster):
            if c.cluster_status == ClusterStatus.INITIALIZING:
                console.print(
                    f"[reset]{cluster_name} is being initialized. Please wait for it to finish, or run [reset][bold italic]`runhouse cluster up {cluster_name} -f`[/bold italic] to abort the initialization and relaunch."
                )
                raise typer.Exit(0)
            node = cast_node_to_ip(node=node or "head", ips=c.ips)
            c.ssh(node=node)

        else:
            if node:
                raise ValueError(
                    "Node argument is only supported for on-demand clusters"
                )
            c.ssh()

    except ValueError as e:
        if str(e) == f"Node {node} is unsupported, could not get its IP.":
            console.print(f"[reset]{str(e)}")
            raise typer.Exit(1)
        try:
            import sky

            state = sky.status(cluster_names=[cluster_name], refresh=False)
        except:
            state = []

        if len(state) == 0:
            console.print(
                "Cluster must either be saved to Den, shared with you, or be a local ondemand cluster "
                "that is currently up."
            )
            raise typer.Exit(1)

        subprocess.run(f"ssh {cluster_name}", shell=True)


@cluster_app.command("status")
def cluster_status(
    cluster_name: str = typer.Argument(
        None,
        help="Name of the cluster to check. If not specified will check the local cluster.",
    ),
    send_to_den: bool = typer.Option(
        default=False,
        help="Whether to update Den with the status.",
    ),
    node: Optional[str] = typer.Option(
        None,
        "-n",
        "--node",
        help="Specify the node by its public IP, an integer index, or specify 'head' to indicate the head node.",
    ),
):
    """Load the status of the cluster.

    Example:
        ``$ runhouse cluster status rh-basic-cpu``
    """
    current_cluster = get_local_or_remote_cluster(cluster_name=cluster_name)

    try:
        cluster_status = current_cluster.status(send_to_den=send_to_den)
    except ValueError:
        console.print("Failed to load status for cluster.")
        return

    except ConnectionError:
        console.print(
            "\N{smiling face with horns} Runhouse Daemon is not running... \N{No Entry} \N{Runner}"
        )
        return

    print_status(cluster_status, current_cluster, node)


@cluster_app.command("list")
def cluster_list(
    show_all: bool = typer.Option(
        False,
        "-a",
        "--all",
        help=f"Get all clusters saved in Den. Up to {MAX_CLUSTERS_DISPLAY} most recently active clusters "
        f"will be displayed.",
    ),
    since: Optional[str] = typer.Option(
        None,
        "--since",
        help="Time duration to filter on. Minimum allowable filter is 1 minute. You may filter by seconds (s), "
        "minutes (m), hours (h) or days (s). Examples: 30s, 15m, 2h, 3d.",
    ),
    cluster_status: Optional[ClusterStatus] = typer.Option(
        None,
        "--status",
        help="Cluster status to filter on.",
    ),
    force: bool = typer.Option(
        False,
        "-f",
        "--force",
        help="Whether to force a status update for all relevant clusters, or load the latest values.",
    ),
):
    """
    Load Runhouse clusters

    Example:
        ``$ runhouse cluster list``

        ``$ runhouse cluster list --all``

        ``$ runhouse cluster list --status running``

        ``$ runhouse cluster list --since 15m``
    """

    # logged out case
    if not rh.configs.token:
        sky_cli_command_formatted = f"{ITALIC_BOLD}`sky status -r{RESET_FORMAT}`"  # will be printed bold and italic
        console.print(
            f"Listing clusters requires a Runhouse token. Please run `runhouse login` to get your token, "
            f"or run {sky_cli_command_formatted} to list locally stored on-demand clusters."
        )
        return

    clusters = Cluster.list(
        show_all=show_all, since=since, status=cluster_status, force=force
    )

    den_clusters = clusters.get("den_clusters", None)
    running_clusters = (
        [
            den_cluster
            for den_cluster in den_clusters
            if den_cluster.get("Status") == ClusterStatus.RUNNING
        ]
        if den_clusters
        else None
    )

    sky_clusters = clusters.get("sky_clusters", None)

    if not den_clusters:
        no_clusters_msg = f"No clusters found in Den for {rns_client.username}"
        if show_all or since or cluster_status:
            no_clusters_msg += " that match the provided filters"
        console.print(no_clusters_msg)
        if sky_clusters:
            console.print(SKY_LIVE_CLUSTERS_MSG)
        return

    filters_requested: bool = show_all or since or cluster_status

    clusters_to_print = den_clusters if filters_requested else running_clusters

    if show_all:
        # if user requesting all den cluster, limit print only to 50 clusters max.
        clusters_to_print = clusters_to_print[
            : (min(len(clusters_to_print), MAX_CLUSTERS_DISPLAY))
        ]

    # creating the clusters table
    table = create_output_table(
        total_clusters=len(den_clusters),
        running_clusters=len(running_clusters),
        displayed_clusters=len(clusters_to_print),
        filters_requested=filters_requested,
    )

    add_clusters_to_output_table(table=table, clusters=clusters_to_print)

    console.print(table)

    # print msg about un-saved live sky clusters.
    if sky_clusters:
        console.print(SKY_LIVE_CLUSTERS_MSG)


@cluster_app.command("up")
def cluster_up(
    cluster_name: str = typer.Argument(
        None,
        help="The name of cluster to bring up.",
    ),
    force: bool = typer.Option(
        False,
        "-f",
        "--force",
        help="Whether to up the cluster regardless of its current initialization status",
    ),
):
    """Bring up the cluster if it is not up. No-op if cluster is already up.
    This only applies to on-demand clusters, and has no effect on self-managed clusters.

    Note: To launch the cluster via Den, set `launcher: den` in your local `~/.rh/config.yaml`.

    Example:
        ``$ runhouse cluster up rh-basic-cpu``
        ``$ runhouse cluster up rh-basic-cpu --force``

    """
    try:
        current_cluster = rh.cluster(name=cluster_name, dryrun=True)
        if current_cluster.is_up():
            console.print("Cluster is already up.")
            return
        current_cluster.up(force=force)
    except ValueError:
        console.print("Cluster not found in Den.")
    except Exception as e:
        console.print(f"Failed to bring up the cluster: {e}")


@cluster_app.command("down")
def cluster_down(
    cluster_name: str = typer.Argument(
        None,
        help="The name of cluster to terminate.",
    ),
    remove_configs: bool = typer.Option(
        False,
        "-rm",
        "--remove-configs",
        help="Whether to delete cluster config from Den (default: ``False``).",
    ),
    remove_all: bool = typer.Option(
        False,
        "-a",
        "--all",
        help="Whether to terminate all running clusters saved in Den (default: ``False``).",
    ),
    force_deletion: bool = typer.Option(
        False, "-y", "--yes", help="Skip confirmation prompt."
    ),
):
    """Terminate cluster if it is not down. No-op if cluster is already down.
    This only applies to on-demand clusters, and has no effect on self-managed clusters.

    Example:
        ``$ runhouse cluster down rh-basic-cpu``
    """
    if not force_deletion:
        if cluster_name:
            proceed = typer.prompt(
                f"Terminating {cluster_name}. Proceed? [Y/n] (Press Enter to terminate)",
                default="",
                show_default=False,
            )
        elif remove_all:
            proceed = typer.prompt(
                "Terminating all running clusters saved in Den. Proceed? [Y/n] (Press Enter to terminate)",
                default="",
                show_default=False,
            )
        else:
            console.print("Cannot determine which cluster to terminate, aborting.")
            raise typer.Exit(1)

        if proceed.lower() in ["n", "no"]:
            console.print("Aborted!")
            raise typer.Exit(0)

    if remove_all:
        running_den_clusters = Cluster.list(status=ClusterStatus.RUNNING).get(
            "den_clusters"
        )

        if not running_den_clusters:
            console.print("No running clusters saved in Den")
            raise typer.Exit(0)

        num_terminated_clusters: int = 0
        for running_cluster in running_den_clusters:
            try:
                current_cluster = rh.cluster(
                    name=f'{running_cluster.get("Name")}',
                    dryrun=True,
                )
                if isinstance(current_cluster, rh.OnDemandCluster):
                    current_cluster.teardown_and_delete() if remove_configs else current_cluster.teardown()
                    num_terminated_clusters += 1
                else:
                    console.print(
                        f"[reset][bold italic]{current_cluster.rns_address} [reset]is not an on-demand cluster and must be terminated manually."
                    )
            except ValueError:
                continue

        if num_terminated_clusters > 0:
            console.print(
                f"Successfully terminated [reset]{num_terminated_clusters} clusters."
            )

        raise typer.Exit(0)

    try:
        current_cluster: rh.Cluster = rh.cluster(name=cluster_name, dryrun=True)
        if isinstance(current_cluster, rh.OnDemandCluster):
            current_cluster.teardown_and_delete() if remove_configs else current_cluster.teardown()
        else:
            console.print(
                f"{current_cluster.rns_address} is not an on-demand cluster and must be terminated manually."
            )
    except ValueError:
        console.print("Cluster is not saved in Den, could not bring it down.")
        sky_live_clusters = get_all_sky_clusters()
        cluster_name = (
            cluster_name.split("/")[-1] if "/" in cluster_name else cluster_name
        )
        if cluster_name in sky_live_clusters:
            console.print(
                f"You can bring down the cluster by running [italic bold] `sky down {cluster_name}`"
            )
    except Exception as e:
        console.print(f"Failed to terminate the cluster: {e}")


@cluster_app.command("logs")
def cluster_logs(
    cluster_name: str = typer.Argument(
        None,
        help="Name of cluster to load the logs from. If not specified, loads the logs of the local cluster.",
    ),
    since: Optional[LogsSince] = typer.Option(
        LogsSince.five.value,
        "--since",
        help="Get the logs from the provided timeframe (in minutes).",
    ),
):
    """Load the logs of the Runhouse server running on a cluster.

    Example:
        ``$ runhouse cluster logs rh-basic-cpu``

        ``$ runhouse cluster logs rh-basic-cpu --since 60``

    """
    current_cluster = get_local_or_remote_cluster(cluster_name=cluster_name)

    cluster_uri = rns_client.resource_uri(current_cluster.rns_address)

    resp = requests.get(
        f"{rns_client.api_server_url}/resource/{cluster_uri}/logs/file?minutes={since}",
        headers=rns_client.request_headers(),
    )

    if resp.status_code == 404:
        console.print("Cluster logs are not available yet.")
        return
    if resp.status_code != 200:
        console.print("Failed to load cluster logs.")
        return

    logs_file_url = resp.json().get("data").get("logs_presigned_url")
    logs_file_content = requests.get(logs_file_url).content.decode("utf-8")

    stripped_lines = "\n".join(line.strip() for line in logs_file_content.splitlines())
    console.print("-" * len(current_cluster.rns_address))
    console.print(f"[reset][cyan]{current_cluster.rns_address}")
    console.print("-" * len(current_cluster.rns_address))
    console.print(f"[reset][cyan]{stripped_lines}")


###############################
# Config CLI commands
###############################
@config_app.command("upload")
def config_upload():
    """Upload your local Runhouse config to Den. This will override any existing values already saved in Den."""
    configs.upload_defaults()


@config_app.command("download")
def config_download():
    """Download your Runhouse config from Den. This will override any existing values already saved locally."""
    configs.load_defaults_from_den()


@config_app.command("set")
def config_set(
    key: str = typer.Argument(
        None,
        help="Config field name",
    ),
    value: str = typer.Argument(
        None,
        help="Config value",
    ),
    sync: bool = typer.Option(
        False,
        "-s",
        "--sync",
        help="Whether to sync the updated config to Den (default: ``False``).",
    ),
):
    """Update a particular config value. Optionally sync the updated config with Den.

    Example:
        ``$ runhouse config set default_ssh_key ~/.ssh/id_rsa``

        ``$ runhouse config set default_autostop 120 --sync``
    """
    success_message = "Successfully updated local config"
    if value is None:
        console.print(f"[red]Must provide a value for {key}[/red]")
        raise typer.Exit(1)

    supported_keys = list(configs.BASE_DEFAULTS) + [
        "username",
        "default_ssh_key",
        "token",
    ]
    if key not in supported_keys:
        console.print(
            f"[yellow]Cannot set key '{key}'. Must be one of: {supported_keys}[/yellow]"
        )
        raise typer.Exit(1)

    value = value.lower()
    if value in {"true", "yes"}:
        value = True
    elif value in {"false", "no"}:
        value = False
    else:
        try:
            value = int(value)
        except ValueError:
            pass

    if value is None:
        console.print(f"[red]Invalid value for {key}[/red]")
        raise typer.Exit(1)

    configs.set(key, value)

    if sync:
        local_config = configs.load_defaults_from_file()
        configs.upload_defaults(defaults=local_config)
        console.print(f"[green]{success_message} and synced to Den[/green]")
    else:
        console.print(f"[green]{success_message}[/green]")


@config_app.callback()
def config_load(
    den: Optional[bool] = typer.Option(
        False,
        "--den",
        "-d",
        help="Whether to load the config stored in Den. If not specified loads the local config.",
    ),
    path: Optional[str] = typer.Option(
        None,
        "--path",
        "-p",
        help="Optional path to load the local Runhouse config. By default will look in ``~/.rh/config.yaml``.",
    ),
):
    """Load your Runhouse config. By default will load the version stored locally.

    Example:
        ``$ runhouse config -d``

        ``$ runhouse config -p ~/path/to/config.yaml``

    """
    if den:
        config = configs.load_defaults_from_den()
    else:
        config = configs.load_defaults_from_file(path)

    if not config:
        console.print("[red]No config found.[/red]")
        raise typer.Exit(1)

    config.pop("token", None)
    config.pop("username", None)
    for k, v in config.items():
        console.print(f"{k}: {v}")


###############################
# Server CLI commands
###############################
def _start_server(
    restart,
    restart_ray,
    screen,
    nohup,
    create_logfile=True,
    host=None,
    port=None,
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

    host_flag = f" --host {host}" if host else ""
    if host_flag:
        logger.info(f"Using host: {host}.")
        flags.append(host_flag)

    port_flag = f" --port {port}" if port else ""
    if port_flag:
        logger.info(f"Using port: {port}.")
        flags.append(port_flag)

    flags.append(" --from-python" if from_python else "")

    # Check if screen or nohup are available
    screen = screen and is_command_available("screen")
    nohup = not screen and nohup and is_command_available("nohup")

    # Create logfile if we are using backgrounding
    if (screen or nohup) and create_logfile and not Path(SERVER_LOGFILE).exists():
        Path(SERVER_LOGFILE).parent.mkdir(parents=True, exist_ok=True)
        Path(SERVER_LOGFILE).touch()

    # Add flags to the server start command
    cmds.append(get_wrapped_server_start_cmd(flags, screen, nohup))
    logger.info(f"Starting API server using the following command: {cmds[-1]}.")

    try:
        # Open and read the lines of the server logfile so we only print the most recent lines after starting
        f = None
        if (screen or nohup) and Path(SERVER_LOGFILE).exists():
            f = open(SERVER_LOGFILE, "r")
            f.readlines()  # Discard these, they're from the previous times the server was started

        # We do these one by one so it's more obvious where the error is if there is one
        for _, cmd in enumerate(cmds):
            console.print(f"Executing `{cmd}`")
            result = subprocess.run(cmd, shell=True)

            if result.returncode != 0:
                # We don't want to raise an error if the server kill fails, as it may simply not be running
                if "pkill" in cmd:
                    continue

                # Retry ray start in case pkill process did not complete in time, up to 10s
                if RAY_START_CMD in cmd:
                    console.print("Retrying:")
                    attempt = 0
                    while result.returncode != 0 and attempt < 10:
                        attempt += 1
                        time.sleep(1)
                        result = subprocess.run(
                            cmd, capture_output=True, text=True, shell=True
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

    except FileNotFoundError as e:
        console.print(f"Encountered FileNotFoundError {str(e)}")
        raise typer.Exit(1)


###############################
# Server CLI commands
###############################


@server_app.command("start")
def server_start(
    cluster_name: Optional[str] = typer.Argument(
        None,
        help="Specify a *saved* remote cluster to start the Runhouse server on that cluster. "
        "If not provided, the locally running server will be started.",
    ),
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
        None, help="Port for server. If not specified will start on 32300."
    ),
    from_python: bool = typer.Option(
        False,
        help="Whether HTTP server started from inside a Python call rather than CLI.",
    ),
):
    """
    Start the HTTP server on the cluster.

    Example:
        ``$ runhouse server start``

        ``$ runhouse server start rh-cpu``
    """
    check_ray_installation()

    # If server is already up, ask the user to restart the server instead.
    if not cluster_name:
        server_status_cmd = "runhouse server status"
        result = subprocess.run(
            server_status_cmd, shell=True, capture_output=True, text=True
        )

        if result.returncode == 0 and "Server is up" in result.stdout:
            console.print(
                "Local Runhouse server is already running. To restart it, please "
                "run [italic bold]`runhouse server restart` [reset]with the relevant parameters."
            )
            raise typer.Exit(0)

    if cluster_name:
        c = get_local_or_remote_cluster(cluster_name=cluster_name)
        c.start_server(resync_rh=resync_rh, restart_ray=restart_ray)
        return

    _start_server(
        restart=False,
        restart_ray=restart_ray,
        screen=screen,
        nohup=nohup,
        create_logfile=True,
        host=host,
        port=port,
        from_python=from_python,
    )


@server_app.command("restart")
def server_restart(
    cluster_name: str = typer.Argument(
        None,
        help="Specify a *saved* remote cluster to restart the Runhouse server on that cluster. "
        "If not provided, the locally running server will be restarted.",
    ),
    restart_ray: bool = typer.Option(True, help="Restart the Ray runtime."),
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
        None, help="Port for server. If not specified will start on 32300."
    ),
    from_python: bool = typer.Option(
        False,
        help="Whether HTTP server started from inside a Python call rather than CLI.",
    ),
):
    """Restart the HTTP server on the cluster.

    Example:
        ``$ runhouse server restart``

        ``$ runhouse server restart rh-cpu``
    """
    check_ray_installation()

    if cluster_name:
        c = get_local_or_remote_cluster(cluster_name=cluster_name)
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
        from_python=from_python,
    )


@server_app.command("stop")
def server_stop(
    cluster_name: Optional[str] = typer.Argument(
        None,
        help="Specify a *saved* remote cluster to stop the Runhouse server on that cluster. "
        "If not provided, the locally running server will be stopped.",
    ),
    stop_ray: bool = typer.Option(False, help="Stop the Ray runtime."),
    cleanup_actors: bool = typer.Option(True, help="Kill all Ray actors."),
):
    """Stop the HTTP server on the cluster.

    Example:
        ``$ runhouse server stop``

        ``$ runhouse server stop rh-cpu``
    """
    check_ray_installation()
    logger.debug("Stopping the server")

    if cluster_name:
        current_cluster = get_local_or_remote_cluster(cluster_name=cluster_name)
        current_cluster.stop_server(stop_ray=stop_ray, cleanup_actors=cleanup_actors)
        return

    subprocess.run(SERVER_STOP_CMD, shell=True)

    if cleanup_actors:
        import ray

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

    console.print("Server stopped.")


@server_app.command("status")
def server_status(
    cluster_name: str = typer.Argument(
        None,
        help="Specify a *saved* remote cluster to check the status of the Runhouse server on that cluster. "
        "If not provided, the status of the locally running server will be checked.",
    ),
):
    """Check the HTTP server status on the cluster.

    Example:
        ``$ runhouse server status``

        ``$ runhouse server status rh-cpu``
    """
    check_ray_installation()
    logger.debug("Checking the server status.")
    current_cluster = get_local_or_remote_cluster(cluster_name=cluster_name)
    try:
        status = current_cluster.status()
        console.print(f"[reset]{BULLET_UNICODE} server pid: {status.get('server_pid')}")
        print_cluster_config(
            cluster_config=status.get("cluster_config"), status_type=StatusType.server
        )
        if rh.here == "file":
            console.print(
                "[reset]For more detailed information about the cluster, please use [italic bold]`runhouse cluster status <cluster_name>`[/italic bold]"
            )
    except (ObjStoreError, ConnectionError):
        console.print("Could not connect to Runhouse server. Is it up?")


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
