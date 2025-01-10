import logging
import shlex
import subprocess
import time
import webbrowser
from pathlib import Path
from typing import Optional

import ray

import requests
import typer
from rich.console import Console

import runhouse as rh
import runhouse.rns.login

from runhouse import __version__, cluster, Cluster, configs

from runhouse.cli_utils import (
    add_clusters_to_output_table,
    create_output_table,
    get_cluster_or_local,
    get_wrapped_server_start_cmd,
    is_command_available,
    LogsSince,
    print_bring_cluster_up_msg,
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
from runhouse.resources.hardware.utils import ClusterStatus

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


###############################
# Cluster CLI commands
###############################
@cluster_app.command("ssh")
def cluster_ssh(cluster_name: str):
    """SSH into a remote cluster.

    Example:
        ``$ runhouse cluster ssh rh-basic-cpu``

    """
    try:
        c = cluster(name=cluster_name)
        if not c.is_up():
            print_bring_cluster_up_msg(cluster_name=cluster_name)
            return

        c.ssh()

    except ValueError:
        try:
            import sky

            state = sky.status(cluster_names=[cluster_name], refresh=False)
        except:
            state = []

        if len(state) == 0:
            console.print(
                f"Could not load cluster called {cluster_name}. Cluster must either be saved to Den, "
                "or be a local ondemand cluster that is currently up."
            )
            raise typer.Exit(1)

        subprocess.run(f"ssh {cluster_name}", shell=True)


@cluster_app.command("status")
def cluster_status(
    cluster_name: str = typer.Argument(
        None,
        help="Name of cluster to check. If not specified will check the local cluster.",
    ),
    send_to_den: bool = typer.Option(
        default=False,
        help="Whether to update Den with the status.",
    ),
):
    """Load the status of the cluster.

    Example:
        ``$ runhouse cluster status rh-basic-cpu``
    """
    current_cluster = get_cluster_or_local(cluster_name=cluster_name)

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

    print_status(cluster_status, current_cluster)


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


@cluster_app.command("keep-warm")
def cluster_keep_warm(
    cluster_name: str = typer.Argument(
        None,
        help="Name of cluster to keep warm. If not specified will check the local cluster.",
    ),
    mins: Optional[int] = typer.Option(
        -1,
        help="Amount of time (in min) to keep the cluster warm after inactivity. "
        "If set to -1, keeps cluster warm indefinitely.",
    ),
):
    """Keep the cluster warm for given number of minutes after inactivity.

    Example:
        ``$ runhouse cluster keep-warm rh-basic-cpu``

    """
    current_cluster = get_cluster_or_local(cluster_name=cluster_name)

    try:
        if not current_cluster.is_up():
            print_bring_cluster_up_msg(cluster_name=cluster_name)
            return
        current_cluster.keep_warm(mins=mins)

    except ValueError:
        console.print(f"{cluster_name} is not saved in Den.")
        sky_live_clusters = get_all_sky_clusters()
        cluster_name = (
            cluster_name.split("/")[-1] if "/" in cluster_name else cluster_name
        )
        if cluster_name in sky_live_clusters:
            console.print(
                f"You can keep warm the cluster by running [italic bold] `sky autostop {cluster_name} -i {mins}`"
            )

    except Exception as e:
        console.print(f"Failed to keep the cluster warm: {e}")


@cluster_app.command("up")
def cluster_up(
    cluster_name: str = typer.Argument(
        None,
        help="The name of cluster to bring up.",
    )
):
    """Bring up the cluster if it is not up. No-op if cluster is already up.
    This only applies to on-demand clusters, and has no effect on self-managed clusters.

    Note: To launch the cluster via Den, set `launcher: den` in your local `~/.rh/config.yaml`.

    Example:
        ``$ runhouse cluster up rh-basic-cpu``

    """
    try:
        current_cluster = rh.cluster(name=cluster_name, dryrun=True)
        if current_cluster.is_up():
            console.print("Cluster is already up.")
            return
        current_cluster.up()
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
            proceed = typer.prompt(f"Terminating {cluster_name}. Proceed? [Y/n]")
        elif remove_all:
            proceed = typer.prompt(
                "Terminating all running clusters saved in Den. Proceed? [Y/n]"
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
                    name=f'/{rns_client.username}/{running_cluster.get("Name")}',
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

    current_cluster = get_cluster_or_local(cluster_name=cluster_name)

    try:
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
    current_cluster = get_cluster_or_local(cluster_name=cluster_name)

    cluster_uri = rns_client.resource_uri(current_cluster.rns_address)

    resp = requests.get(
        f"{rns_client.api_server_url}/resource/{cluster_uri}/logs/file?minutes={since}",
        headers=rns_client.request_headers(),
    )

    if resp.status_code != 200:
        console.print("Failed to get cluster logs.")
        return

    logs_file_url = resp.json().get("data").get("logs_presigned_url")
    logs_file_content = requests.get(logs_file_url).content.decode("utf-8")
    console.print(f"[reset][cyan]{logs_file_content}")


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
    use_https=False,
    den_auth=False,
    ssl_keyfile=None,
    ssl_certfile=None,
    restart_proxy=False,
    use_caddy=False,
    domain=None,
    certs_address=None,
    api_server_url=None,
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

    conda_env_flag = f" --conda-env {conda_env}" if conda_env else ""
    if conda_env_flag:
        logger.info(f"Creating runtime env for conda env: {conda_env}")
        flags.append(conda_env_flag)

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
    use_https: bool = typer.Option(
        False, help="Start an HTTPS server with TLS verification."
    ),
    use_den_auth: bool = typer.Option(
        False, help="Whether to authenticate requests with a Runhouse token."
    ),
    ssl_keyfile: Optional[str] = typer.Option(
        None, help="Path to custom SSL key file to use for enabling HTTPS."
    ),
    ssl_certfile: Optional[str] = typer.Option(
        None, help="Path to custom SSL cert file to use for enabling HTTPS."
    ),
    restart_proxy: bool = typer.Option(
        False, help="Whether to reinstall server configs on the cluster."
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
        help="Public IP address of the server. Required for generating self-signed certs and enabling HTTPS.",
    ),
    api_server_url: str = typer.Option(
        default="https://api.run.house",
        help="URL of Runhouse Den",
    ),
    conda_env: str = typer.Option(
        None, help="Name of conda env where Runhouse server is started, if applicable."
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
        c = get_cluster_or_local(cluster_name=cluster_name)
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
        use_https=use_https,
        den_auth=use_den_auth,
        ssl_keyfile=ssl_keyfile,
        ssl_certfile=ssl_certfile,
        restart_proxy=restart_proxy,
        use_caddy=use_caddy,
        domain=domain,
        certs_address=certs_address,
        api_server_url=api_server_url,
        conda_env=conda_env,
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
    use_https: bool = typer.Option(
        False, help="Start an HTTPS server with TLS verification."
    ),
    use_den_auth: bool = typer.Option(
        False, help="Whether to authenticate requests with a Runhouse token."
    ),
    ssl_keyfile: Optional[str] = typer.Option(
        None, help="Path to custom SSL key file to use for enabling HTTPS."
    ),
    ssl_certfile: Optional[str] = typer.Option(
        None, help="Path to custom SSL cert file to use for enabling HTTPS."
    ),
    restart_proxy: bool = typer.Option(
        False, help="Whether to reinstall server configs on the cluster."
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
        help="Public IP address of the server. Required for generating self-signed certs and enabling HTTPS.",
    ),
    api_server_url: str = typer.Option(
        default="https://api.run.house",
        help="URL of Runhouse Den",
    ),
    conda_env: str = typer.Option(
        None, help="Name of conda env where Runhouse server is started, if applicable."
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
    if cluster_name:
        c = get_cluster_or_local(cluster_name=cluster_name)
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
        conda_env=conda_env,
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
    logger.debug("Stopping the server")

    if cluster_name:
        current_cluster = get_cluster_or_local(cluster_name=cluster_name)
        current_cluster.stop_server(stop_ray=stop_ray, cleanup_actors=cleanup_actors)
        return

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
    logger.debug("Checking the server status.")
    current_cluster = get_cluster_or_local(cluster_name=cluster_name)
    try:
        status = current_cluster.status()
        console.print(f"[reset]{BULLET_UNICODE} server pid: {status.get('server_pid')}")
        print_cluster_config(
            cluster_config=status.get("cluster_config"), status_type=StatusType.server
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
