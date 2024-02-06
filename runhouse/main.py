import logging
import shlex
import subprocess
import time
import webbrowser
from pathlib import Path
from typing import List, Optional

import requests

import typer
from rich.console import Console

import runhouse as rh

import runhouse.rns.login

from runhouse import __version__, cluster, configs
from runhouse.constants import (
    RAY_KILL_CMD,
    RAY_START_CMD,
    SERVER_LOGFILE,
    SERVER_START_CMD,
    SERVER_STOP_CMD,
    START_NOHUP_CMD,
    START_SCREEN_CMD,
)
from runhouse.resources.hardware.ray_utils import check_for_existing_ray_instance


# create an explicit Typer application
app = typer.Typer(add_completion=False)
state = {"verbose": False}

# For printing with typer
console = Console()
logger = logging.getLogger(__name__)


@app.command()
def login(
    token: Optional[str] = typer.Argument(None, help="Your Runhouse API token"),
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
            token=token, interactive=True, ret_token=True, from_cli=True
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
            f"Cluster {cluster_name} is not up. Please run `runhouse notebook {cluster_name} --up`."
        )
        raise typer.Exit(1)
    c.notebook()


@app.command()
def ssh(cluster_name: str, up: bool = typer.Option(False, help="Start the cluster")):
    """SSH into a cluster created elsewhere (so `ssh cluster` doesn't work out of the box) or not yet up."""
    c = cluster(name=cluster_name)
    if up:
        c.up_if_not()
    if not c.is_up():
        console.print(
            f"Cluster {cluster_name} is not up. Please run `runhouse ssh {cluster_name} --up`."
        )
        raise typer.Exit(1)
    c.ssh()


def _print_status(config):
    """
    Prints the status of the cluster to the console
    :param config: cluster's  config
    :return: cluster's  config
    """
    envs = config["envs"]
    config.pop("envs", [])

    # print headlines
    daemon_headline_txt = (
        "\N{smiling face with horns} Runhouse Daemon is running \N{Runner}"
    )

    console.print(daemon_headline_txt, style="bold royal_blue1")
    if "name" in config.keys():
        console.print(config["name"])

    first_info_to_print = ["den_auth", "server_connection_type", "server_port"]

    for info in config:
        if info in first_info_to_print:
            console.print(f"\u2022 {info}: {config[info]}")
    first_info_to_print.append("name")

    console.print("\u2022 backend config:")
    for info in config:
        if info not in first_info_to_print:
            console.print(f"\t\u2022 {info}: {config[info]}")

    # print the environments in the cluster, and the resources associated with each  environment.
    envs_in_cluster_headline = "Serving 🍦 :"
    console.print(envs_in_cluster_headline, style="bold")

    if len(envs) == 0:
        console.print("This cluster has no environment nor resources.")

    for env_name in envs:
        resources_in_env = envs[env_name]
        if len(resources_in_env) == 0:
            console.print(f"{env_name} (Env):", style="italic underline")
            console.print("This environment has no resources.")

        else:
            current_env = [
                resource
                for resource in resources_in_env
                if resource["name"] == env_name
            ]

            # sometimes the env itself is not a resource (key) inside the env's servlet.
            if len(current_env) == 0:
                env_name_txt = f"{env_name} (Env):"
            else:
                current_env = current_env[0]
                env_name_txt = (
                    f"{current_env['name']} ({current_env['resource_type']}):"
                )

            console.print(
                env_name_txt,
                style="italic underline",
            )

            resources_in_env = [
                resource for resource in resources_in_env if resource is not current_env
            ]

            for resource in resources_in_env:
                resource_name = resource["name"]
                resource_type = resource["resource_type"]
                console.print(f"\u2022{resource_name} ({resource_type})")

    return config


@app.command()
def status(
    cluster_name: str = typer.Argument(
        None,
        help="Name of cluster to check. If not specified will check"
        " the local cluster.",
    )
):
    """Get the config info about your runhouse cluster(s), as well as the environments setup of the cluster(s)"""

    cluster_or_local = rh.here

    if cluster_or_local == "file":
        if cluster_name is None:
            console.print("Missing argument CLUSTER_NAME.")
            return
        else:
            try:
                current_cluster = rh.cluster(name=cluster_name)
                config = current_cluster.status()
            except ValueError:
                console.print(
                    f"Cluster {cluster_name} is not found in Den. Please save it, in order to get its"
                    f" status"
                )
                return
            except requests.exceptions.ConnectionError:
                console.print(
                    "\N{smiling face with horns} Runhouse Daemon is not running... \N{No Entry} \N{Runner}"
                )
                return
    else:
        from runhouse.globals import obj_store

        config = obj_store.get_status()

    config = _print_status(config)

    return config


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
    use_nginx=False,
    certs_address=None,
    use_local_telemetry=False,
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
        logger.info("Reinstalling Nginx and server configs.")
        flags.append(restart_proxy_flag)

    use_nginx_flag = " --use-nginx" if use_nginx else ""
    if use_nginx_flag:
        logger.info("Configuring Nginx on the cluster.")
        flags.append(use_nginx_flag)

    ssl_keyfile_flag = f" --ssl-keyfile {ssl_keyfile}" if ssl_keyfile else ""
    if ssl_keyfile_flag:
        logger.info(f"Using SSL keyfile in path: {ssl_keyfile}")
        flags.append(ssl_keyfile_flag)

    ssl_certfile_flag = f" --ssl-certfile {ssl_certfile}" if ssl_certfile else ""
    if ssl_certfile_flag:
        logger.info(f"Using SSL certfile in path: {ssl_certfile}")
        flags.append(ssl_certfile_flag)

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

    use_local_telemetry_flag = " --use-local-telemetry" if use_local_telemetry else ""
    if use_local_telemetry_flag:
        logger.info("Configuring local telemetry on the cluster.")
        flags.append(use_local_telemetry_flag)

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
        if screen and Path(SERVER_LOGFILE).exists():
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
            # We don't want to raise an error if the server kill fails, as it may simply not be running
            if result.returncode != 0 and "pkill" not in cmd:
                console.print(f"Error while executing `{cmd}`")
                raise typer.Exit(1)

        server_started_str = "Uvicorn running on"
        # Read and print the server logs until the
        if screen:
            while not Path(SERVER_LOGFILE).exists():
                time.sleep(1)
            f = f or open(SERVER_LOGFILE, "r")
            start_time = time.time()
            # Wait for input for 60 seconds max (for nginx to download and set up)
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
    restart_ray: bool = typer.Option(False, help="Restart the Ray runtime"),
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
    use_nginx: bool = typer.Option(
        False,
        help="Whether to configure Nginx on the cluster as a reverse proxy. By default will not install "
        "and configure Nginx.",
    ),
    certs_address: Optional[str] = typer.Option(
        None,
        help="Public IP address of the server. Required for generating self-signed certs and enabling HTTPS",
    ),
    use_local_telemetry: bool = typer.Option(
        False, help="Whether to use local telemetry"
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
        use_nginx=use_nginx,
        certs_address=certs_address,
        use_local_telemetry=use_local_telemetry,
    )


@app.command()
def restart(
    name: str = typer.Option(None, help="A *saved* remote cluster object to restart."),
    restart_ray: bool = typer.Option(False, help="Restart the Ray runtime"),
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
        False, help="Whether to reinstall Nginx and other server configs on the cluster"
    ),
    use_nginx: bool = typer.Option(
        False,
        help="Whether to configure Nginx on the cluster as a reverse proxy. By default will not install "
        "and configure Nginx.",
    ),
    certs_address: Optional[str] = typer.Option(
        None,
        help="Public IP address of the server. Required for generating self-signed certs and enabling HTTPS",
    ),
    use_local_telemetry: bool = typer.Option(
        False,
        help="Whether to use local telemetry",
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
        use_nginx=use_nginx,
        certs_address=certs_address,
        use_local_telemetry=use_local_telemetry,
    )


@app.command()
def stop(stop_ray: bool = typer.Option(False, help="Stop the Ray runtime")):
    subprocess.run(SERVER_STOP_CMD, shell=True)

    if stop_ray:
        subprocess.run(RAY_KILL_CMD, shell=True)


@app.callback()
def main(verbose: bool = False):
    """
    Runhouse CLI
    """
    if verbose:
        name = "runhouse"
        console.print(f"{name}=={__version__}", style="bold green")
        state["verbose"] = True
