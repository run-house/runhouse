import shlex
import subprocess
import time
import webbrowser
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

import runhouse.rns.login

from runhouse import __version__, cluster, configs

# create an explicit Typer application
app = typer.Typer(add_completion=False)
state = {"verbose": False}

# For printing with typer
console = Console()


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


def load_cluster(cluster_name: str):
    """Load a cluster from RNS into the local environment, e.g. to be able to ssh."""
    c = cluster(name=cluster_name)
    if not c.address:
        c._update_from_sky_status(dryrun=True)


def _start_server(
    restart,
    restart_ray,
    screen,
    create_logfile=True,
    host=None,
    port=None,
    use_https=False,
    den_auth=False,
    ssl_keyfile=None,
    ssl_certfile=None,
    force_reinstall=False,
    use_nginx=False,
    certs_address=None,
    use_local_telemetry=False,
):
    from runhouse.resources.hardware.cluster import Cluster

    cmds = Cluster._start_server_cmds(
        restart=restart,
        restart_ray=restart_ray,
        screen=screen,
        create_logfile=create_logfile,
        host=host,
        port=port,
        use_https=use_https,
        den_auth=den_auth,
        ssl_keyfile=ssl_keyfile,
        ssl_certfile=ssl_certfile,
        force_reinstall=force_reinstall,
        use_nginx=use_nginx,
        certs_address=certs_address,
        use_local_telemetry=use_local_telemetry,
    )

    try:
        # Open and read the lines of the server logfile so we only print the most recent lines after starting
        f = None
        if screen and Path(Cluster.SERVER_LOGFILE).exists():
            f = open(Cluster.SERVER_LOGFILE, "r")
            f.readlines()  # Discard these, they're from the previous times the server was started

        # We do these one by one so it's more obvious where the error is if there is one
        for cmd in cmds:
            console.print(f"Executing `{cmd}`")
            result = subprocess.run(shlex.split(cmd), text=True)
            # We don't want to raise an error if the server kill fails, as it may simply not be running
            if result.returncode != 0 and "pkill" not in cmd:
                console.print(f"Error while executing `{cmd}`")
                raise typer.Exit(1)

        server_started_str = "Uvicorn running on"
        # Read and print the server logs until the
        if screen:
            while not Path(Cluster.SERVER_LOGFILE).exists():
                time.sleep(1)
            f = f or open(Cluster.SERVER_LOGFILE, "r")
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
    restart_ray: bool = typer.Option(True, help="Restart the Ray runtime"),
    screen: bool = typer.Option(
        True,
        help="Start the server in a screen. Only relevant when restarting locally.",
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
    force_reinstall: bool = typer.Option(
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
        create_logfile=True,
        host=host,
        port=port,
        use_https=use_https,
        den_auth=use_den_auth,
        ssl_keyfile=ssl_keyfile,
        ssl_certfile=ssl_certfile,
        force_reinstall=force_reinstall,
        use_nginx=use_nginx,
        certs_address=certs_address,
        use_local_telemetry=use_local_telemetry,
    )


@app.callback()
def main(verbose: bool = False):
    """
    Runhouse CLI
    """
    if verbose:
        name = "runhouse"
        console.print(f"{name}=={__version__}", style="bold green")
        state["verbose"] = True
