import subprocess
import webbrowser
from typing import Optional

import pkg_resources
import typer
from rich.console import Console

from runhouse import cluster, configs
from runhouse.rns import (  # Need to rename it because it conflicts with the login command
    login as login_module,
)

# create an explicit Typer application
app = typer.Typer(add_completion=False)
state = {"verbose": False}

# For printing with typer
console = Console()


@app.command()
def login(token: Optional[str] = typer.Argument(None, help="Your Runhouse API token")):
    """Login to Runhouse. Validates token provided, with options to upload or download stored secrets or config between
    local environment and Runhouse / Vault.
    """
    valid_token: str = login_module.login(token=token, interactive=True, ret_token=True)
    if valid_token:
        webbrowser.open(
            f"{configs.get('api_server_url')}/dashboard?token={valid_token}"
        )
        raise typer.Exit()
    else:
        raise typer.Exit(code=1)


@app.command()
def logout():
    """Logout of Runhouse. Provides options to delete locally configured secrets and local Runhouse configs"""
    login_module.logout(interactive=True)
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
    subprocess.call(f"ssh {c.name}", shell=True)


@app.command()
def cancel(
    cluster_name: str,
    run_key: str,
    force: Optional[bool] = typer.Option(False, help="Force cancel"),
):
    """Cancel a run on a cluster."""
    c = cluster(name=cluster_name)
    c.cancel(run_key, force=force)


@app.command()
def logs(
    cluster_name: str,
    run_key: str,
    print_results: Optional[bool] = typer.Option(False, help="Print results"),
):
    """Get logs from a run on a cluster."""
    c = cluster(name=cluster_name)
    res = c.get(run_key, stream_logs=True)
    if print_results:
        console.print(res)


def load_cluster(cluster_name: str):
    """Load a cluster from RNS into the local environment, e.g. to be able to ssh."""
    c = cluster(name=cluster_name)
    if not c.address:
        c.update_from_sky_status(dryrun=True)


@app.command()
def restart_grpc(
    cluster_name: str,
    restart_ray: bool = typer.Option(False, help="Restart the Ray runtime"),
    resync_rh: bool = typer.Option(False, help="Resync the Runhouse package"),
):
    """Restart the gRPC server on a cluster."""
    c = cluster(name=cluster_name)
    c.restart_grpc_server(resync_rh=resync_rh, restart_ray=restart_ray)


@app.callback()
def main(verbose: bool = False):
    """
    Runhouse CLI
    """
    if verbose:
        name = "runhouse"
        version = pkg_resources.get_distribution(name).version
        console.print(f"{name}=={version}", style="bold green")
        state["verbose"] = True
