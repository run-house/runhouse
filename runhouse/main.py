import subprocess
from typing import Optional

import pkg_resources
import typer
import webbrowser
from rich.console import Console

from runhouse import configs
from runhouse.rns import login as login_module  # Need to rename it because it conflicts with the login command
from runhouse import Cluster

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
    valid_token: str = login_module.login(token=token,
                                          interactive=True,
                                          ret_token=True)
    if valid_token:
        webbrowser.open(f"{configs.get('api_server_url')}/dashboard?token={valid_token}")
        raise typer.Exit()
    else:
        raise typer.Exit(code=1)


@app.command()
def notebook(cluster: str, up: Optional[bool] = typer.Option(False, help="Start the cluster")):
    """Open a Jupyter notebook on a cluster.
    """
    c = Cluster.from_name(cluster)
    if up:
        c.up_if_not()
    if not c.is_up():
        console.print(f"Cluster {cluster} is not up. Please run `runhouse notebook {cluster} --up`.")
        raise typer.Exit(1)
    c.notebook()


@app.command()
def ssh(cluster: str, up: Optional[bool] = typer.Option(False, help="Start the cluster")):
    """SSH into a cluster created elsewhere (so `ssh cluster` doesn't work out of the box) or not yet up. """
    c = Cluster.from_name(cluster)
    if up:
        c.up_if_not()
    if not c.is_up():
        console.print(f"Cluster {cluster} is not up. Please run `runhouse ssh {cluster} --up`.")
        raise typer.Exit(1)
    subprocess.call(f"ssh {c.name}", shell=True)


def load_cluster(cluster: str):
    """Load a cluster from RNS into the local environment, e.g. to be able to ssh.
    """
    c = Cluster.from_name(cluster)
    if not c.address:
        c.populate_vars_from_status(dryrun=True)


@app.callback()
def main(verbose: bool = False):
    """
    Runhouse CLI app. Currently we support login, but more on the way :)
    """
    if verbose:
        name = 'runhouse'
        version = pkg_resources.get_distribution(name).version
        console.print(f"{name}=={version}", style="bold green")
        state["verbose"] = True
