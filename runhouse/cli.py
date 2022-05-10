"""This module provides the runhouse CLI."""
from typing import Optional
import typer

from runhouse import __app_name__, __version__

# creates an explicit Typer application, app
app = typer.Typer()


def _version_callback() -> None:
    typer.echo(f"{__app_name__}=={__version__}")
    raise typer.Exit()


def _valid_hardware(hardware) -> bool:
    valid_hardware = {'rh_1_gpu', 'rh_8_gpu', 'rh_16_gpu'}
    return hardware in valid_hardware


def _hardware_callback(hardware: str) -> None:
    if not _valid_hardware(hardware):
        typer.echo("invalid hardware specification")
        raise typer.Exit()

    typer.echo(f"initializing {hardware} on remote resources")
    # TODO launch the remote python shell
    from time import sleep
    sleep(5)
    raise typer.Exit()


@app.callback()
def main(
        version: Optional[bool] = typer.Option(
            None,
            "--version",
            "-v",
            help="Show the application's version",
            callback=_version_callback,
            is_eager=True,
        ),
        hardware: Optional[str] = typer.Option(
            None,
            "--hardware",
            "-h",
            help="Specify which hardware to run this job",
            callback=_hardware_callback,
            is_eager=True,
        )
) -> None:
    return

