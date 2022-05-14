"""This module provides the runhouse CLI."""
import os
import json
import logging
from typing import Optional
import typer
from runhouse import __app_name__, __version__
from runhouse.shell_handler import ShellHandler
from runhouse.ssh_manager import SSHManager

from dotenv import load_dotenv

# For now load from .env
load_dotenv()

# # creates an explicit Typer application, app
app = typer.Typer(add_completion=False)

logger = logging.getLogger(__name__)


def version_callback(value: bool) -> None:
    if value:
        typer.echo(f"{__app_name__}=={__version__}")
        raise typer.Exit()


def filename_callback(filepath) -> None:
    if filepath is None:
        return

    if not valid_filepath(filepath):
        typer.echo(f"invalid filepath provided: '{filepath}'")
        raise typer.Exit()

    # TODO this hardware should be dynamic
    # Copy the python script to remote server and run it
    run_python_job_on_remote_server(filepath, hardware=os.getenv('DEFAULT_HARDWARE'))

    raise typer.Exit()


def hardware_callback(hardware: str) -> None:
    if not valid_hardware(hardware):
        typer.echo(f"invalid hardware specification {hardware}")
        raise typer.Exit()

    open_bash_on_remote_server(hardware)


def valid_filepath(filepath) -> bool:
    return os.path.exists(filepath)


def hardware_to_hostname() -> dict:
    return json.loads(os.getenv('HARDWARE_TO_HOSTNAME'))


def valid_hardware(hardware) -> bool:
    return hardware in list(hardware_to_hostname())


def get_hostname_from_hardware(hardware):
    hostname = hardware_to_hostname().get(hardware)

    if hostname is not None:
        return hostname

    typer.echo(f"host name not found for hardware {hardware}")
    typer.Exit()


def open_bash_on_remote_server(hardware):
    typer.echo(f"Opening shell on remote resource with {hardware}")
    host = get_hostname_from_hardware(hardware)
    sh = ShellHandler(host=host)
    # TODO execute commands based on user input


def run_python_job_on_remote_server(filepath, hardware):
    try:
        # Connect/ssh to an instance
        hostname = get_hostname_from_hardware(hardware)
        sm = SSHManager(hostname=hostname)
        sm.copy_file_to_remote_server(filepath)
        # Execute a cmd after connecting/ssh to the instance
        cmd = f'python {filepath}'
        stdin, stdout, stderr = sm.client.exec_command(cmd)
        stdin.close()
        # close the client connection once the job is done
        sm.client.close()

    except Exception as e:
        logger.error(f'Unable to run python job on remote server: {e}')


@app.callback()
def main(
        version: Optional[bool] = typer.Option(
            None,
            "--version",
            "-v",
            help="Show the application's version",
            callback=version_callback,
            is_eager=False,
        ),
        hardware: Optional[str] = typer.Option(
            None,
            "--hardware",
            "-h",
            help="Hardware used to run this job (ex: 'rh_1_gpu')",
            callback=hardware_callback,
            envvar=os.getenv('DEFAULT_HARDWARE'),
            is_eager=False
        ),
        filename: Optional[str] = typer.Option(
            None,
            "--filename",
            "-f",
            help="Python file to run (ex: 'training_script')",
            callback=filename_callback,
            is_eager=False,
        )
) -> None:
    return
