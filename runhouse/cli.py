"""This module provides the runhouse CLI."""
import os
import json
import logging
from pathlib import Path
from typing import Optional
import typer
from runhouse import __app_name__, __version__
from runhouse.shell_handler import ShellHandler
from runhouse.ssh_manager import SSHManager

from dotenv import load_dotenv

# For now load from .env
from runhouse.user_commands import cmd_commands
from runhouse.utils import save_to_file, read_file, valid_filepath

load_dotenv()

# # creates an explicit Typer application, app
app = typer.Typer(add_completion=False)

logger = logging.getLogger(__name__)

# map each hardware option to its IP / host
HARDWARE_TO_HOSTNAME = json.loads(os.getenv('HARDWARE_TO_HOSTNAME'))


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

    # TODO this hardware should be dynamic (i.e. we need access to hardware param in this callback)
    hardware = os.getenv('DEFAULT_HARDWARE')

    # Copy the python script (and possible dependencies) to remote server and run it
    run_python_job_on_remote_server(filepath, hardware=hardware)

    raise typer.Exit()


def hardware_callback(hardware: str) -> None:
    if not valid_hardware(hardware):
        typer.echo(f"invalid hardware specification {hardware}")
        typer.echo(f"Hardware options: {list(HARDWARE_TO_HOSTNAME)}")
        raise typer.Exit()

    open_bash_on_remote_server(hardware)


def valid_hardware(hardware) -> bool:
    return hardware in list(HARDWARE_TO_HOSTNAME)


def get_hostname_from_hardware(hardware):
    hostname = HARDWARE_TO_HOSTNAME.get(hardware)

    if hostname is not None:
        return hostname

    typer.echo(f"host name not found for hardware {hardware}")
    typer.Exit()


def open_bash_on_remote_server(hardware):
    host = get_hostname_from_hardware(hardware)
    typer.echo(f"Opening shell on remote resource with {hardware} on host {host}")
    sh = ShellHandler(host=host)
    cmd_commands(sh)


def run_python_job_on_remote_server(filepath, hardware):
    try:
        # Connect/ssh to an instance
        hostname = get_hostname_from_hardware(hardware)
        sm = SSHManager(hostname=hostname)
        ftp_client = sm.create_ftp_client()

        # TODO identify if we need to also copy any directories (in addition to the file)
        #  without the user having to explicitly define them
        copy_dir = False
        if copy_dir:
            # TODO Here we arbitrarily copy the contents of the parent's parent directory
            path = Path(filepath)
            source_dir = path.parent.parent
            sm.put_dir(ftp_client=ftp_client, source_dir=source_dir)
        else:
            sm.copy_file_to_remote_server(ftp_client=ftp_client, filepath=filepath)

        # open the module locally as a text file
        txt_file = read_file(filepath)

        # file path to the python bin on remote server
        remote_python_dir = os.getenv('REMOTE_PYTHON_PATH')

        # Execute the file after connecting/ssh to the instance
        stdin, stdout, stderr = sm.client.exec_command(
            "{p}python3 - <<EOF\n{s}\nEOF".format(p=remote_python_dir,
                                                  s=txt_file))
        stdout = stdout.readlines()

        # TODO incorporate some logging / optionality to save them locally
        save_to_file(stdout)

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
            os.getenv('DEFAULT_HARDWARE'),
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
