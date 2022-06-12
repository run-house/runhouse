"""This module provides the runhouse CLI."""
import os
import json
import logging
from dataclasses import dataclass
from typing import Optional

import pkg_resources
from pathlib import Path
import typer
from runhouse.redis.db_api import DatabaseAPI
from runhouse.shell_handler import ShellHandler
from runhouse.ssh_manager import SSHManager
from runhouse.process_commands import process_cmd_commands
from runhouse.utils.utils import save_to_file, read_file, valid_filepath

# For now load from .env
from dotenv import load_dotenv
load_dotenv()

# # creates an explicit Typer application, app
app = typer.Typer(add_completion=False)

logger = logging.getLogger(__name__)

# map each hardware option to its IP / host
HARDWARE_TO_HOSTNAME = json.loads(os.getenv('HARDWARE_TO_HOSTNAME'))


def valid_hardware(hardware) -> bool:
    return hardware in list(HARDWARE_TO_HOSTNAME)


def get_hostname_from_hardware(hardware):
    hostname = HARDWARE_TO_HOSTNAME.get(hardware)

    if hostname is not None:
        return hostname

    typer.echo(f"host name not found for hardware {hardware}")
    typer.Exit(code=1)


def open_bash_on_remote_server(hardware):
    host = get_hostname_from_hardware(hardware)
    typer.echo(f"Opening shell with {hardware} on host {host}\n")
    sh = ShellHandler(host=host)
    process_cmd_commands(sh)


def run_python_job_on_remote_server(filepath, hardware):
    try:
        # Connect/ssh to an instance
        hostname = get_hostname_from_hardware(hardware)
        sm = SSHManager(hostname=hostname)
        ftp_client = sm.create_ftp_client()

        sm.copy_file_to_remote_server(ftp_client=ftp_client, filepath=filepath)

        # open the module locally as a text file
        txt_file = read_file(filepath)

        # file path to the python bin on remote server
        remote_python_dir = os.getenv('REMOTE_PYTHON_PATH')

        # Execute the file after connecting/ssh to the instance
        stdin, stdout, stderr = sm.client.exec_command("{p}python3 - <<EOF\n{s}\nEOF".format(p=remote_python_dir,
                                                                                             s=txt_file))
        stdout = stdout.readlines()

        # TODO incorporate some logging / optionality to save them locally
        save_to_file(stdout)

        stdin.close()
        # close the client connection once the job is done
        sm.client.close()

    except Exception as e:
        logger.error(f'Unable to run python job on remote server: {e}')


@dataclass
class Common:
    hardware: str
    code: str
    filename: str


@app.command()
def run(ctx: typer.Context):
    """Run file based on path and hardware provided"""
    # Full path based on file system
    file_name = ctx.obj.filename
    full_path = os.path.join(os.getcwd(), Path(__file__).parent, file_name)
    if not valid_filepath(full_path):
        typer.echo(f"invalid filepath provided: '{file_name}'")
        raise typer.Exit(code=1)

    hardware = ctx.obj.hardware
    # Copy the python script (and possible dependencies) to remote server and run it
    run_python_job_on_remote_server(full_path, hardware=hardware)
    typer.echo(f"Finished running job on remote server with hardware {hardware}")
    raise typer.Exit()


@app.command()
def shell(ctx: typer.Context) -> None:
    """Open a bash shell on remote server with provided hardware"""
    hardware = ctx.obj.hardware
    if not valid_hardware(hardware):
        typer.echo(f"invalid hardware specification {hardware}")
        typer.echo(f"Hardware options: {list(HARDWARE_TO_HOSTNAME)}")
        raise typer.Exit(code=1)

    open_bash_on_remote_server(hardware)
    raise typer.Exit()


@app.command()
def register(ctx: typer.Context, user: str = typer.Option(None, '--user', '-u', help='user who is registering the URI'),
             name: str = typer.Option(None, '--name', '-n', help='name of the URI')) -> None:
    """Register python function as URI"""

    # TODO grab the source code for the function path provided
    code = """
           def bert_preprocessing():
           DEST_DIR = 'training_folder_bert'
           print("Starting model training")
           create_directory(DEST_DIR)
           time.sleep(5)
           print(f"Finished training - saved results to {DEST_DIR}")
       """
    hardware = ctx.obj.hardware

    registered_user = user or os.getenv('DEMO_USER')
    registered_name = name or os.getenv('DEMO_URI_NAME')
    uri = f"/{registered_user}/{registered_name}"

    # cache the provided uri, function contents, and specified hardware in redis
    # TODO implement redis here as the DB
    db_api = DatabaseAPI(uri=uri)
    if db_api.key_exists_in_db():
        typer.echo(f'URI already exists for hardware {hardware} and user {registered_user} and name {registered_name}')
    else:
        typer.echo(f'Adding URI for hardware {hardware} and user {registered_user} and name {registered_name}')
        db_api.add_cached_uri_to_db(hardware=hardware, code=code)

    raise typer.Exit()


def version_callback(value: bool) -> None:
    if value:
        name = 'runhouse'
        version = pkg_resources.get_distribution(name).version
        typer.echo(f"{name}=={version}")
        raise typer.Exit()


@app.callback()
def common(ctx: typer.Context,
           hardware: str = typer.Option(os.getenv('DEFAULT_HARDWARE'), '--hardware', '-h', help='hardware'),
           code: str = typer.Option(None, '--code', '-c', help='register code to URI'),
           filename: str = typer.Option(None, '--filename', '-f', help='run a file on remote server'),
           version: Optional[bool] = typer.Option(None, '--version', '-v', callback=version_callback,
                                                  help='current package version')):
    """Common Entry Point"""
    ctx.obj = Common(hardware, code, filename)
