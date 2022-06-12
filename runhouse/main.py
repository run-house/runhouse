"""This module provides the runhouse CLI."""
import os
import json
import logging
from dataclasses import dataclass
import inspect
import importlib
from typing import Optional

import pkg_resources
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


def validate_hardware(hardware):
    if hardware not in list(HARDWARE_TO_HOSTNAME):
        typer.echo(f"invalid hardware specification")
        typer.echo(f"Please choose from the following options: {list(HARDWARE_TO_HOSTNAME)}")
        raise typer.Exit(code=1)


def convert_path_to_module(file_path):
    """Convert path to file received by CLI arg to an importable module"""
    return file_path.replace('/', '.').rsplit('.', 1)[0]


def get_hostname_from_hardware(hardware):
    hostname = HARDWARE_TO_HOSTNAME.get(hardware)

    if hostname is not None:
        return hostname

    typer.echo(f"host name not found for hardware {hardware}")
    typer.Exit(code=1)


def open_bash_on_remote_server(hardware):
    host = get_hostname_from_hardware(hardware)
    typer.echo(f"Opening shell with {hardware}\n")
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
    filepath: str


@app.command()
def run(ctx: typer.Context):
    """Run file provided on specified hardware"""
    # Full path based on file system
    file_path = ctx.obj.filepath
    if not valid_filepath(file_path):
        typer.echo(f"invalid filepath provided: '{file_path}'")
        raise typer.Exit(code=1)

    hardware = ctx.obj.hardware or os.getenv('DEFAULT_HARDWARE')
    typer.echo(f'Running job on server with hardware: {hardware}')
    # Copy the python script (and possible dependencies) to remote server and run it
    run_python_job_on_remote_server(file_path, hardware=hardware)
    typer.echo(f"Finished running job on server")
    raise typer.Exit()


@app.command()
def shell(ctx: typer.Context) -> None:
    """Open a bash shell on remote server with provided hardware"""
    hardware = ctx.obj.hardware
    validate_hardware(hardware)
    open_bash_on_remote_server(hardware)
    raise typer.Exit()


@app.command()
def register(ctx: typer.Context,
             user: str = typer.Option(os.getenv('DEMO_USER'), '--user', '-u', help='user who is registering the URI'),
             name: str = typer.Option(os.getenv('DEMO_URI_NAME'), '--name', '-n', help='name of the URI')) -> None:
    """Register python function as URI"""
    hardware = ctx.obj.hardware
    validate_hardware(hardware)

    file_path = ctx.obj.filepath
    if file_path is None or not valid_filepath(file_path):
        typer.echo("Must provide a valid file path to the code to register")
        raise typer.Exit(code=1)

    # Get the code provided
    module_path = convert_path_to_module(file_path)
    m = importlib.import_module(module_path)
    code = inspect.getsource(m)

    # TODO uri should be dynamic? user defined?
    uri = f"/{user}/{name}"

    # cache the provided uri, function contents, and specified hardware in redis
    # TODO implement redis here as the DB
    db_api = DatabaseAPI(uri=uri)
    if db_api.key_exists_in_db():
        typer.echo(f'URI already exists for hardware {hardware}')
    else:
        typer.echo(f'Adding URI for hardware {hardware} and user {user}')
        db_api.add_cached_uri_to_db(hardware=hardware, code=code)

    raise typer.Exit()


def version_callback(value: bool) -> None:
    if value is not None:
        name = 'runhouse'
        version = pkg_resources.get_distribution(name).version
        typer.echo(f"{name}=={version}")
        raise typer.Exit()


@app.callback()
def common(ctx: typer.Context,
           hardware: str = typer.Option(os.getenv('DEFAULT_HARDWARE'), '--hardware', '-h', help='hardware'),
           filepath: str = typer.Option(None, '--path', '-p',
                                        help='Path to file to run on specified hardware or to register as URI'),
           version: Optional[bool] = typer.Option(None, '--version', '-v', callback=version_callback,
                                                  help='current package version')):
    """Welcome to Runhouse! Here's what you need to get started"""
    ctx.obj = Common(hardware, filepath)
