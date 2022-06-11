"""This module provides the runhouse CLI."""
import os
import json
import logging
from pathlib import Path
from typing import Optional
import typer
from runhouse import __app_name__, __version__
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


def version_callback(value: bool) -> None:
    if value:
        typer.echo(f"{__app_name__}=={__version__}")
        raise typer.Exit()


def filename_callback(filepath: str) -> None:
    if filepath is None:
        return

    # Full path based on file system
    full_path = os.path.join(os.getcwd(), Path(__file__).parent, filepath)
    if not valid_filepath(full_path):
        typer.echo(f"invalid filepath provided: '{filepath}'")
        raise typer.Exit(code=1)

    # TODO this hardware should be dynamic (i.e. we need access to hardware param in this callback)
    hardware = os.getenv('DEFAULT_HARDWARE')

    # Copy the python script (and possible dependencies) to remote server and run it
    run_python_job_on_remote_server(full_path, hardware=hardware)
    typer.echo(f"Finished running job on remote server with hardware {hardware}")
    raise typer.Exit()


def register_callback(func_path: str) -> None:
    if func_path is None:
        return

    # TODO grab the source code for the function path provided
    code = """
        def bert_preprocessing():
        DEST_DIR = 'training_folder_bert'
        print("Starting model training")
        create_directory(DEST_DIR)
        time.sleep(5)
        print(f"Finished training - saved results to {DEST_DIR}")
    """
    # TODO implement redis here as the DB
    # TODO get actual user

    # TODO this hardware should be dynamic (i.e. we need access to hardware param in this callback)
    hardware = os.getenv('DEFAULT_HARDWARE')

    # TODO uri should be dynamic? user defined?
    uri = f"/{os.getenv('DEMO_USER')}/{os.getenv('DEMO_URI_NAME')}"

    # cache the provided uri, function contents, and specified hardware in redis
    db_api = DatabaseAPI(uri=uri)
    if db_api.key_exists_in_db():
        typer.echo(f'URI already exists for hardware {hardware}')
    else:
        typer.echo(f'Adding URI for hardware {hardware} and user {os.getenv("DEMO_USER")}')
        db_api.add_cached_uri_to_db(hardware=hardware, code=code)

    raise typer.Exit()


def hardware_callback(hardware: str) -> None:
    if not valid_hardware(hardware):
        typer.echo(f"invalid hardware specification {hardware}")
        typer.echo(f"Hardware options: {list(HARDWARE_TO_HOSTNAME)}")
        raise typer.Exit(code=1)

    open_bash_on_remote_server(hardware)
    raise typer.Exit()


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
            help="File path to run (ex: './training_script.py')",
            callback=filename_callback,
            is_eager=False,
        ),
        register: Optional[str] = typer.Option(
            None,
            "--register",
            "-r",
            help="Register a specific function to a URI",
            callback=register_callback,
            is_eager=False,
        )
) -> None:
    return
