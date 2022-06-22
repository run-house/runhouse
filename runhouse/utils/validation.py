import json
from pathlib import Path
import typer
import os
from runhouse.utils.utils import ERROR_FLAG

MAX_DIR_LEN = 12


def valid_filepath(filepath) -> bool:
    if filepath is None:
        return False
    return os.path.exists(filepath)


def validate_name(name):
    # TODO other checks on the directory name we want to add?
    if len(name) > MAX_DIR_LEN:
        typer.echo(f'{ERROR_FLAG} Runhouse does not support a name longer than ({MAX_DIR_LEN})')
        raise typer.Exit(code=1)


def validate_runnable_file_path(path_to_runnable_file):
    if not path_to_runnable_file:
        # If we did not explicitly receive the path to the file (-f) by the user (and not provided in the config file)
        typer.echo(f'{ERROR_FLAG} Please include the path to the file to run (using -f option)')
        raise typer.Exit(code=1)

    if not valid_filepath(path_to_runnable_file):
        # make sure the path the user provided is ok
        typer.echo(f'{ERROR_FLAG} No file found in path: {path_to_runnable_file}')
        raise typer.Exit(code=1)


def validate_hardware(hardware):
    """Throw an error for invalid hardware specs"""
    hardware_to_hostname = json.loads(os.getenv('HARDWARE_TO_HOSTNAME'))
    if hardware not in list(hardware_to_hostname):
        typer.echo(f"{ERROR_FLAG} Invalid hardware specification")
        typer.echo(f"Please choose from the following options: {list(hardware_to_hostname)}")
        raise typer.Exit(code=1)


def validate_pem_file():
    path_to_pem = os.getenv('PATH_TO_PEM')
    if path_to_pem is None:
        resp = typer.prompt('Please specify path to your runhouse pem file (or add as env variable "PATH_TO_PEM")')
        path_to_pem = Path(resp)
        if not valid_filepath(path_to_pem):
            typer.echo('Invalid file path to pem')
            raise typer.Exit(code=1)
    return path_to_pem