import typer
import os
from runhouse.utils.utils import ERROR_FLAG

MAX_DIR_LEN = 12


def valid_filepath(filepath) -> bool:
    return os.path.exists(filepath)


def validate_name(name):
    # TODO maybe add some other checks on the directory name?
    if len(name) > MAX_DIR_LEN:
        typer.echo(f'{ERROR_FLAG} Runhouse does not support a name longer than ({MAX_DIR_LEN})')
        raise typer.Exit(code=1)


def validate_runnable_file_path(path_to_runnable_file):
    if not path_to_runnable_file:
        # If we did not explicitly receive the path to the file (-f) by the user (or not provided in the config file)
        typer.echo(f'{ERROR_FLAG} Please include the path to the file to run (using -f option)')
        raise typer.Exit(code=1)

    if not valid_filepath(path_to_runnable_file):
        # make sure the path the user provided is ok
        typer.echo(f'{ERROR_FLAG} No file found in path: {path_to_runnable_file}')
        raise typer.Exit(code=1)


def validate_hardware(hardware, hardware_to_hostname):
    """Throw an error for invalid hardware specs"""
    if hardware not in list(hardware_to_hostname):
        typer.echo(f"{ERROR_FLAG} Invalid hardware specification")
        typer.echo(f"Please choose from the following options: {list(hardware_to_hostname)}")
        raise typer.Exit(code=1)


