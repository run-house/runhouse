import json
from pathlib import Path
import typer
import os
from runhouse.rns.api_utils.utils import ERROR_FLAG

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


def validate_pem_file():
    path_to_pem = os.getenv('PATH_TO_PEM')
    if path_to_pem is None:
        resp = typer.prompt('Please specify path to your runhouse pem file (or add as env variable "PATH_TO_PEM")')
        path_to_pem = Path(resp)
        if not valid_filepath(path_to_pem):
            typer.echo('Invalid file path to pem')
            raise typer.Exit(code=1)
    return path_to_pem


def is_jsonable(myjson):
    try:
        json.dumps(myjson)
    except ValueError as e:
        return False
    return True
