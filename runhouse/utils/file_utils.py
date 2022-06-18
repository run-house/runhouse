import shutil
import os
import typer
from runhouse.utils.utils import random_string_generator
from runhouse.utils.validation import valid_filepath, validate_name

RUNNABLE_FILE_NAME = 'run'


def create_directory(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def read_file(filepath):
    if not valid_filepath(filepath):
        raise Exception(f'File not found: {filepath}')

    with open(filepath, "r") as f:
        data = f.read()
    return data


def delete_directory(directory):
    try:
        shutil.rmtree(directory)
    except OSError as e:
        print("Error: %s - %s." % (e.filename, e.strerror))


def copy_runnable_file_to_runhouse_subdir(path_to_runnable_file, name_dir, ext):
    """Runhouse will be executing the runnable file from its internal subdirectory"""
    shutil.copyfile(path_to_runnable_file, os.path.join(name_dir, f'{RUNNABLE_FILE_NAME}{ext}'))


def create_name_for_folder(name):
    if name is None:
        # if user did not provide a names we'll make one up
        name = random_string_generator().lower()
        typer.echo(f'Creating URI with name: {name}')
        return name

    # make sure the user provided name is in line with runhouse conventions
    validate_name(name)
    return name.lower()
