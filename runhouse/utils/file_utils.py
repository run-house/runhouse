import shutil
import os
import typer
from runhouse.utils.utils import random_string_generator
from runhouse.utils.validation import valid_filepath, validate_name


def create_directory(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def create_directories(dir_names):
    for dir_name in dir_names:
        create_directory(dir_name)


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
        pass


def copy_file_to_directory(source_dir, dest_dir):
    shutil.copyfile(source_dir, dest_dir)


def create_name_for_folder(name):
    if name is None:
        # if user did not provide a names we'll make one up
        name = random_string_generator().lower()
        typer.echo(f'Creating URI with name: {name}')
        return name

    # make sure the user provided name is in line with runhouse conventions
    validate_name(name)
    return name.lower()
