import shutil
import os
import typer
from runhouse.utils.utils import random_string_generator
from runhouse.utils.validation import validate_name


def create_directory(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def create_directories(dir_names):
    for dir_name in dir_names:
        create_directory(dir_name)


def delete_directory(directory):
    try:
        shutil.rmtree(directory)
    except OSError:
        pass


def create_name_for_folder(name):
    if name is None:
        # if user did not provide a name we'll make one up (inspired by Moby Dick)
        name = random_string_generator().lower()
        typer.echo(f'Creating namespace for: {name}')
        return name

    # make sure the user provided name is in line with runhouse conventions
    validate_name(name)
    return name.lower()


def write_stdout_to_file(text, path_to_file):
    text_file = open(path_to_file, "w")
    text_file.write(text)
    text_file.close()
