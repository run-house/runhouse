import shutil
import os
from pathlib import Path
from runhouse.rns.api_utils.utils import random_string_generator
from runhouse.rns.api_utils.validation import validate_name


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


def delete_file(file_path):
    try:
        os.remove(file_path)
    except OSError:
        pass


def create_name_for_folder(name):
    if name is None:
        # if user did not provide a name we'll make one up (inspired by Moby Dick)
        name = random_string_generator().lower()
        return name

    # make sure the user provided name is in line with runhouse conventions
    validate_name(name)
    return name.lower()


def write_stdout_to_file(text, path_to_file):
    text_file = open(path_to_file, "w")
    text_file.write(text)
    text_file.close()


def get_name_from_path(path: str):
    return os.path.basename(path)


def convert_text_file_to_list(file_path: Path) -> list:
    return file_path.read_text().split('\n')
