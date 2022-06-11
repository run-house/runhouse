import os
import time


def create_directory(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def save_to_file(stdout):
    output = ""
    for line in stdout:
        output += line
    # TODO now save this somewhere ...


def read_file(filepath):
    if not valid_filepath(filepath):
        raise Exception(f'File not found: {filepath}')

    with open(filepath, "r") as f:
        data = f.read()
    return data


def valid_filepath(filepath) -> bool:
    # NOTE: Assumes the scripts live in the "scripts" folder
    return os.path.exists(filepath)


def current_time() -> float:
    return time.time()
