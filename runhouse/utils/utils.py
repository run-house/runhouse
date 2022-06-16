import os
import shutil
import time
import random

import nltk
# TODO downloading nltk seems unnecessary
# nltk.download('gutenberg')
from nltk.corpus import gutenberg


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
    return os.path.exists(filepath)


def current_time() -> float:
    return time.time()


def random_string_generator():
    """Return random name based on moby dick"""
    moby = set(nltk.Text(gutenberg.words('melville-moby_dick.txt')))
    moby = [word.lower() for word in moby if len(word) > 2]
    return '-'.join(random.sample(set(moby), 2))


def delete_directory(directory):
    try:
        shutil.rmtree(directory)
    except OSError as e:
        print("Error: %s - %s." % (e.filename, e.strerror))


def error_flag():
    return "[ERROR]"
