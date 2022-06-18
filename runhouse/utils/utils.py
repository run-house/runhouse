import os
import shutil
import time
import random


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
    words = read_list_from_file(file_name="names.txt")
    return '-'.join(random.sample(set(words), 2))


def delete_directory(directory):
    try:
        shutil.rmtree(directory)
    except OSError as e:
        print("Error: %s - %s." % (e.filename, e.strerror))


def write_list_to_file(file_name, lst):
    with open(file_name, 'w') as filehandle:
        filehandle.writelines("%s\n" % word for word in lst)


def read_list_from_file(file_name):
    # define empty list
    words = []

    # open file and read the content in a list
    with open(file_name, 'r') as filehandle:
        filecontents = filehandle.readlines()

        for line in filecontents:
            # remove linebreak which is the last character of the string
            current_word = line[:-1]
            # add item to the list
            words.append(current_word)

    return words
