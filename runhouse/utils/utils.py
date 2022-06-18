import time
import random

from runhouse.utils.names import names

ERROR_FLAG = "[ERROR]"


def current_time() -> float:
    return time.time()


def random_string_generator():
    """Return random name based on moby dick corpus"""
    return '-'.join(random.sample(set(names), 2))
