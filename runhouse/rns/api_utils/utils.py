import json
import time
import random
import ast
from requests import Response
from runhouse.rns.api_utils.names import names
from dataclasses import dataclass

ERROR_FLAG = "[ERROR]"
WARNING_FLAG = "[WARNING]"


def current_time() -> float:
    return time.time()


def random_string_generator():
    """Return random name based on moby dick corpus"""
    return '-'.join(random.sample(set(names), 2))


def timing(func):
    def wrapper(*args, **kwargs):
        import time
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f'Finished {func.__name__.title()} in {int((end - start))} seconds')
        return result

    return wrapper


def remove_null_values_from_dict(source_dic: dict) -> dict:
    return {k: v for k, v in source_dic.items() if v is not None}


def read_response_data(resp: Response):
    return json.loads(resp.content).get('data', {})


def error_message(msg):
    return f'[bold red]{ERROR_FLAG} {msg}[/bold red]'


def warning_message(msg):
    return f'[bold yellow]{WARNING_FLAG} {msg}[/bold yellow]'


def pprint_color(msg, color='green'):
    return f'[{color}]{msg}[/{color}]'


def to_bool(value):
    try:
        return ast.literal_eval(value)
    except:
        return value


@dataclass
class Common:
    """Context manager for all the possible CLI options the user can provide"""
    cluster: str
    name: str
    hardware: str
    dockerfile: str
    fn: str
    image: str
    status: bool
    resource_dir: str
    working_dir: str

    def to_dict(self):
        return vars(self)
