import ast
import json

from requests import Response

ERROR_FLAG = "[ERROR]"
WARNING_FLAG = "[WARNING]"


def timing(func):
    def wrapper(*args, **kwargs):
        import time

        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"Finished {func.__name__.title()} in {int((end - start))} seconds")
        return result

    return wrapper


def remove_null_values_from_dict(source_dic: dict) -> dict:
    return {k: v for k, v in source_dic.items() if v is not None}


def read_response_data(resp: Response):
    return json.loads(resp.content).get("data", {})


def to_bool(value):
    try:
        return ast.literal_eval(value)
    except:
        return value


def is_jsonable(myjson):
    try:
        json.dumps(myjson)
    except ValueError:
        return False
    return True
