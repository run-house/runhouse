import asyncio
import logging
import shlex
import subprocess
import threading
from functools import wraps
from typing import List, Union


def run_with_logs(cmd: Union[List[str], str], **kwargs) -> int:
    """Runs a command and prints the output to sys.stdout.
    We can't just pipe to sys.stdout, and when in a `call` method
    we overwrite sys.stdout with a multi-logger to a file and stdout.

    Args:
        cmd: The command to run.
        kwargs: Keyword arguments to pass to subprocess.Popen.

    Returns:
        The returncode of the command.
    """
    if isinstance(cmd, str):
        cmd = shlex.split(cmd) if not kwargs.get("shell", False) else [cmd]
    require_outputs = kwargs.pop("require_outputs", False)
    stream_logs = kwargs.pop("stream_logs", True)

    p = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, **kwargs
    )
    stdout, stderr = p.communicate()

    if stream_logs:
        print(stdout)

    if require_outputs:
        return p.returncode, stdout, stderr

    return p.returncode


def install_conda():
    if run_with_logs("conda --version") != 0:
        logging.info("Conda is not installed")
        run_with_logs(
            "wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh",
            shell=True,
        )
        run_with_logs("bash ~/miniconda.sh -b -p ~/miniconda", shell=True)
        run_with_logs("source $HOME/miniconda3/bin/activate", shell=True)
        if run_with_logs("conda --version") != 0:
            raise RuntimeError("Could not install Conda.")


# I got led to this solution from an answer in here:
# https://stackoverflow.com/questions/46827007/runtimeerror-this-event-loop-is-already-running-in-python
# That originally used nest_asyncio, but I wanted to avoid that, and some actual good ass engineer thought of this:
# https://stackoverflow.com/questions/52232177/runtimeerror-timeout-context-manager-should-be-used-inside-a-task/69514930#69514930
# which I still don't really get


def _start_background_loop(loop):
    asyncio.set_event_loop(loop)
    loop.run_forever()


# This should only run per process, I guess
_LOOP = asyncio.new_event_loop()
_LOOP_THREAD = threading.Thread(
    target=_start_background_loop, args=(_LOOP,), daemon=True
)
_LOOP_THREAD.start()


def sync_function(coroutine_func):
    @wraps(coroutine_func)
    def wrapper(*args, **kwargs):
        return asyncio.run_coroutine_threadsafe(
            coroutine_func(*args, **kwargs), _LOOP
        ).result()

    return wrapper


####################################################################################################
# Other implementations I've tried of this
####################################################################################################

# Useful StackOverflows:

# def sync_function(coroutine_func):

#     @wraps(coroutine_func)
#     def wrapper(*args, **kwargs):
#         try:
#             old_loop = asyncio.get_running_loop()
#         except RuntimeError:
#             old_loop = None

#         inner_new_loop = asyncio.new_event_loop()
#         asyncio.set_event_loop(inner_new_loop)

#         try:
#             return inner_new_loop.run_until_complete(coroutine_func(*args, **kwargs))
#         finally:
#             inner_new_loop.close()
#             if old_loop is not None:
#                 asyncio.set_event_loop(old_loop)

#     return wrapper


# def sync_function(coroutine_func):

#     @wraps(coroutine_func)
#     def wrapper(*args, **kwargs):
#         try:
#             old_loop = asyncio.get_running_loop()
#         except RuntimeError:
#             old_loop = None

#         inner_new_loop = asyncio.new_event_loop()
#         asyncio.set_event_loop(inner_new_loop)

#         try:
#             future = asyncio.run_coroutine_threadsafe(coroutine_func(*args, **kwargs), inner_new_loop)
#             return future.result()
#         finally:
#             inner_new_loop.close()
#             if old_loop is not None:
#                 asyncio.set_event_loop(old_loop)

#     return wrapper


# def sync_function(coroutine_func):
#     from asgiref.sync import sync_to_async
#     return sync_to_async(coroutine_func)
