import asyncio
import contextvars
import logging
import shlex
import subprocess
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


from concurrent.futures import ThreadPoolExecutor


def _thread_coroutine(coroutine, context):
    # Copy contextvars from the parent thread to the new thread
    for var, value in context.items():
        var.set(value)

    # Technically, event loop logic is not threadsafe. However, this event loop is only in this thread.
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        # The loop runs only for the duration of the thread
        return loop.run_until_complete(coroutine)
    finally:
        # We don't need to do asyncio.set_event_loop(None) since the thread will just end completely
        loop.close()


# We should minimize calls to this since each one will start a new thread.
# Technically we should not have many threads running async logic at once, however, the calling thread
# will actually block until the async logic that is spawned in the other thread is done.
def sync_function(coroutine_func):
    @wraps(coroutine_func)
    def wrapper(*args, **kwargs):
        # Better API than using threading.Thread, since we just need the thread temporarily
        # and the resources are cleaned up
        with ThreadPoolExecutor() as executor:
            future = executor.submit(
                _thread_coroutine,
                coroutine_func(*args, **kwargs),
                contextvars.copy_context(),
            )
            return future.result()

    return wrapper


####################################################################################################
# Other implementations I've tried of this
####################################################################################################
# WORKING
#
# import nest_asyncio
# from asyncio import Task

# def _to_task(future, as_task, loop):
#     if not as_task or isinstance(future, Task):
#         return future
#     return loop.create_task(future)

# def sync_function(coroutine_func, as_task=True):
#     """
#     A better implementation of `asyncio.run`.

#     :param future: A future or task or call of an async method.
#     :param as_task: Forces the future to be scheduled as task (needed for e.g. aiohttp).
#     """
#     @wraps(coroutine_func)
#     def wrapper(*args, **kwargs):
#         future = coroutine_func(*args, **kwargs)
#         try:
#             loop = asyncio.get_running_loop()
#         except RuntimeError:  # no event loop running:
#             loop = asyncio.new_event_loop()
#             return loop.run_until_complete(_to_task(future, as_task, loop))
#         else:
#             nest_asyncio.apply(loop)
#             return asyncio.run(_to_task(future, as_task, loop))

#     return wrapper


# KINDA WORKING
#
# I got led to this solution from an answer in here:
# https://stackoverflow.com/questions/46827007/runtimeerror-this-event-loop-is-already-running-in-python
# That originally used nest_asyncio, but I wanted to avoid that, and some actual good ass engineer thought of this:
# https://stackoverflow.com/questions/52232177/runtimeerror-timeout-context-manager-should-be-used-inside-a-task/69514930#69514930
# which I still don't really get

# Another useful post: https://stackoverflow.com/questions/57238316/future-from-asyncio-run-coroutine-threadsafe-hangs-forever

# import threading

# def _start_background_loop(loop):
#     asyncio.set_event_loop(loop)
#     loop.run_forever()


# # This should only run per process, I guess
# _LOOP = asyncio.new_event_loop()
# _LOOP_THREAD = threading.Thread(
#     target=_start_background_loop, args=(_LOOP,), daemon=True
# )
# _LOOP_THREAD.start()

# def sync_function(coroutine_func):
#     @wraps(coroutine_func)
#     def wrapper(*args, **kwargs):
#         return asyncio.run_coroutine_threadsafe(
#             coroutine_func(*args, **kwargs), _LOOP
#         ).result()

#     return wrapper


# NOT WORKING
#
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

# NOT WORKING
#
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

# NOT WORKING
#
# def sync_function(coroutine_func):
#     from asgiref.sync import async_to_sync
#     return async_to_sync(coroutine_func)
