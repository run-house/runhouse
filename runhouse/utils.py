import asyncio
import contextvars
from concurrent.futures import ThreadPoolExecutor
from functools import wraps

from alive_progress import alive_bar
from rich.emoji import Emoji


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


def string_to_dict(dict_as_string):
    parts = dict_as_string.split(":")
    key = parts[0].strip()
    value = parts[1].strip()
    return key, value


####################################################################################################
# Styling utils
####################################################################################################
def success_emoji(text: str) -> str:
    return f"{Emoji('white_check_mark')} {text}"


def failure_emoji(text: str) -> str:
    return f"{Emoji('cross_mark')} {text}"


def alive_bar_spinner_only(*args, **kwargs):
    return alive_bar(
        bar=None,  # No actual bar
        enrich_print=False,  # Print statements while the bar is running are unmodified
        monitor=False,
        stats=False,
        monitor_end=True,
        stats_end=False,
        title_length=0,
        *args,
        **kwargs,
    )
