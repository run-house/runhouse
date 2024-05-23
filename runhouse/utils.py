import asyncio
import contextvars
import functools
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
    @functools.wraps(coroutine_func)
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


def get_pid():
    import os

    return os.getpid()


def get_node_ip():
    import socket

    return socket.gethostbyname(socket.gethostname())


async def arun_in_thread(method_to_run, *args, **kwargs):
    def _run_sync_fn_with_context(context_to_set, sync_fn, method_args, method_kwargs):
        for var, value in context_to_set.items():
            var.set(value)

        return sync_fn(*method_args, **method_kwargs)

    with ThreadPoolExecutor() as executor:
        return await asyncio.get_event_loop().run_in_executor(
            executor,
            functools.partial(
                _run_sync_fn_with_context,
                context_to_set=contextvars.copy_context(),
                sync_fn=method_to_run,
                method_args=args,
                method_kwargs=kwargs,
            ),
        )
