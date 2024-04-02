import asyncio
import contextvars
import logging
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from functools import wraps

from runhouse.constants import CONDA_INSTALL_CMDS


def run_setup_command(cmd: str, cluster: "Cluster" = None, stream_logs: bool = False):
    """
    Helper function to run a command during possibly the cluster default env setup. If a cluster is provided,
    run command on the cluster using SSH. If the cluster is not provided, run locally, as if already on the
    cluster (rpc call).

    Args:
        cmd (str): Command to run on the
        cluster (Optional[Cluster]): (default: None)
        stream_logs (bool): (default: False)

    Returns:
       (status code, stdout)
    """
    if not cluster:
        return run_with_logs(cmd, stream_logs=stream_logs, require_outputs=True)[:2]
    return cluster._run_commands_with_ssh([cmd], stream_logs=stream_logs)[0]


def run_with_logs(cmd: str, **kwargs):
    """Runs a command and prints the output to sys.stdout.
    We can't just pipe to sys.stdout, and when in a `call` method
    we overwrite sys.stdout with a multi-logger to a file and stdout.

    Args:
        cmd: The command to run.
        kwargs: Keyword arguments to pass to subprocess.Popen.

    Returns:
        The returncode of the command.
    """
    require_outputs = kwargs.pop("require_outputs", False)
    stream_logs = kwargs.pop("stream_logs", True)

    p = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        shell=True,
        **kwargs
    )

    out = ""
    if stream_logs:
        while True:
            line = p.stdout.readline()
            if line == "" and p.poll() is not None:
                break
            sys.stdout.write(line)
            sys.stdout.flush()
            if require_outputs:
                out += line

    stdout, stderr = p.communicate()

    if require_outputs:
        stdout = stdout or out
        return p.returncode, stdout, stderr

    return p.returncode


def install_conda(cluster: "Cluster" = None):
    if run_setup_command("conda --version")[0] != 0:
        logging.info("Conda is not installed. Installing...")
        for cmd in CONDA_INSTALL_CMDS:
            run_setup_command(cmd, stream_logs=True)
        if run_setup_command("conda --version")[0] != 0:
            raise RuntimeError("Could not install Conda.")


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
