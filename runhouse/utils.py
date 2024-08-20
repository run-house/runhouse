import asyncio
import contextvars
import functools

try:
    import importlib.metadata as metadata
except ImportError as e:
    # User is probably on Python<3.8
    try:
        import importlib_metadata as metadata
    except ImportError:
        # User needs to install importlib_metadata
        raise ImportError(
            f"importlib_metadata is not installed in Python<3.8. Please install it with "
            f"'pip install importlib_metadata'. {e}"
        )
import inspect
import json
import logging
import os
import re
import subprocess
import sys
import threading

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional, Type, Union

import pexpect

from runhouse.constants import LOGS_DIR
from runhouse.logger import get_logger

logger = get_logger(name=__name__)
####################################################################################################
# Python package utilities
####################################################################################################
def find_locally_installed_version(package_name: str) -> Optional[str]:
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return None


def get_local_install_path(package_name: str) -> Optional[str]:
    distribution = metadata.distribution(package_name)
    direct_url_json = distribution.read_text("direct_url.json")
    if direct_url_json:
        # File URL starts with file://
        try:
            url = json.loads(direct_url_json).get("url", None)
            if url:
                if url.startswith("file://"):
                    return url[len("file://") :]
        except json.JSONDecodeError:
            return None


def is_python_package_string(s: str) -> bool:
    return isinstance(s, str) and re.match(r"^[a-zA-Z0-9\._-]+$", s) is not None


####################################################################################################
# Simple running utility
####################################################################################################
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
        **kwargs,
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


####################################################################################################
# Module discovery and import logic
####################################################################################################


def _find_directory_containing_any_file(dir_path, files, searched_dirs=None):
    if Path(dir_path) == Path.home() or dir_path == Path("/"):
        return None

    if any(Path(dir_path, file).exists() for file in files):
        return str(dir_path)

    searched_dirs.add(dir_path)
    parent_path = Path(dir_path).parent
    if parent_path in searched_dirs:
        return None
    return _find_directory_containing_any_file(
        parent_path, files, searched_dirs=searched_dirs
    )


def locate_working_dir(start_dir=None):
    if start_dir is None:
        start_dir = os.getcwd()

    # Search first for anything that represents a Python package
    target_files = [
        ".git",
        "setup.py",
        "setup.cfg",
        "pyproject.toml",
        "requirements.txt",
    ]

    dir_with_target = _find_directory_containing_any_file(
        start_dir, target_files, searched_dirs=set()
    )

    if dir_with_target is None:
        dir_with_target = _find_directory_containing_any_file(
            start_dir, ["rh"], searched_dirs=set()
        )

    return dir_with_target if dir_with_target is not None else start_dir


def extract_module_path(raw_cls_or_fn: Union[Type, Callable]):
    py_module = inspect.getmodule(raw_cls_or_fn)

    # Need to resolve in case just filename is given
    module_path = (
        str(Path(inspect.getfile(py_module)).resolve())
        if hasattr(py_module, "__file__")
        else None
    )

    return module_path


def get_module_import_info(raw_cls_or_fn: Union[Type, Callable]):
    """
    Given a class or function in Python, get all the information needed to import it in another Python process.
    """

    # Background on all these dunders: https://docs.python.org/3/reference/import.html
    py_module = inspect.getmodule(raw_cls_or_fn)

    # Need to resolve in case just filename is given
    module_path = extract_module_path(raw_cls_or_fn)

    # TODO better way of detecting if in a notebook or interactive Python env
    if not module_path or module_path.endswith("ipynb"):
        # The only time __file__ wouldn't be present is if the function is defined in an interactive
        # interpreter or a notebook. We can't import on the server in that case, so we need to cloudpickle
        # the fn to send it over. The __call__ function will serialize the function if we return it this way.
        # This is a short-term hack.
        # return None, "notebook", raw_fn.__name__
        root_path = os.getcwd()
        module_name = "notebook"
        cls_or_fn_name = raw_cls_or_fn.__name__
    else:
        root_path = os.path.dirname(module_path)
        module_name = inspect.getmodulename(module_path)
        # TODO __qualname__ doesn't work when fn is aliased funnily, like torch.sum
        cls_or_fn_name = getattr(raw_cls_or_fn, "__qualname__", raw_cls_or_fn.__name__)

        # Adapted from https://github.com/modal-labs/modal-client/blob/main/modal/_function_utils.py#L94
        if getattr(py_module, "__package__", None):
            module_path = os.path.abspath(py_module.__file__)
            package_paths = [
                os.path.abspath(p) for p in __import__(py_module.__package__).__path__
            ]
            base_dirs = [
                base_dir
                for base_dir in package_paths
                if os.path.commonpath((base_dir, module_path)) == base_dir
            ]

            if len(base_dirs) != 1:
                logger.debug(f"Module files: {module_path}")
                logger.debug(f"Package paths: {package_paths}")
                logger.debug(f"Base dirs: {base_dirs}")
                raise Exception("Wasn't able to find the package directory!")
            root_path = os.path.dirname(base_dirs[0])
            module_name = py_module.__spec__.name

    return root_path, module_name, cls_or_fn_name


####################################################################################################
# Run command with password
####################################################################################################


def run_command_with_password_login(
    command: str, password: str, stream_logs: bool = True
):
    command_run = pexpect.spawn(command, encoding="utf-8", timeout=None)
    if stream_logs:
        # FYI This will print a ton of of stuff to stdout
        command_run.logfile_read = sys.stdout

    # If CommandRunner uses the control path, the password may not be requested
    next_line = command_run.expect(["assword:", pexpect.EOF])
    if next_line == 0:
        command_run.sendline(password)
        command_run.expect(pexpect.EOF)
    command_run.close()

    return command_run


####################################################################################################
# Async helpers
####################################################################################################


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


####################################################################################################
# Misc helpers
####################################################################################################


def get_pid():
    import os

    return os.getpid()


def get_node_ip():
    import socket

    return socket.gethostbyname(socket.gethostname())


class ThreadWithException(threading.Thread):
    def run(self):
        self._exc = None
        try:
            super().run()
        except Exception as e:
            self._exc = e

    def join(self, timeout=None):
        super().join(timeout=timeout)
        if self._exc:
            raise self._exc


def client_call_wrapper(client, system, client_method_name, *args, **kwargs):
    from runhouse.resources.hardware import Cluster

    if system and isinstance(system, Cluster) and not system.on_this_cluster():
        return system.call_client_method(client_method_name, *args, **kwargs)
    method = getattr(client, client_method_name)
    return method(*args, **kwargs)


####################################################################################################
# Logging redirection
####################################################################################################
class StreamTee(object):
    def __init__(self, instream, outstreams):
        self.instream = instream
        self.outstreams = outstreams

    def write(self, message):
        self.instream.write(message)
        for stream in self.outstreams:
            if message:
                stream.write(message)
                # We flush here to ensure that the logs are written to the file immediately
                # see https://github.com/run-house/runhouse/pull/724
                stream.flush()

    def writelines(self, lines):
        self.instream.writelines(lines)
        for stream in self.outstreams:
            stream.writelines(lines)
            stream.flush()

    def flush(self):
        self.instream.flush()
        for stream in self.outstreams:
            stream.flush()

    def __getattr__(self, item):
        # Needed in case someone calls a method on instream, such as Ray calling sys.stdout.istty()
        return getattr(self.instream, item)


class LogToFolder:
    def __init__(self, name: str):
        self.name = name
        self.directory = self._base_local_folder_path(name)
        self.root_logger = logging.getLogger("")
        # We do exist_ok=True here because generator runs are separate calls to the same directory.
        os.makedirs(self.directory, exist_ok=True)

    def __enter__(self):
        # TODO fix the fact that we keep appending and then stream back the full file
        sys.stdout = StreamTee(sys.stdout, [Path(self._stdout_path).open(mode="a")])
        sys.stderr = StreamTee(sys.stderr, [Path(self._stderr_path).open(mode="a")])

        # Add the stdout and stderr handlers to the root logger
        self._stdout_handler = logging.StreamHandler(sys.stdout)
        self.root_logger.addHandler(self._stdout_handler)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.root_logger.removeHandler(self._stdout_handler)

        # Flush stdout and stderr
        # sys.stdout.flush()
        # sys.stderr.flush()

        # Restore stdout and stderr
        if hasattr(sys.stdout, "instream"):
            sys.stdout = sys.stdout.instream
        if hasattr(sys.stderr, "instream"):
            sys.stderr = sys.stderr.instream

        # return False to propagate any exception that occurred inside the with block
        return False

    @property
    def _stdout_path(self) -> str:
        """Path to the stdout file for the Run."""
        return self._path_to_file_by_ext(ext=".out")

    @property
    def _stderr_path(self) -> str:
        """Path to the stderr file for the Run."""
        return self._path_to_file_by_ext(ext=".err")

    @staticmethod
    def _base_local_folder_path(name: str):
        """Path to the base folder for this Run on a local system."""
        return f"{LOGS_DIR}/{name}"

    @staticmethod
    def _filter_files_by_ext(files: list, ext: str):
        return list(filter(lambda x: x.endswith(ext), files))

    def _find_file_path_by_ext(self, ext: str) -> Union[str, None]:
        """Get the file path by provided extension. Needed when loading the stdout and stderr files associated
        with a particular run."""
        try:
            # List all files in self.directory
            folder_contents = os.listdir(self.directory)
        except FileNotFoundError:
            return None

        files_with_ext = self._filter_files_by_ext(folder_contents, ext)
        if not files_with_ext:
            # No .out / .err file already created in the logs folder for this Run
            return None

        # Return the most recent file with this extension
        return f"{self.directory}/{files_with_ext[0]}"

    def _path_to_file_by_ext(self, ext: str) -> str:
        """Path the file for the Run saved on the system for a provided extension (ex: ``.out`` or ``.err``)."""
        existing_file = self._find_file_path_by_ext(ext=ext)
        if existing_file:
            # If file already exists in file (ex: with function on a Ray cluster this will already be
            # generated for us)
            return existing_file

        path_to_ext = f"{self.directory}/{self.name}" + ext
        return path_to_ext


####################################################################################################
# Name generation
####################################################################################################


def generate_default_name(prefix: str = None, precision: str = "s", sep="_") -> str:
    """Name of the Run's parent folder which contains the Run's data (config, stdout, stderr, etc).
    If a name is provided, prepend that to the current timestamp to complete the folder name."""
    if precision == "d":
        timestamp_key = f"{datetime.now().strftime('%Y%m%d')}"
    elif precision == "s":
        timestamp_key = f"{datetime.now().strftime(f'%Y%m%d{sep}%H%M%S')}"
    elif precision == "ms":
        timestamp_key = f"{datetime.now().strftime(f'%Y%m%d{sep}%H%M%S_%f')}"
    if prefix is None:
        return timestamp_key
    return f"{prefix}{sep}{timestamp_key}"


####################################################################################################
# Logger utils
####################################################################################################
class ColoredFormatter:
    COLORS = {
        "black": "\u001b[30m",
        "red": "\u001b[31m",
        "green": "\u001b[32m",
        "yellow": "\u001b[33m",
        "blue": "\u001b[34m",
        "magenta": "\u001b[35m",
        "cyan": "\u001b[36m",
        "white": "\u001b[37m",
        "reset": "\u001b[0m",
    }

    @classmethod
    def get_color(cls, color: str):
        return cls.COLORS.get(color, "")

    # TODO: This method is a temp solution, until we'll update logging architecture. Remove once logging is cleaned up.
    @classmethod
    def format_log(cls, text):
        ansi_escape = re.compile(r"(?:\x1B[@-_][0-?]*[ -/]*[@-~])")
        return ansi_escape.sub("", text)


class ClusterLogsFormatter:
    def __init__(self, system):
        self.system = system
        self._display_title = False

    def format(self, output_type):
        from runhouse import Resource
        from runhouse.servers.http.http_utils import OutputType

        system_color = ColoredFormatter.get_color("cyan")
        reset_color = ColoredFormatter.get_color("reset")

        prettify_logs = output_type in [
            OutputType.STDOUT,
            OutputType.EXCEPTION,
            OutputType.STDERR,
        ]

        if (
            isinstance(self.system, Resource)
            and prettify_logs
            and not self._display_title
        ):
            # Display the system name before subsequent logs only once
            system_name = self.system.name
            dotted_line = "-" * len(system_name)
            print(dotted_line)
            print(f"{system_color}{system_name}{reset_color}")
            print(dotted_line)

            # Only display the system name once
            self._display_title = True

        return system_color, reset_color
