# Source: https://github.com/skypilot-org/skypilot/blob/feb52cf/sky/utils/subprocess_utils.py

import psutil
from typing import Callable, List, Optional, Union

from runhouse.logger import get_logger

logger = get_logger(__name__)

class CommandError(Exception):
    """Raised when a command fails.

    Args:
        returncode: The returncode of the command.
        command: The command that was run.
        error_message: The error message to print.
        detailed_reason: The stderr of the command.
    """

    def __init__(self, returncode: int, command: str, error_msg: str,
                 detailed_reason: Optional[str]) -> None:
        self.returncode = returncode
        self.command = command
        self.error_msg = error_msg
        self.detailed_reason = detailed_reason
        message = (f'Command {command} failed with return code {returncode}.'
                   f'\n{error_msg}')
        super().__init__(message)


def handle_returncode(returncode: int,
                      command: str,
                      error_msg: Union[str, Callable[[], str]],
                      stderr: Optional[str] = None,
                      stream_logs: bool = True) -> None:
    """Handle the returncode of a command.

    Args:
        returncode: The returncode of the command.
        command: The command that was run.
        error_msg: The error message to print.
        stderr: The stderr of the command.
    """
    echo = logger.error if stream_logs else lambda _: None
    if returncode != 0:
        if stderr is not None:
            echo(stderr)

        if callable(error_msg):
            error_msg = error_msg()
        raise CommandError(returncode, command, error_msg, stderr)

def kill_children_processes(
        first_pid_to_kill: Optional[Union[int, List[Optional[int]]]] = None,
        force: bool = False):
    """Kill children processes recursively.

    We need to kill the children, so that
    1. The underlying subprocess will not print the logs to the terminal,
       after this program exits.
    2. The underlying subprocess will not continue with starting a cluster
       etc. while we are cleaning up the clusters.

    Args:
        first_pid_to_kill: Optional PID of a process, or PIDs of a series of
         processes to be killed first. If a list of PID is specified, it is
         killed by the order in the list.
         This is for guaranteeing the order of cleaning up and suppress
         flaky errors.
    """
    pid_to_proc = dict()
    child_processes = []
    if isinstance(first_pid_to_kill, int):
        first_pid_to_kill = [first_pid_to_kill]
    elif first_pid_to_kill is None:
        first_pid_to_kill = []

    def _kill_processes(processes: List[psutil.Process]) -> None:
        for process in processes:
            try:
                if force:
                    process.kill()
                else:
                    process.terminate()
            except psutil.NoSuchProcess:
                # The process may have already been terminated.
                pass

    parent_process = psutil.Process()
    for child in parent_process.children(recursive=True):
        if child.pid in first_pid_to_kill:
            pid_to_proc[child.pid] = child
        else:
            child_processes.append(child)

    _kill_processes([
        pid_to_proc[proc] for proc in first_pid_to_kill if proc in pid_to_proc
    ])
    _kill_processes(child_processes)
