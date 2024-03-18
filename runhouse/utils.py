import logging
import shlex
import subprocess
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
