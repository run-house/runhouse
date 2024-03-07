# Source: https://github.com/skypilot-org/skypilot/blob/feb52cf/sky/skylet/log_lib.py

import copy
import io
import multiprocessing.pool  # note: changed this
import os
import subprocess
import sys
from typing import List, Optional, Tuple, Union

from runhouse.resources.hardware.sky import subprocess_utils

class LineProcessor(object):
    """A processor for log lines."""

    def __enter__(self):
        pass

    def process_line(self, log_line):
        pass

    def __exit__(self, except_type, except_value, traceback):
        del except_type, except_value, traceback  # unused
        pass

class _ProcessingArgs:
    """Arguments for processing logs."""

    def __init__(self,
                 log_path: str,
                 stream_logs: bool,
                 start_streaming_at: str = '',
                 end_streaming_at: Optional[str] = None,
                 skip_lines: Optional[List[str]] = None,
                 replace_crlf: bool = False,
                 line_processor: Optional[LineProcessor] = None,
                 streaming_prefix: Optional[str] = None) -> None:
        self.log_path = log_path
        self.stream_logs = stream_logs
        self.start_streaming_at = start_streaming_at
        self.end_streaming_at = end_streaming_at
        self.skip_lines = skip_lines
        self.replace_crlf = replace_crlf
        self.line_processor = line_processor
        self.streaming_prefix = streaming_prefix


def process_subprocess_stream(proc, args: _ProcessingArgs) -> Tuple[str, str]:
    """Redirect the process's filtered stdout/stderr to both stream and file"""
    if proc.stderr is not None:
        # Asyncio does not work as the output processing can be executed in a
        # different thread.
        # selectors is possible to handle the multiplexing of stdout/stderr,
        # but it introduces buffering making the output not streaming.
        with multiprocessing.pool.ThreadPool(processes=1) as pool:
            err_args = copy.copy(args)
            err_args.line_processor = None
            stderr_fut = pool.apply_async(_handle_io_stream,
                                          args=(proc.stderr, sys.stderr,
                                                err_args))
            # Do not launch a thread for stdout as the rich.status does not
            # work in a thread, which is used in
            # log_utils.RayUpLineProcessor.
            stdout = _handle_io_stream(proc.stdout, sys.stdout, args)
            stderr = stderr_fut.get()
    else:
        stdout = _handle_io_stream(proc.stdout, sys.stdout, args)
        stderr = ''
    return stdout, stderr

def _handle_io_stream(io_stream, out_stream, args: _ProcessingArgs):
    """Process the stream of a process."""
    out_io = io.TextIOWrapper(io_stream,
                              encoding='utf-8',
                              newline='',
                              errors='replace',
                              write_through=True)

    start_streaming_flag = False
    end_streaming_flag = False
    streaming_prefix = args.streaming_prefix if args.streaming_prefix else ''
    line_processor = (LineProcessor()
                      if args.line_processor is None else args.line_processor)

    out = []
    with open(args.log_path, 'a') as fout:
        with line_processor:
            while True:
                line = out_io.readline()
                if not line:
                    break
                # start_streaming_at logic in processor.process_line(line)
                if args.replace_crlf and line.endswith('\r\n'):
                    # Replace CRLF with LF to avoid ray logging to the same
                    # line due to separating lines with '\n'.
                    line = line[:-2] + '\n'
                if (args.skip_lines is not None and
                        any(skip in line for skip in args.skip_lines)):
                    continue
                if args.start_streaming_at in line:
                    start_streaming_flag = True
                if (args.end_streaming_at is not None and
                        args.end_streaming_at in line):
                    # Keep executing the loop, only stop streaming.
                    # E.g., this is used for `sky bench` to hide the
                    # redundant messages of `sky launch` while
                    # saving them in log files.
                    end_streaming_flag = True
                if (args.stream_logs and start_streaming_flag and
                        not end_streaming_flag):
                    print(streaming_prefix + line,
                          end='',
                          file=out_stream,
                          flush=True)
                if args.log_path != '/dev/null':
                    fout.write(line)
                    fout.flush()
                line_processor.process_line(line)
                out.append(line)
    return ''.join(out)




def run_with_log(
    cmd: Union[List[str], str],
    log_path: str,
    *,
    require_outputs: bool = False,
    stream_logs: bool = False,
    start_streaming_at: str = '',
    end_streaming_at: Optional[str] = None,
    skip_lines: Optional[List[str]] = None,
    shell: bool = False,
    with_ray: bool = False,
    process_stream: bool = True,
    line_processor: Optional[LineProcessor] = None,
    streaming_prefix: Optional[str] = None,
    ray_job_id: Optional[str] = None,
    use_sudo: bool = False,
    **kwargs,
) -> Union[int, Tuple[int, str, str]]:
    """Runs a command and logs its output to a file.

    Args:
        cmd: The command to run.
        log_path: The path to the log file.
        stream_logs: Whether to stream the logs to stdout/stderr.
        require_outputs: Whether to return the stdout/stderr of the command.
        process_stream: Whether to post-process the stdout/stderr of the
            command, such as replacing or skipping lines on the fly. If
            enabled, lines are printed only when '\r' or '\n' is found.
        ray_job_id: The id for a ray job.
        use_sudo: Whether to use sudo to create log_path.

    Returns the returncode or returncode, stdout and stderr of the command.
      Note that the stdout and stderr is already decoded.
    """
    assert process_stream or not require_outputs, (
        process_stream, require_outputs,
        'require_outputs should be False when process_stream is False')

    log_path = os.path.expanduser(log_path)
    dirname = os.path.dirname(log_path)
    if use_sudo:
        # Sudo case is encountered when submitting
        # a job for Sky on-prem, when a non-admin user submits a job.
        subprocess.run(f'sudo mkdir -p {dirname}', shell=True, check=True)
        subprocess.run(f'sudo touch {log_path}; sudo chmod a+rwx {log_path}',
                       shell=True,
                       check=True)
        # Hack: Subprocess Popen does not accept sudo.
        # subprocess.Popen in local mode with shell=True does not work,
        # as it does not understand what -H means for sudo.
        shell = False
    else:
        os.makedirs(dirname, exist_ok=True)
    # Redirect stderr to stdout when using ray, to preserve the order of
    # stdout and stderr.
    stdout_arg = stderr_arg = None
    if process_stream:
        stdout_arg = subprocess.PIPE
        stderr_arg = subprocess.PIPE if not with_ray else subprocess.STDOUT
    with subprocess.Popen(cmd,
                          stdout=stdout_arg,
                          stderr=stderr_arg,
                          start_new_session=True,
                          shell=shell,
                          **kwargs) as proc:
        try:
            # The proc can be defunct if the python program is killed. Here we
            # open a new subprocess to gracefully kill the proc, SIGTERM
            # and then SIGKILL the process group.
            # Adapted from ray/dashboard/modules/job/job_manager.py#L154
            parent_pid = os.getpid()
            daemon_script = os.path.join(
                ### note: changed this
                os.path.dirname(os.path.abspath(__file__)),
                'subprocess_daemon.py')
            daemon_cmd = [
                'python3',
                daemon_script,
                '--parent-pid',
                str(parent_pid),
                '--proc-pid',
                str(proc.pid),
            ]
            # Bool use_sudo is true in the Sky On-prem case.
            # In this case, subprocess_daemon.py should run on the root user
            # and the Ray job id should be passed for daemon to poll for
            # job status (as `ray job stop` does not work in the
            # multitenant case).
            if use_sudo:
                daemon_cmd.insert(0, 'sudo')
                daemon_cmd.extend(['--local-ray-job-id', str(ray_job_id)])
            subprocess.Popen(
                daemon_cmd,
                start_new_session=True,
                # Suppress output
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                # Disable input
                stdin=subprocess.DEVNULL,
            )
            stdout = ''
            stderr = ''

            if process_stream:
                if skip_lines is None:
                    skip_lines = []
                # Skip these lines caused by `-i` option of bash. Failed to
                # find other way to turn off these two warning.
                # https://stackoverflow.com/questions/13300764/how-to-tell-bash-not-to-issue-warnings-cannot-set-terminal-process-group-and # pylint: disable=line-too-long
                # `ssh -T -i -tt` still cause the problem.
                skip_lines += [
                    'bash: cannot set terminal process group',
                    'bash: no job control in this shell',
                ]
                # We need this even if the log_path is '/dev/null' to ensure the
                # progress bar is shown.
                # NOTE: Lines are printed only when '\r' or '\n' is found.
                args = _ProcessingArgs(
                    log_path=log_path,
                    stream_logs=stream_logs,
                    start_streaming_at=start_streaming_at,
                    end_streaming_at=end_streaming_at,
                    skip_lines=skip_lines,
                    line_processor=line_processor,
                    # Replace CRLF when the output is logged to driver by ray.
                    replace_crlf=with_ray,
                    streaming_prefix=streaming_prefix,
                )
                stdout, stderr = process_subprocess_stream(proc, args)
            proc.wait()
            if require_outputs:
                return proc.returncode, stdout, stderr
            return proc.returncode
        except KeyboardInterrupt:
            # Kill the subprocess directly, otherwise, the underlying
            # process will only be killed after the python program exits,
            # causing the stream handling stuck at `readline`.
            subprocess_utils.kill_children_processes()
            raise
