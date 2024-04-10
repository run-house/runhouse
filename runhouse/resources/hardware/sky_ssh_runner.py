import copy
import logging
import os
import pathlib
import shlex
import subprocess
import time
from typing import Dict, List, Optional, Tuple, Union

from runhouse.constants import LOCALHOST

from runhouse.globals import sky_ssh_runner_cache

from runhouse.resources.hardware.sky import common_utils, log_lib, subprocess_utils

from runhouse.resources.hardware.sky.command_runner import (
    GIT_EXCLUDE,
    RSYNC_DISPLAY_OPTION,
    RSYNC_EXCLUDE_OPTION,
    RSYNC_FILTER_OPTION,
    ssh_options_list,
    SSHCommandRunner,
    SshMode,
)


logger = logging.getLogger(__name__)


# Get rid of the constant "Found credentials in shared credentials file: ~/.aws/credentials" message
try:
    import logging

    import boto3

    boto3.set_stream_logger(name="botocore.credentials", level=logging.ERROR)
except ImportError:
    pass


def is_port_in_use(port: int) -> bool:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0


class SkySSHRunner(SSHCommandRunner):
    def __init__(
        self,
        ip,
        ssh_user=None,
        ssh_private_key=None,
        ssh_control_name: Optional[str] = "__default__",
        ssh_proxy_command: Optional[str] = None,
        port: int = 22,
        docker_user: Optional[str] = None,
        disable_control_master: Optional[bool] = False,
        local_bind_port: Optional[int] = None,
    ):
        super().__init__(
            ip,
            ssh_user,
            ssh_private_key,
            ssh_control_name,
            ssh_proxy_command,
            port,
            docker_user,
            disable_control_master,
        )
        self.tunnel_proc = None
        self.local_bind_port = local_bind_port
        self.remote_bind_port = None

    def _ssh_base_command(
        self, *, ssh_mode: SshMode, port_forward: Optional[List[int]]
    ) -> List[str]:
        ssh = ["ssh"]
        if ssh_mode == SshMode.NON_INTERACTIVE:
            # Disable pseudo-terminal allocation. Otherwise, the output of
            # ssh will be corrupted by the user's input.
            ssh += ["-T"]
        else:
            # Force pseudo-terminal allocation for interactive/login mode.
            ssh += ["-tt"]
        if port_forward is not None:
            # RH MODIFIED: Accept port int (to forward same port) or pair of ports
            for fwd in port_forward:
                if isinstance(fwd, int):
                    local, remote = fwd, fwd
                else:
                    local, remote = fwd
                logger.info(f"Forwarding port {local} to port {remote} on localhost.")
                ssh += ["-L", f"{local}:localhost:{remote}"]
        if self._docker_ssh_proxy_command is not None:
            docker_ssh_proxy_command = self._docker_ssh_proxy_command(ssh)
        else:
            docker_ssh_proxy_command = None
        return (
            ssh
            + ssh_options_list(
                self.ssh_private_key,
                self.ssh_control_name,
                ssh_proxy_command=self._ssh_proxy_command,
                docker_ssh_proxy_command=docker_ssh_proxy_command,
                # TODO change to None like before?
                port=self.port,
                disable_control_master=self.disable_control_master,
            )
            + [f"{self.ssh_user}@{self.ip}"]
        )

    def run(
        self,
        cmd: Union[str, List[str]],
        *,
        require_outputs: bool = False,
        port_forward: Optional[List[int]] = None,
        # Advanced options.
        log_path: str = os.devnull,
        # If False, do not redirect stdout/stderr to optimize performance.
        process_stream: bool = True,
        stream_logs: bool = True,
        ssh_mode: SshMode = SshMode.NON_INTERACTIVE,
        separate_stderr: bool = False,
        return_cmd: bool = False,  # RH MODIFIED
        quiet_ssh: bool = False,  # RH MODIFIED
        **kwargs,
    ) -> Union[int, Tuple[int, str, str]]:
        """Uses 'ssh' to run 'cmd' on a node with ip.

        Args:
            ip: The IP address of the node.
            cmd: The command to run.
            port_forward: A list of ports to forward from the localhost to the
            remote host.

            Advanced options:

            require_outputs: Whether to return the stdout/stderr of the command.
            log_path: Redirect stdout/stderr to the log_path.
            stream_logs: Stream logs to the stdout/stderr.
            check: Check the success of the command.
            ssh_mode: The mode to use for ssh.
                See SSHMode for more details.
            separate_stderr: Whether to separate stderr from stdout.
            return_cmd: If True, return the command string instead of running it.
            quiet_ssh: If True, do not print the OpenSSH outputs (i.e. add "-q" option to ssh).


        Returns:
            returncode
            or
            A tuple of (returncode, stdout, stderr).
        """
        base_ssh_command = self._ssh_base_command(
            ssh_mode=ssh_mode, port_forward=port_forward
        )
        if ssh_mode == SshMode.LOGIN:
            assert isinstance(cmd, list), "cmd must be a list for login mode."
            command = base_ssh_command + cmd
            proc = subprocess_utils.run(command, shell=False, check=False)
            return proc.returncode, "", ""
        if isinstance(cmd, list):
            cmd = " ".join(cmd)

        # RH MODIFIED: Add quiet_ssh option
        if quiet_ssh:
            base_ssh_command.append("-q")

        log_dir = os.path.expanduser(os.path.dirname(log_path))
        os.makedirs(log_dir, exist_ok=True)
        # We need this to correctly run the cmd, and get the output.
        command = [
            "bash",
            "--login",
            "-c",
            # Need this `-i` option to make sure `source ~/.bashrc` work.
            "-i",
        ]

        command += [
            shlex.quote(
                f"true && source ~/.bashrc && export OMP_NUM_THREADS=1 "
                f"PYTHONWARNINGS=ignore && ({cmd})"
            ),
        ]
        if not separate_stderr:
            command.append("2>&1")
        if not process_stream and ssh_mode == SshMode.NON_INTERACTIVE:
            command += [
                # A hack to remove the following bash warnings (twice):
                #  bash: cannot set terminal process group
                #  bash: no job control in this shell
                "| stdbuf -o0 tail -n +5",
                # This is required to make sure the executor of command can get
                # correct returncode, since linux pipe is used.
                "; exit ${PIPESTATUS[0]}",
            ]

        command_str = " ".join(command)
        command = base_ssh_command + [shlex.quote(command_str)]

        executable = None
        if not process_stream:
            if stream_logs:
                command += [
                    f"| tee {log_path}",
                    # This also requires the executor to be '/bin/bash' instead
                    # of the default '/bin/sh'.
                    "; exit ${PIPESTATUS[0]}",
                ]
            else:
                command += [f"> {log_path}"]
            executable = "/bin/bash"

        # RH MODIFIED: Return command instead of running it
        logging.debug(f"Running command: {' '.join(command)}")
        if return_cmd:
            return " ".join(command)

        return log_lib.run_with_log(
            " ".join(command),
            log_path,
            require_outputs=require_outputs,
            stream_logs=stream_logs,
            process_stream=process_stream,
            shell=True,
            executable=executable,
            **kwargs,
        )

    def tunnel(self, local_port, remote_port):
        base_cmd = self._ssh_base_command(
            ssh_mode=SshMode.NON_INTERACTIVE, port_forward=[(local_port, remote_port)]
        )
        command = " ".join(base_cmd)
        logger.debug(f"Running forwarding command: {command}")
        proc = subprocess.Popen(
            shlex.split(command),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        # Wait for the ssh connection to start
        time.sleep(1)

        # Set the tunnel process and ports to be cleaned up later
        self.tunnel_proc = proc
        self.local_bind_port = local_port
        self.remote_bind_port = remote_port

    def tunnel_is_up(self):
        # Try and do as much as we can to check that this is still alive and the port is still forwarded
        return self.local_bind_port is not None and is_port_in_use(self.local_bind_port)

    def __del__(self):
        self.terminate()

    def terminate(self):
        if self.tunnel_proc is not None:

            # Process keeping tunnel alive can only be killed with EOF
            self.tunnel_proc.stdin.close()
            self.tunnel_proc.wait()

            # Remove port forwarding
            port_fwd_cmd = " ".join(
                self._ssh_base_command(
                    ssh_mode=SshMode.NON_INTERACTIVE,
                    port_forward=[(self.local_bind_port, self.remote_bind_port)],
                )
            )
            cancel_port_fwd = port_fwd_cmd.replace("-T", "-O cancel")
            logger.debug(f"Running cancel command: {cancel_port_fwd}")
            completed_cancel_cmd = subprocess.run(
                shlex.split(cancel_port_fwd),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            if completed_cancel_cmd.returncode != 0:
                logger.warning(
                    f"Failed to cancel port forwarding from {self.local_bind_port} to {self.remote_bind_port}. "
                    f"Error: {completed_cancel_cmd.stderr}"
                )

            self.tunnel_proc = None
            self.local_bind_port = None
            self.remote_bind_port = None

    def rsync(
        self,
        source: str,
        target: str,
        *,
        up: bool,
        # Advanced options.
        filter_options: Optional[str] = None,  # RH MODIFIED
        log_path: str = os.devnull,
        stream_logs: bool = True,
        max_retry: int = 1,
        return_cmd: bool = False,
    ) -> None:
        """Uses 'rsync' to sync 'source' to 'target'.

        Args:
            source: The source path.
            target: The target path.
            up: The direction of the sync, True for local to cluster, False
              for cluster to local.
            filter_options: The filter options for rsync.
            log_path: Redirect stdout/stderr to the log_path.
            stream_logs: Stream logs to the stdout/stderr.
            max_retry: The maximum number of retries for the rsync command.
              This value should be non-negative.
            return_cmd: If True, return the command string instead of running it.

        Raises:
            exceptions.CommandError: rsync command failed.
        """
        # Build command.
        # TODO(zhwu): This will print a per-file progress bar (with -P),
        # shooting a lot of messages to the output. --info=progress2 is used
        # to get a total progress bar, but it requires rsync>=3.1.0 and Mac
        # OS has a default rsync==2.6.9 (16 years old).
        rsync_command = ["rsync", RSYNC_DISPLAY_OPTION]

        # RH MODIFIED: add --filter option
        addtl_filter_options = f" --filter='{filter_options}'" if filter_options else ""
        rsync_command.append(RSYNC_FILTER_OPTION + addtl_filter_options)

        if up:
            # The source is a local path, so we need to resolve it.
            # --exclude-from
            resolved_source = pathlib.Path(source).expanduser().resolve()
            if (resolved_source / GIT_EXCLUDE).exists():
                # Ensure file exists; otherwise, rsync will error out.
                rsync_command.append(
                    RSYNC_EXCLUDE_OPTION.format(str(resolved_source / GIT_EXCLUDE))
                )

        if self._docker_ssh_proxy_command is not None:
            docker_ssh_proxy_command = self._docker_ssh_proxy_command(["ssh"])
        else:
            docker_ssh_proxy_command = None
        ssh_options = " ".join(
            ssh_options_list(
                self.ssh_private_key,
                self.ssh_control_name,
                ssh_proxy_command=self._ssh_proxy_command,
                docker_ssh_proxy_command=docker_ssh_proxy_command,
                port=self.port,
                disable_control_master=self.disable_control_master,
            )
        )
        rsync_command.append(f'-e "ssh {ssh_options}"')
        # To support spaces in the path, we need to quote source and target.
        # rsync doesn't support '~' in a quoted local path, but it is ok to
        # have '~' in a quoted remote path.
        if up:
            full_source_str = str(resolved_source)
            if resolved_source.is_dir():
                full_source_str = os.path.join(full_source_str, "")
            rsync_command.extend(
                [
                    f"{full_source_str!r}",
                    f"{self.ssh_user}@{self.ip}:{target!r}",
                ]
            )
        else:
            rsync_command.extend(
                [
                    f"{self.ssh_user}@{self.ip}:{source!r}",
                    f"{os.path.expanduser(target)!r}",
                ]
            )
        command = " ".join(rsync_command)

        # RH MODIFIED: return command instead of running it
        if return_cmd:
            return command

        backoff = common_utils.Backoff(initial_backoff=5, max_backoff_factor=5)
        while max_retry >= 0:
            returncode, _, stderr = log_lib.run_with_log(
                command,
                log_path=log_path,
                stream_logs=stream_logs,
                shell=True,
                require_outputs=True,
            )
            if returncode == 0:
                break
            max_retry -= 1
            time.sleep(backoff.current_backoff())

        direction = "up" if up else "down"
        error_msg = (
            f"Failed to rsync {direction}: {source} -> {target}. "
            "Ensure that the network is stable, then retry."
        )
        subprocess_utils.handle_returncode(
            returncode, command, error_msg, stderr=stderr, stream_logs=stream_logs
        )


####################################################################################################
# Cache and retrieve existing SSH Runners that are set up for a given address and port
####################################################################################################
# TODO: Shouldn't the control master prevent new ssh connections from being created?
def get_existing_sky_ssh_runner(address: str, ssh_port: int) -> Optional[SkySSHRunner]:
    if (address, ssh_port) in sky_ssh_runner_cache:
        existing_runner = sky_ssh_runner_cache.get((address, ssh_port))
        if existing_runner.tunnel_is_up():
            return existing_runner
        else:
            sky_ssh_runner_cache.pop((address, ssh_port))

    else:
        return None


def cache_existing_sky_ssh_runner(
    address: str, ssh_port: int, runner: SkySSHRunner
) -> None:
    sky_ssh_runner_cache[(address, ssh_port)] = runner


def ssh_tunnel(
    address: str,
    ssh_creds: Dict,
    local_port: int,
    ssh_port: int = 22,
    remote_port: Optional[int] = None,
    num_ports_to_try: int = 0,
) -> SkySSHRunner:
    """Initialize an ssh tunnel from a remote server to localhost

    Args:
        address (str): The address of the server we are trying to port forward an address to our local machine with.
        ssh_creds (Dict): A dictionary of ssh credentials used to connect to the remote server.
        local_port (int): The port locally where we are attempting to bind the remote server address to.
        ssh_port (int): The port on the machine where the ssh server is running.
            This is generally port 22, but occasionally
            we may forward a container's ssh port to a different port
            on the actual machine itself (for example on a Docker VM). Defaults to 22.
        remote_port (Optional[int], optional): The port of the remote server
            we're attempting to port forward. Defaults to None.
        num_ports_to_try (int, optional): The number of local ports to attempt to bind to,
            starting at local_port and incrementing by 1 till we hit the max. Defaults to 0.

    Returns:
        SkySSHRunner: The initialized tunnel.
    """

    # Debugging cmds (mac):
    # netstat -vanp tcp | grep 32300
    # lsof -i :32300
    # kill -9 <pid>

    # If remote_port isn't specified,
    # assume that the first attempted local port is
    # the same as the remote port on the server.
    remote_port = remote_port or local_port

    tunnel = get_existing_sky_ssh_runner(address, ssh_port)
    if tunnel and tunnel.ip == address and tunnel.remote_bind_port == remote_port:
        logger.info(
            f"SSH tunnel on to server's port {remote_port} "
            f"via server's ssh port {ssh_port} already created with the cluster."
        )
        return tunnel

    while is_port_in_use(local_port):
        if num_ports_to_try < 0:
            raise Exception(
                f"Failed to create find open port after {num_ports_to_try} attempts"
            )

        logger.info(f"Port {local_port} is already in use. Trying next port.")
        local_port += 1
        num_ports_to_try -= 1

    # Start a tunnel using self.run in a thread, instead of ssh_tunnel
    ssh_credentials = copy.copy(ssh_creds)

    # Host could be a proxy specified in credentials or is the provided address
    host = ssh_credentials.pop("ssh_host", address)
    ssh_control_name = ssh_credentials.pop("ssh_control_name", f"{address}:{ssh_port}")

    runner = SkySSHRunner(
        ip=host,
        ssh_user=ssh_creds.get("ssh_user"),
        ssh_private_key=ssh_creds.get("ssh_private_key"),
        ssh_proxy_command=ssh_creds.get("ssh_proxy_command"),
        ssh_control_name=ssh_control_name,
        port=ssh_port,
    )
    runner.tunnel(local_port, remote_port)

    logger.debug(
        f"Successfully bound "
        f"{LOCALHOST}:{remote_port} via ssh port {ssh_port} "
        f"on remote server {address} "
        f"to {LOCALHOST}:{local_port} on local machine."
    )

    cache_existing_sky_ssh_runner(address, ssh_port, runner)
    return runner
