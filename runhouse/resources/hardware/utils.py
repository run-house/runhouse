import json
import logging
import os
import pathlib
import shlex
import subprocess
import time
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from sky.skylet import log_lib
from sky.utils import subprocess_utils

from sky.utils.command_runner import (
    common_utils,
    GIT_EXCLUDE,
    RSYNC_DISPLAY_OPTION,
    RSYNC_EXCLUDE_OPTION,
    RSYNC_FILTER_OPTION,
    ssh_options_list,
    SSHCommandRunner,
    SshMode,
)

from sshtunnel import SSHTunnelForwarder

from runhouse.globals import ssh_tunnel_cache

logger = logging.getLogger(__name__)

RESERVED_SYSTEM_NAMES = ["file", "s3", "gs", "azure", "here", "ssh", "sftp"]
CLUSTER_CONFIG_PATH = "~/.rh/cluster_config.json"


# Get rid of the constant "Found credentials in shared credentials file: ~/.aws/credentials" message
try:
    import logging

    import boto3

    boto3.set_stream_logger(name="botocore.credentials", level=logging.ERROR)
except ImportError:
    pass

# TODO: Move the following two functions into a networking module
def get_open_tunnel(address: str, ssh_port: str):
    if (address, ssh_port) in ssh_tunnel_cache:
        ssh_tunnel, connected_port = ssh_tunnel_cache[(address, ssh_port)]
        if isinstance(ssh_tunnel, SSHTunnelForwarder):
            # Initializes tunnel_is_up dictionary
            ssh_tunnel.check_tunnels()

            if (
                ssh_tunnel.is_active
                and ssh_tunnel.tunnel_is_up[ssh_tunnel.local_bind_address]
            ):
                return ssh_tunnel, connected_port

            else:
                # If the tunnel is no longer active or up, pop it from the global cache
                ssh_tunnel_cache.pop((address, ssh_port))

    return None, None


def cache_open_tunnel(
    address: str,
    ssh_port: str,
    ssh_tunnel: SSHTunnelForwarder,
    connected_port: int,
):
    ssh_tunnel_cache[(address, ssh_port)] = (ssh_tunnel, connected_port)


class ServerConnectionType(str, Enum):
    """Manage the type of connection Runhouse will make with the API server started on the cluster.
    ``ssh``: Use port forwarding to connect to the server via SSH, by default on port 32300.
    ``tls``: Do not use port forwarding and start the server with HTTPS (using custom or fresh TLS certs), by default
        on port 443.
    ``none``: Do not use port forwarding, and start the server with HTTP, by default on port 80.
    ``aws_ssm``: Use AWS SSM to connect to the server, by default on port 32300.
    ``paramiko``: Use paramiko to connect to the server (e.g. if you provide a password with SSH credentials), by
        default on port 32300.
    """

    SSH = "ssh"
    TLS = "tls"
    NONE = "none"
    AWS_SSM = "aws_ssm"


def _current_cluster(key="name"):
    """Retrive key value from the current cluster config.
    If key is "config", returns entire config."""
    cluster_config = _load_cluster_config()
    if cluster_config:
        if key == "config":
            return cluster_config
        elif key == "cluster_name":
            return cluster_config["name"].rsplit("/", 1)[-1]
        return cluster_config[key]
    else:
        return None


def _load_cluster_config() -> Dict:
    if Path(CLUSTER_CONFIG_PATH).expanduser().exists():
        with open(Path(CLUSTER_CONFIG_PATH).expanduser()) as f:
            cluster_config = json.load(f)
        return cluster_config
    else:
        return {}


def _get_cluster_from(system, dryrun=False):
    from .cluster import Cluster

    if isinstance(system, Cluster):
        return system
    if system in RESERVED_SYSTEM_NAMES:
        return system

    if isinstance(system, Dict):
        return Cluster.from_config(system, dryrun)

    if isinstance(system, str):
        config = _current_cluster(key="config")
        if config and system == config["name"]:
            return Cluster.from_config(config, dryrun)
        try:
            system = Cluster.from_name(name=system, dryrun=dryrun)
        except ValueError:
            # Name not found in RNS. Doing the lookup this way saves us a hop to RNS
            pass

    return system


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
        self._tunnel_procs = []

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
                ssh += ["-L", f"{remote}:localhost:{local}"]
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
        logging.info(f"Running command: {' '.join(command)}")
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
        command = " ".join(base_cmd + ["tail"])
        logger.info(f"Running command: {command}")
        proc = subprocess.Popen(shlex.split(command), stdout=subprocess.PIPE)

        time.sleep(3)
        self._tunnel_procs.append(proc)

    def __del__(self):
        for proc in self._tunnel_procs:
            proc.kill()

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
