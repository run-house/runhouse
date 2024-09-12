import logging
import os
import shlex
from typing import Dict, List, Optional, Tuple, Union

from runhouse.constants import DEFAULT_DOCKER_CONTAINER_NAME

from runhouse.logger import get_logger

from runhouse.resources.hardware.sky import log_lib, subprocess_utils

from runhouse.resources.hardware.sky.command_runner import (
    _DEFAULT_CONNECT_TIMEOUT,
    KubernetesCommandRunner,
    ssh_options_list,
    SSHCommandRunner,
    SshMode,
)

from runhouse.resources.hardware.utils import _ssh_base_command

logger = get_logger(__name__)

# Get rid of the constant "Found credentials in shared credentials file: ~/.aws/credentials" message
try:
    import boto3

    boto3.set_stream_logger(name="botocore.credentials", level=logging.ERROR)
except ImportError:
    pass


def get_docker_user(cluster: "Cluster", ssh_creds: Dict) -> str:
    """Find docker container username."""
    runner = SkySSHRunner(
        node=(cluster.address, cluster.ssh_port),
        ssh_user=ssh_creds.get("ssh_user", None),
        ssh_private_key=ssh_creds.get("ssh_private_key", None),
        ssh_control_name=ssh_creds.get(
            "ssh_control_name", f"{cluster.address}:{cluster.ssh_port}"
        ),
    )
    container_name = DEFAULT_DOCKER_CONTAINER_NAME
    whoami_returncode, whoami_stdout, whoami_stderr = runner.run(
        f"sudo docker exec {container_name} whoami",
        stream_logs=True,
        require_outputs=True,
    )
    assert whoami_returncode == 0, (
        f"Failed to get docker container user. Return "
        f"code: {whoami_returncode}, Error: {whoami_stderr}"
    )
    docker_user = whoami_stdout.strip()
    logger.debug(f"Docker container user: {docker_user}")
    return docker_user


class SkySSHRunner(SSHCommandRunner):
    def __init__(
        self,
        node: Tuple[str, int],
        ssh_user: Optional[str] = None,
        ssh_private_key: Optional[str] = None,
        ssh_control_name: Optional[str] = "__default__",
        ssh_proxy_command: Optional[str] = None,
        docker_user: Optional[str] = None,
        disable_control_master: Optional[bool] = False,
        local_bind_port: Optional[int] = None,
        use_docker_exec: Optional[bool] = False,
    ):
        super().__init__(
            node,
            ssh_user,
            ssh_private_key,
            ssh_control_name,
            ssh_proxy_command,
            docker_user,
            disable_control_master,
        )

        # RH modified
        self.docker_user = docker_user
        self.local_bind_port = local_bind_port
        self.remote_bind_port = None
        self.use_docker_exec = use_docker_exec

    def _ssh_base_command(
        self,
        *,
        ssh_mode: SshMode,
        port_forward: Optional[List[int]],
        connect_timeout: Optional[int] = None,
    ) -> List[str]:
        docker_ssh_proxy_command = (
            self._docker_ssh_proxy_command(["ssh"])
            if self._docker_ssh_proxy_command
            else None
        )

        return _ssh_base_command(
            address=self.ip,
            ssh_user=self.ssh_user,
            ssh_private_key=self.ssh_private_key,
            ssh_control_name=self.ssh_control_name,
            ssh_proxy_command=self._ssh_proxy_command,
            ssh_port=self.port,
            docker_ssh_proxy_command=docker_ssh_proxy_command,
            disable_control_master=self.disable_control_master,
            ssh_mode=ssh_mode,
            port_forward=port_forward,
            connect_timeout=connect_timeout,
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
        connect_timeout: Optional[int] = None,
        source_bashrc: bool = True,  # RH MODIFIED
        skip_lines: int = 0,
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
                        connect_timeout: timeout in seconds for the ssh connection.
            source_bashrc: Whether to source the bashrc before running the
                command.
            skip_lines: The number of lines to skip at the beginning of the
                output. This is used when the output is not processed by
                SkyPilot but we still want to get rid of some warning messages,
                such as SSH warnings.
            return_cmd: If True, return the command string instead of running it.
            quiet_ssh: If True, do not print the OpenSSH outputs (i.e. add "-q" option to ssh).


        Returns:
            returncode
            or
            A tuple of (returncode, stdout, stderr).
        """
        base_ssh_command = self._ssh_base_command(
            ssh_mode=ssh_mode,
            port_forward=port_forward,
            connect_timeout=connect_timeout,
        )
        if ssh_mode == SshMode.LOGIN:
            assert isinstance(cmd, list), "cmd must be a list for login mode."
            command = base_ssh_command + cmd
            proc = subprocess_utils.run(command, shell=False, check=False)
            return proc.returncode, "", ""

        if quiet_ssh:  # RH MODIFIED
            base_ssh_command.append("-q")

        if self.use_docker_exec:  # RH MODIFIED
            cmd = " ".join(cmd) if isinstance(cmd, list) else cmd
            cmd = f"sudo docker exec {DEFAULT_DOCKER_CONTAINER_NAME} bash -c {shlex.quote(cmd)}"
        elif self.docker_user:
            cmd = " ".join(cmd) if isinstance(cmd, list) else cmd
            cmd = f"conda deactivate && {cmd}"

        command_str = self._get_command_to_run(
            cmd,
            process_stream,
            separate_stderr,
            skip_lines=skip_lines,
            source_bashrc=source_bashrc,
        )
        command = base_ssh_command + [shlex.quote(command_str)]

        log_dir = os.path.expanduser(os.path.dirname(log_path))
        os.makedirs(log_dir, exist_ok=True)

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
        logger.debug(f"Running command: {' '.join(command)}")
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
        return_cmd: bool = False,  # RH MODIFIED
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
        rsh_option = f"ssh {ssh_options}"
        return self._rsync(
            source,
            target,
            node_destination=f"{self.ssh_user}@{self.ip}",
            up=up,
            rsh_option=rsh_option,
            log_path=log_path,
            stream_logs=stream_logs,
            max_retry=max_retry,
            filter_options=filter_options,
            return_cmd=return_cmd,
        )


class SkyKubernetesRunner(KubernetesCommandRunner):
    def __init__(
        self,
        node: Tuple[str, str],
        docker_user: Optional[str] = None,
        **kwargs,
    ):
        del kwargs
        super().__init__(node)
        self.namespace, self.pod_name = node
        self.docker_user = docker_user

    def run(
        self,
        cmd: Union[str, List[str]],
        *,
        port_forward: Optional[List[int]] = None,
        require_outputs: bool = False,
        # Advanced options.
        log_path: str = os.devnull,
        # If False, do not redirect stdout/stderr to optimize performance.
        process_stream: bool = True,
        stream_logs: bool = True,
        ssh_mode: SshMode = SshMode.NON_INTERACTIVE,
        separate_stderr: bool = False,
        connect_timeout: Optional[int] = None,
        source_bashrc: bool = True,  # RH MODIFIED
        skip_lines: int = 0,
        return_cmd: bool = False,  # RH MODIFIED
        quiet_ssh: bool = None,  # RH MODIFIED
        **kwargs,
    ) -> Union[int, Tuple[int, str, str]]:
        """Uses 'kubectl exec' to run 'cmd' on a pod by its name and namespace.
        Args:
            cmd: The command to run.
            port_forward: This should be None for k8s.
            Advanced options:
            require_outputs: Whether to return the stdout/stderr of the command.
            log_path: Redirect stdout/stderr to the log_path.
            stream_logs: Stream logs to the stdout/stderr.
            check: Check the success of the command.
            ssh_mode: The mode to use for ssh.
                See SSHMode for more details.
            separate_stderr: Whether to separate stderr from stdout.
            connect_timeout: timeout in seconds for the pod connection.
            source_bashrc: Whether to source the bashrc before running the
                command.
            skip_lines: The number of lines to skip at the beginning of the
                output. This is used when the output is not processed by
                SkyPilot but we still want to get rid of some warning messages,
                such as SSH warnings.
        Returns:
            returncode
            or
            A tuple of (returncode, stdout, stderr).
        """
        # TODO(zhwu): implement port_forward for k8s.
        assert port_forward is None, (
            "port_forward is not supported for k8s " f"for now, but got: {port_forward}"
        )
        if connect_timeout is None:
            connect_timeout = _DEFAULT_CONNECT_TIMEOUT
        kubectl_args = [
            "--pod-running-timeout",
            f"{connect_timeout}s",
            "-n",
            self.namespace,
            self.pod_name,
        ]
        if ssh_mode == SshMode.LOGIN:
            assert isinstance(cmd, list), "cmd must be a list for login mode."
            base_cmd = ["kubectl", "exec", "-it", *kubectl_args, "--"]
            command = base_cmd + cmd
            proc = subprocess_utils.run(command, shell=False, check=False)
            return proc.returncode, "", ""

        kubectl_base_command = ["kubectl", "exec"]

        if ssh_mode == SshMode.INTERACTIVE:
            kubectl_base_command.append("-i")
        kubectl_base_command += [*kubectl_args, "--"]

        if self.docker_user:  # RH MODIFIED
            cmd = " ".join(cmd) if isinstance(cmd, list) else cmd
            cmd = f"conda deactivate && {cmd}"

        command_str = self._get_command_to_run(
            cmd,
            process_stream,
            separate_stderr,
            skip_lines=skip_lines,
            source_bashrc=source_bashrc,
        )
        command = kubectl_base_command + [
            # It is important to use /bin/bash -c here to make sure we quote the
            # command to be run properly. Otherwise, directly appending commands
            # after '--' will not work for some commands, such as '&&', '>' etc.
            "/bin/bash",
            "-c",
            shlex.quote(command_str),
        ]

        log_dir = os.path.expanduser(os.path.dirname(log_path))
        os.makedirs(log_dir, exist_ok=True)

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

    # @timeline.event
    def rsync(
        self,
        source: str,
        target: str,
        *,
        up: bool,
        # Advanced options.
        log_path: str = os.devnull,
        stream_logs: bool = True,
        max_retry: int = 1,
        filter_options: bool = False,  # RH MODIFIED
        return_cmd: bool = False,  # RH MODIFIED
    ) -> None:
        """Uses 'rsync' to sync 'source' to 'target'.
        Args:
            source: The source path.
            target: The target path.
            up: The direction of the sync, True for local to cluster, False
              for cluster to local.
            log_path: Redirect stdout/stderr to the log_path.
            stream_logs: Stream logs to the stdout/stderr.
            max_retry: The maximum number of retries for the rsync command.
              This value should be non-negative.
        Raises:
            exceptions.CommandError: rsync command failed.
        """

        def get_remote_home_dir() -> str:
            # Use `echo ~` to get the remote home directory, instead of pwd or
            # echo $HOME, because pwd can be `/` when the remote user is root
            # and $HOME is not always set.
            rc, remote_home_dir, stderr = self.run(
                "echo ~", require_outputs=True, separate_stderr=True, stream_logs=False
            )
            if rc != 0:
                raise ValueError(
                    "Failed to get remote home directory: "
                    f"{remote_home_dir + stderr}"
                )
            remote_home_dir = remote_home_dir.strip()
            return remote_home_dir

        # Build command.
        helper_path = os.path.join(
            os.path.abspath(os.path.dirname(__file__)), "kubernetes", "rsync_helper.sh"
        )
        self._rsync(
            source,
            target,
            node_destination=f"{self.pod_name}@{self.namespace}",
            up=up,
            rsh_option=helper_path,
            log_path=log_path,
            stream_logs=stream_logs,
            max_retry=max_retry,
            prefix_command=f"chmod +x {helper_path} && ",
            # rsync with `kubectl` as the rsh command will cause ~/xx parsed as
            # /~/xx, so we need to replace ~ with the remote home directory. We
            # only need to do this when ~ is at the beginning of the path.
            get_remote_home_dir=get_remote_home_dir,
            filter_options=filter_options,
            return_cmd=return_cmd,
        )
