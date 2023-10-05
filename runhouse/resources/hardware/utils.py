import logging
import os
import shlex
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import sky.utils.command_runner as cr

import yaml
from sky.skylet import log_lib

from sky.utils.command_runner import ssh_options_list, SSHCommandRunner, SshMode

logger = logging.getLogger(__name__)

RESERVED_SYSTEM_NAMES = ["file", "s3", "gs", "azure", "here", "ssh", "sftp"]


def _current_cluster(key="name"):
    """Retrive key value from the current cluster config.
    If key is "config", returns entire config."""
    if Path("~/.rh/cluster_config.yaml").expanduser().exists():
        with open(Path("~/.rh/cluster_config.yaml").expanduser()) as f:
            cluster_config = yaml.safe_load(f)
        if key == "config":
            return cluster_config
        elif key == "cluster_name":
            return cluster_config["name"].rsplit("/", 1)[-1]
        return cluster_config[key]
    else:
        return None


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
        ssh_control_name=None,
        ssh_proxy_command=None,
        disable_control_master: Optional[bool] = False
    ):
        super().__init__(
            ip, ssh_user, ssh_private_key, ssh_control_name, ssh_proxy_command, disable_control_master
        )
        self._tunnel_procs = []

    def _ssh_base_command(
        self,
        *,
        ssh_mode: SshMode,
        port_forward: Union[None, List[int], List[Tuple[int, int]]],
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
            for fwd in port_forward:
                if isinstance(fwd, int):
                    local, remote = fwd, fwd
                else:
                    local, remote = fwd
                logger.info(f"Forwarding port {local} to port {remote} on localhost.")
                ssh += ["-L", f"{remote}:localhost:{local}"]
        return (
            ssh
            + ssh_options_list(
                self.ssh_private_key,
                self.ssh_control_name,
                port=None,
                ssh_proxy_command=self._ssh_proxy_command,
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
        **kwargs,
    ) -> Union[int, Tuple[int, str, str]]:
        """This is identical to the SkyPilot command runner, other than logging the full command to be run
        before running it."""
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

        # This log should be the only difference between our command
        logging.info(f"Running command: {' '.join(command)}")
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
        filter_options: Optional[str] = None,
        log_path: str = os.devnull,
        stream_logs: bool = True,
        max_retry: int = 1,
    ) -> None:
        # temp update rsync filters to exclude docs, when syncing over runhouse folder
        org_rsync_filter = cr.RSYNC_FILTER_OPTION
        if filter_options:
            cr.RSYNC_FILTER_OPTION = f"--filter='{filter_options}'"
        super().rsync(
            source,
            target,
            up=up,
            log_path=log_path,
            stream_logs=stream_logs,
            max_retry=max_retry,
        )
        if filter_options:
            cr.RSYNC_FILTER_OPTION = org_rsync_filter
