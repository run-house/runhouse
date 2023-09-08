import logging
import os
import shlex
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import sky.utils.command_runner as cr

import yaml

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
    ):
        super().__init__(
            ip, ssh_user, ssh_private_key, ssh_control_name, ssh_proxy_command
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
                ssh_proxy_command=self._ssh_proxy_command,
            )
            + [f"{self.ssh_user}@{self.ip}"]
        )

    def tunnel(self, local_port, remote_port):
        base_cmd = self._ssh_base_command(
            ssh_mode=SshMode.NON_INTERACTIVE, port_forward=[(local_port, remote_port)]
        )
        command = " ".join(base_cmd + ["tail"])
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
