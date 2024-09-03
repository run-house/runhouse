import copy
import shlex
import subprocess
import time
from typing import Dict, Optional

from runhouse.constants import LOCALHOST, TUNNEL_TIMEOUT
from runhouse.globals import ssh_tunnel_cache

from runhouse.logger import get_logger
from runhouse.resources.hardware.sky.command_runner import SshMode
from runhouse.resources.hardware.sky.constants import DEFAULT_DOCKER_PORT
from runhouse.resources.hardware.utils import (
    _docker_ssh_proxy_command,
    _generate_ssh_control_hash,
    _ssh_base_command,
)

logger = get_logger(__name__)


class SshTunnel:
    def __init__(
        self,
        ip: str,
        ssh_user: str = None,
        ssh_private_key: str = None,
        ssh_control_name: Optional[str] = "__default__",
        ssh_proxy_command: Optional[str] = None,
        ssh_port: int = 22,
        disable_control_master: Optional[bool] = False,
        docker_user: Optional[str] = None,
        cloud: Optional[str] = None,
    ):
        """Initialize an ssh tunnel from a remote server to localhost

        Args:
            ip (str): The address of the server we are trying to port forward an address to our local machine with.
            ssh_user (str, optional): The SSH username to use for connecting to the remote server. Defaults to None.
            ssh_private_key (str, optional): The path to the SSH private key file. Defaults to None.
            ssh_control_name (str, optional): The name for the SSH control connection.
                Defaults to `"__default__"`.
            ssh_proxy_command (str, optional): The SSH proxy command to use for connecting to the remote
                server. Defaults to None.
            ssh_port (int, optional): The port on the remote machine where the SSH server is running. Defaults to 22.
            disable_control_master (bool, optional): Whether to disable SSH ControlMaster. Defaults to False.
            docker_user (str, optional): The Docker username to use if connecting through Docker. Defaults to None.
            cloud (str, optional): The cloud provider, if applicable. Defaults to None.
        """
        self.ip = ip
        self.ssh_port = ssh_port
        self.ssh_private_key = ssh_private_key
        self.ssh_control_name = (
            None
            if ssh_control_name is None
            else _generate_ssh_control_hash(ssh_control_name)
        )
        self.ssh_proxy_command = ssh_proxy_command
        self.disable_control_master = disable_control_master

        if docker_user:
            self.ssh_user = docker_user
            self.docker_ssh_proxy_command = _docker_ssh_proxy_command(
                ip, ssh_user, ssh_private_key
            )(["ssh"])

            if cloud != "kubernetes":
                self.ip = "localhost"
                self.ssh_port = DEFAULT_DOCKER_PORT
        else:
            self.ssh_user = ssh_user
            self.docker_ssh_proxy_command = None

        self.tunnel_proc = None

    def tunnel(self, local_port, remote_port):
        base_cmd = _ssh_base_command(
            address=self.ip,
            ssh_user=self.ssh_user,
            ssh_private_key=self.ssh_private_key,
            ssh_control_name=self.ssh_control_name,
            ssh_proxy_command=self.ssh_proxy_command,
            ssh_port=self.ssh_port,
            docker_ssh_proxy_command=self.docker_ssh_proxy_command,
            disable_control_master=self.disable_control_master,
            ssh_mode=SshMode.NON_INTERACTIVE,
            port_forward=[(local_port, remote_port)],
        )

        command = " ".join(base_cmd)
        logger.info(f"Running forwarding command: {command}")
        proc = subprocess.Popen(
            shlex.split(command),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        # Wait until tunnel is formed by trying to create a socket in a loop

        start_time = time.time()
        while not is_port_in_use(local_port):
            time.sleep(0.1)
            if time.time() - start_time > TUNNEL_TIMEOUT:
                raise ConnectionError(
                    f"Failed to create tunnel from {local_port} to {remote_port} on {self.ip}"
                )

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

            # Remove port forwarding
            port_fwd_cmd = " ".join(
                _ssh_base_command(
                    address=self.ip,
                    ssh_user=self.ssh_user,
                    ssh_private_key=self.ssh_private_key,
                    ssh_control_name=self.ssh_control_name,
                    ssh_proxy_command=self.ssh_proxy_command,
                    ssh_port=self.ssh_port,
                    docker_ssh_proxy_command=self.docker_ssh_proxy_command,
                    disable_control_master=self.disable_control_master,
                    ssh_mode=SshMode.NON_INTERACTIVE,
                    port_forward=[(self.local_bind_port, self.remote_bind_port)],
                )
            )

            if "ControlMaster" in port_fwd_cmd:
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


####################################################################################################
# Cache and retrieve existing SSH Runners that are set up for a given address and port
####################################################################################################
# TODO: Shouldn't the control master prevent new ssh connections from being created?


def is_port_in_use(port: int) -> bool:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0


def cache_existing_ssh_tunnel(address: str, ssh_port: int, tunnel: SshTunnel) -> None:
    ssh_tunnel_cache[(address, ssh_port)] = tunnel


def get_existing_sky_ssh_runner(address: str, ssh_port: int) -> Optional[SshTunnel]:
    if (address, ssh_port) in ssh_tunnel_cache:
        existing_tunnel = ssh_tunnel_cache.get((address, ssh_port))
        if existing_tunnel.tunnel_is_up():
            return existing_tunnel
        else:
            ssh_tunnel_cache.pop((address, ssh_port))
    else:
        return None


def ssh_tunnel(
    address: str,
    ssh_creds: Dict,
    local_port: int,
    ssh_port: int = 22,
    remote_port: Optional[int] = None,
    num_ports_to_try: int = 0,
    docker_user: Optional[str] = None,
    cloud: Optional[str] = None,
) -> SshTunnel:
    """Initialize an ssh tunnel from a remote server to localhost

    Args:
        address (str): The address of the server we are trying to port forward an address to our local machine with.
        ssh_creds (Dict): A dictionary of ssh credentials used to connect to the remote server.
        local_port (int): The port locally where we are attempting to bind the remote server address to.
        ssh_port (int): The port on the machine where the ssh server is running.
            This is generally port 22, but occasionally
            we may forward a container's ssh port to a different port
            on the actual machine itself (for example on a Docker VM). Defaults to 22.
        remote_port (int, optional): The port of the remote server
            we're attempting to port forward. Defaults to None.
        num_ports_to_try (int, optional): The number of local ports to attempt to bind to,
            starting at local_port and incrementing by 1 till we hit the max. Defaults to 0.
        docker_user (str, optional): The Docker username to use if connecting through Docker. Defaults to None.
        cloud (str, Optional): Cluster cloud, if an on-demand cluster.

    Returns:
        SshTunnel: The initialized tunnel.
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
    tunnel_address = address if not docker_user else "localhost"
    if (
        tunnel
        and tunnel.ip == tunnel_address
        and tunnel.remote_bind_port == remote_port
    ):
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

    ssh_credentials = copy.copy(ssh_creds)

    # Host could be a proxy specified in credentials or is the provided address
    host = ssh_credentials.pop("ssh_host", address)
    ssh_control_name = ssh_credentials.pop("ssh_control_name", f"{address}:{ssh_port}")

    tunnel = SshTunnel(
        ip=host,
        ssh_user=ssh_creds.get("ssh_user"),
        ssh_private_key=ssh_creds.get("ssh_private_key"),
        ssh_proxy_command=ssh_creds.get("ssh_proxy_command"),
        ssh_control_name=ssh_control_name,
        docker_user=docker_user,
        ssh_port=ssh_port,
        cloud=cloud,
    )
    tunnel.tunnel(local_port, remote_port)

    logger.debug(
        f"Successfully bound "
        f"{LOCALHOST}:{remote_port} via ssh port {ssh_port} "
        f"on remote server {address} "
        f"to {LOCALHOST}:{local_port} on local machine."
    )

    cache_existing_ssh_tunnel(address, ssh_port, tunnel)
    return tunnel
