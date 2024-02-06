import json
from enum import Enum
from pathlib import Path
from typing import Dict, Optional

from sshtunnel import SSHTunnelForwarder

from runhouse.constants import CLUSTER_CONFIG_PATH, RESERVED_SYSTEM_NAMES
from runhouse.globals import ssh_tunnel_cache

# TODO: Move the following two functions into a networking module
def get_open_tunnel(address: str, ssh_port: int) -> Optional[SSHTunnelForwarder]:
    if (address, ssh_port) in ssh_tunnel_cache:
        ssh_tunnel = ssh_tunnel_cache[(address, ssh_port)]
        if isinstance(ssh_tunnel, SSHTunnelForwarder):
            # Initializes tunnel_is_up dictionary
            ssh_tunnel.check_tunnels()

            if (
                ssh_tunnel.is_active
                and ssh_tunnel.tunnel_is_up[ssh_tunnel.local_bind_address]
            ):
                return ssh_tunnel

            else:
                # If the tunnel is no longer active or up, pop it from the global cache
                ssh_tunnel_cache.pop((address, ssh_port))

    return None


def cache_open_tunnel(
    address: str,
    ssh_port: str,
    ssh_tunnel: SSHTunnelForwarder,
):
    ssh_tunnel_cache[(address, ssh_port)] = ssh_tunnel


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


def cluster_config_file_exists() -> bool:
    return Path(CLUSTER_CONFIG_PATH).expanduser().exists()


def load_cluster_config_from_file() -> Dict:
    if cluster_config_file_exists():
        with open(Path(CLUSTER_CONFIG_PATH).expanduser()) as f:
            cluster_config = json.load(f)
        return cluster_config
    else:
        return {}


def _current_cluster(key="config"):
    """Retrive key value from the current cluster config.
    If key is "config", returns entire config."""
    from runhouse.globals import obj_store

    cluster_config = obj_store.get_cluster_config()
    if cluster_config:
        # This could be a local cluster started via runhouse start,
        # in which case it would have no Name.
        if key in ["cluster_name", "name"] and "name" not in cluster_config:
            return None
        if key == "config":
            return cluster_config
        if key == "cluster_name":
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
