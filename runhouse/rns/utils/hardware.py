from pathlib import Path
from typing import Dict

import yaml

from runhouse import rh_config

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
    from runhouse.rns import Resource

    if isinstance(system, Resource) and system.RESOURCE_TYPE == "cluster":
        return system
    if isinstance(system, str):
        if system in RESERVED_SYSTEM_NAMES or not rh_config.rns_client.exists(
            system, resource_type="cluster"
        ):
            return system

    from runhouse.rns.hardware import Cluster

    if isinstance(system, Dict):
        return Cluster.from_config(system, dryrun)
    elif isinstance(system, str) and rh_config.rns_client.exists(
        system, resource_type="cluster"
    ):
        return Cluster.from_name(system, dryrun)
    return system
