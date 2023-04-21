from pathlib import Path
from typing import Dict

import yaml

from runhouse import rh_config


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


def _get_cluster_from(system):
    from runhouse.rns import Resource

    if isinstance(system, Resource) or (
        isinstance(system, str)
        and not rh_config.rns_client.exists(system, resource_type="cluster")
    ):
        return system

    from runhouse.rns.hardware import Cluster

    if isinstance(system, Dict):
        return Cluster.from_config(system)
    elif isinstance(system, str) and rh_config.rns_client.exists(
        system, resource_type="cluster"
    ):
        return Cluster.from_name(system)
    return system
