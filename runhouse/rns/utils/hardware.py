from pathlib import Path
from typing import Dict

import yaml

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
    from runhouse.rns.hardware import Cluster

    if isinstance(system, Cluster):
        return system
    if system in RESERVED_SYSTEM_NAMES:
        return system

    if isinstance(system, Dict):
        return Cluster.from_config(system, dryrun)

    if isinstance(system, str):
        try:
            system = Cluster.from_name(name=system, dryrun=dryrun)
        except ValueError:
            # Name not found in RNS. Doing the lookup this way saves us a hop to RNS
            pass

    return system
