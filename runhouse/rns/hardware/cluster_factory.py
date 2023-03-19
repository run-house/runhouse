from typing import List, Optional, Union

from runhouse.rh_config import rns_client

from .cluster import Cluster
from .on_demand_cluster import OnDemandCluster


# Cluster factory method
def cluster(
    name: str,
    instance_type: Optional[str] = None,
    num_instances: Optional[int] = None,
    provider: Optional[str] = None,
    autostop_mins: Optional[int] = None,
    use_spot: bool = False,
    image_id: Optional[str] = None,
    region: Optional[str] = None,
    ips: List[str] = None,
    ssh_creds: Optional[dict] = None,
    dryrun: bool = False,
    load: bool = True,
) -> Union[Cluster, OnDemandCluster]:
    """
    Builds an instance of :class:`Cluster`.

    Args:
        name (str): Name for the cluster, to re-use later on.
        instance_type (int, optional): Type of cloud instance to use for the cluster. This could
            be a Runhouse built-in type, or your choice of instance type.
        num_instances (int, optional): Number of instances to use for the cluster.
        provider (str, optional): Cloud provider to use for the cluster.
        autostop_mins (int, optional): Number of minutes to keep the cluster up after inactivity,
            or ``-1`` to keep cluster up indefinitely.
        use_spot (bool, optional): Whether or not to use spot instance.
        image_id (str, optional): Custom image ID for the cluster.
        region (str, optional): The region to use for the cluster.
        ips (List[str], optional): List of IP addresses for the BYO cluster.
        ssh_creds (dict, optional): Dictionary mapping SSH credentials.
            Example: ``ssh_creds={'ssh_user': '...', 'ssh_private_key':'<path_to_key>'}``
        dryrun (bool): Whether to create the Cluster if it doesn't exist, or load a Cluster object as a dryrun.
            (Default: ``False``)
        load (bool): Whether to load an existing config for the Cluster. (Default: ``True``)

    Returns:
        Cluster or OnDemandCluster: The resulting cluster.

    Example:
        >>> # BYO Cluster
        >>> gpu = rh.cluster(ips=['<ip of the cluster>'],
        >>>          ssh_creds={'ssh_user': '...', 'ssh_private_key':'<path_to_key>'},
        >>>          name='rh-a10x')

        >>> # On-Demand SkyPilot Cluster (OnDemandCluster)
        >>> gpu = rh.cluster(name='rh-4-a100s',
        >>>                  instance_type='A100:4',
        >>>                  provider='gcp',
        >>>                  autostop_mins=-1,
        >>>                  use_spot=True,
        >>>                  image_id='my_ami_string',
        >>>                  region='us-east-1',
        >>>                  )
    """
    config = rns_client.load_config(name) if load else {}
    config["name"] = name or config.get("rns_address", None) or config.get("name")
    config["ips"] = ips or config.get("ips", None)
    # ssh creds should only be in Secrets management, not in config
    config["ssh_creds"] = ssh_creds or config.get("ssh_creds", None)
    if config["ips"]:
        return Cluster.from_config(config, dryrun=dryrun)

    config["instance_type"] = instance_type or config.get("instance_type", None)
    config["num_instances"] = num_instances or config.get("num_instances", None)
    config["provider"] = provider or config.get("provider", None)
    config["autostop_mins"] = (
        autostop_mins
        if autostop_mins is not None
        else config.get("autostop_mins", None)
    )
    config["use_spot"] = (
        use_spot if use_spot is not None else config.get("use_spot", None)
    )
    config["image_id"] = (
        image_id if image_id is not None else config.get("image_id", None)
    )
    config["region"] = region if region is not None else config.get("region", None)

    new_cluster = OnDemandCluster.from_config(config, dryrun=dryrun)

    return new_cluster
