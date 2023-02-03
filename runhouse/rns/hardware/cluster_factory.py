from typing import List, Optional, Union

from runhouse.rh_config import rns_client

from .cluster import Cluster
from .skycluster import SkyCluster


# Cluster factory method
def cluster(
    name: str,
    instance_type: Optional[str] = None,
    num_instances: Optional[int] = None,
    provider: Optional[str] = None,
    autostop_mins: Optional[int] = None,
    use_spot: Optional[bool] = None,
    image_id: Optional[str] = None,
    region: Optional[str] = None,
    ips: List[str] = None,
    ssh_creds: Optional[dict] = None,
    dryrun: Optional[bool] = False,
) -> Union[Cluster, SkyCluster]:
    config = rns_client.load_config(name)
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

    new_cluster = SkyCluster.from_config(config, dryrun=dryrun)

    return new_cluster
