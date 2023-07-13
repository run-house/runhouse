from typing import List, Optional, Union

from ..utils.hardware import RESERVED_SYSTEM_NAMES

from .cluster import Cluster
from .on_demand_cluster import OnDemandCluster
from .slurm_cluster import SlurmCluster


# Cluster factory method
def cluster(
    name: str,
    ips: List[str] = None,
    ssh_creds: Optional[dict] = None,
    dryrun: bool = False,
    **kwargs,
) -> Union[Cluster, OnDemandCluster]:
    """
    Builds an instance of :class:`Cluster`.

    Args:
        name (str): Name for the cluster, to re-use later on.
        ips (List[str], optional): List of IP addresses for the BYO cluster.
        ssh_creds (dict, optional): Dictionary mapping SSH credentials.
            Example: ``ssh_creds={'ssh_user': '...', 'ssh_private_key':'<path_to_key>'}``
        dryrun (bool): Whether to create the Cluster if it doesn't exist, or load a Cluster object as a dryrun.
            (Default: ``False``)

    Returns:
        Union[Cluster, OnDemandCluster]: The resulting cluster.

    Example:
        >>> import runhouse as rh
        >>> gpu = rh.cluster(ips=['<ip of the cluster>'],
        >>>                  ssh_creds={'ssh_user': '...', 'ssh_private_key':'<path_to_key>'},
        >>>                  name='rh-a10x').save()

        >>> # Load cluster from above
        >>> reloaded_cluster = rh.cluster(name="rh-a10x")
    """
    if name and ips is None and ssh_creds is None and not kwargs:
        # If only the name is provided and dryrun is set to True
        return Cluster.from_name(name, dryrun)

    if name in RESERVED_SYSTEM_NAMES:
        raise ValueError(
            f"Cluster name {name} is a reserved name. Please use a different name which is not one of "
            f"{RESERVED_SYSTEM_NAMES}."
        )

    if "instance_type" in kwargs.keys():
        # Commenting out for now. If two creation paths creates confusion let's push people to use
        # ondemand_cluster() instead.
        # warnings.warn(
        #     "The `cluster` factory is intended to be used for static clusters. "
        #     "If you would like to create an on-demand cluster, please use `rh.ondemand_cluster()` instead."
        # )
        return ondemand_cluster(name=name, **kwargs)

    return Cluster(ips=ips, ssh_creds=ssh_creds, name=name, dryrun=dryrun)


# OnDemandCluster factory method
def ondemand_cluster(
    name: str,
    instance_type: Optional[str] = None,
    num_instances: Optional[int] = None,
    provider: Optional[str] = None,
    autostop_mins: Optional[int] = None,
    use_spot: bool = False,
    image_id: Optional[str] = None,
    region: Optional[str] = None,
    dryrun: bool = False,
) -> OnDemandCluster:
    """
    Builds an instance of :class:`OnDemandCluster`.

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
        dryrun (bool): Whether to create the Cluster if it doesn't exist, or load a Cluster object as a dryrun.
            (Default: ``False``)

    Returns:
        OnDemandCluster: The resulting cluster.

    Example:
        >>> import runhouse as rh
        >>> # On-Demand SkyPilot Cluster (OnDemandCluster)
        >>> gpu = rh.ondemand_cluster(name='rh-4-a100s',
        >>>                  instance_type='A100:4',
        >>>                  provider='gcp',
        >>>                  autostop_mins=-1,
        >>>                  use_spot=True,
        >>>                  image_id='my_ami_string',
        >>>                  region='us-east-1')

        >>> # Slurm Cluster via REST API
        >>> gpu = rh.cluster(name='my_slurm_cluster',
        >>>                  api_url='http://*.**.**.***:6820',
        >>>                  api_auth_user='ubuntu',
        >>>                  api_jwt_token='eyJhbGc*******')

        >>> # Slurm Cluster via SSH
        >>> gpu = rh.cluster(name='my_slurm_cluster',
        >>>                  ssh_creds={'ssh_user': '...', 'ssh_private_key':'<path_to_key>'},
        >>>                  ips=["3.91.***.***", "3.237.**.**"],
        >>>                  partition="rhcluster",
        >>>                  log_folder="slurm_logs"
    ).save()

        >>> # Load cluster from above
        >>> reloaded_cluster = rh.cluster(name="rh-4-a100s")
    """
    if name and not any([instance_type, num_instances, provider, image_id, region]):
        # If only the name is provided and dryrun is set to True
        return Cluster.from_name(name, dryrun)

    if name in RESERVED_SYSTEM_NAMES:
        raise ValueError(
            f"Cluster name {name} is a reserved name. Please use a different name which is not one of "
            f"{RESERVED_SYSTEM_NAMES}."
        )

    return OnDemandCluster(
        instance_type=instance_type,
        provider=provider,
        num_instances=num_instances,
        autostop_mins=autostop_mins,
        use_spot=use_spot,
        image_id=image_id,
        region=region,
        name=name,
        dryrun=dryrun,
    )


def slurm_cluster(
    name: str,
    ip: str,
    ssh_creds: dict = None,
    job_params: dict = None,
    partition: str = None,
    log_folder: str = None,
) -> SlurmCluster:
    """
    Builds an instance of :class:`SlurmCluster`.

    Args:
        name (str): Name for the Cluster, to re-use later on within Runhouse.
        ip: (str): IP address for the cluster. Could be either a jump server or login node used
            for requesting compute and submitting jobs.
        partition (str, optional): Name of specific partition for the Slurm scheduler to use.
        log_folder (str, optional): Name of folder to store logs for the jobs on the cluster.
            Used to dump job information, logs and result when finished. Defaults to ``~/.rh/logs/<job_id>``.
        ssh_creds (dict, optional): Required for submitting a job via SSH and for accessing the
            node(s) running the job.
        job_params (dict, optional): Optonal params to pass to the job submission executor.
            Params include: ``timeout_min: str``, ``mem_gb: str``, ``nodes: str``, ``cpus_per_task: str``,
            ``gpus_per_node: str``, ``tasks_per_node: str``

    Returns:
        SlurmCluster: The resulting cluster.

    Example:
        >>> import runhouse as rh
        >>> cluster = rh.slurm_cluster(ips=['<ip of the cluster>'],
        >>>                  ssh_creds={'ssh_user': '...', 'ssh_private_key':'<path_to_key>'},
        >>>                  name='slurm_cluster').save()

        >>> # Load cluster from above
        >>> reloaded_cluster = rh.slurm_cluster(name="rh-a10x")
    """
    if name and not any([partition, log_folder, ssh_creds, ip, job_params]):
        # Try reloading existing cluster
        return SlurmCluster.from_name(name, dryrun=True)

    if name in RESERVED_SYSTEM_NAMES:
        raise ValueError(
            f"Cluster name {name} is a reserved name. Please use a different name which is not one of "
            f"{RESERVED_SYSTEM_NAMES}."
        )

    return SlurmCluster(
        ip=ip,
        ssh_creds=ssh_creds,
        name=name,
        partition=partition,
        log_folder=log_folder,
        job_params=job_params,
    )
