import warnings
from typing import Dict, List, Optional, Union

from ..utils.hardware import RESERVED_SYSTEM_NAMES

from .cluster import Cluster
from .on_demand_cluster import OnDemandCluster
from .sagemaker_cluster import SageMakerCluster


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
        >>>                  region='us-east-1',
        >>>                  ).save()

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


def sagemaker_cluster(
    name: str,
    arn_role: str = None,
    estimator: Union["sagemaker.estimator.EstimatorBase", Dict] = None,
    connection_wait_time: int = None,
    dryrun: bool = False,
) -> SageMakerCluster:
    """
    Builds an instance of :class:`SageMakerCluster`.

    Args:
        name (str): Name for the cluster, to re-use later on.
        arn_role (str, optional): AWS IAM role ARN to use for connecting to the cluster.
        estimator (Union[str, "sagemaker.estimator.EstimatorBase"], optional): Estimator to use for the job.
        connection_wait_time (int, optional): Amount of time the SSH helper will wait inside SageMaker before
            it continues normal execution. Useful if you want to connect before the job starts (e.g. training).
            If you don't want to wait, set it to 0.
        dryrun (bool): Whether to create the SageMakerCluster if it doesn't exist, or load
            a SageMakerCluster object as a dryrun.
            (Default: ``False``)

    Returns:
        SageMakerCluster: The resulting cluster.

    Example:
        >>> import runhouse as rh
        >>> c = rh.sagemaker_cluster(name='sagemaker-cluster',
        >>>                          arn_role='arn:aws:iam::123456789012:role/MySageMakerRole',
        >>>                          estimator=PyTorch(entry_point='train.py', role=arn_role,
        >>>                                            source_dir='/Users/myuser/dev/sagemaker',
        >>>                                            framework_version='1.8.1', py_version='py36',
        >>>                                            instance_type='ml.p3.2xlarge'),
        >>>                          ).save()

        >>> # Load cluster from above
        >>> reloaded_cluster = rh.cluster(name="sagemaker-cluster")
    """
    if name and not any([arn_role, estimator, connection_wait_time]):
        # If only the name is provided and dryrun is set to True
        return Cluster.from_name(name, dryrun)

    if name in RESERVED_SYSTEM_NAMES:
        raise ValueError(
            f"Cluster name {name} is a reserved name. Please use a different name which is not one of "
            f"{RESERVED_SYSTEM_NAMES}."
        )

    # TODO [JL] Manually add the SageMaker SSH helper to the train script?
    warnings.warn(
        "Please make sure the SageMaker SSH helper is included in the estimator's script."
        "To do so, add the below lines at the top of the file:"
        "\n\nimport sagemaker_ssh_helper\nsagemaker_ssh_helper.setup_and_start_ssh()\n\n"
        "For more information and examples: "
        "https://github.com/aws-samples/sagemaker-ssh-helper#step-3-modify-your-training-script"
    )

    return SageMakerCluster(
        name=name,
        arn_role=arn_role,
        estimator=estimator,
        connection_wait_time=connection_wait_time,
        dryrun=dryrun,
    )
