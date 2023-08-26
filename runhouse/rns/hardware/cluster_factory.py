import warnings
from typing import Dict, List, Optional, Union

from ..utils.hardware import RESERVED_SYSTEM_NAMES

from .cluster import Cluster
from .on_demand_cluster import OnDemandCluster
from .sagemaker_cluster import SageMakerCluster


# Cluster factory method
def cluster(
    name: str,
    host: Union[str, List[str]] = None,
    ssh_creds: Optional[dict] = None,
    dryrun: bool = False,
    **kwargs,
) -> Union[Cluster, OnDemandCluster, SageMakerCluster]:
    """
    Builds an instance of :class:`Cluster`.

    Args:
        name (str): Name for the cluster, to re-use later on.
        host (str or List[str], optional): Hostname, IP address, or list of IP addresses for the BYO cluster.
        ssh_creds (dict, optional): Dictionary mapping SSH credentials.
            Example: ``ssh_creds={'ssh_user': '...', 'ssh_private_key':'<path_to_key>'}``
        dryrun (bool): Whether to create the Cluster if it doesn't exist, or load a Cluster object as a dryrun.
            (Default: ``False``)

    Returns:
        Union[Cluster, OnDemandCluster, SageMakerCluster]: The resulting cluster.

    Example:
        >>> # using private key
        >>> gpu = rh.cluster(host='<hostname>',
        >>>                  ssh_creds={'ssh_user': '...', 'ssh_private_key':'<path_to_key>'},
        >>>                  name='rh-a10x').save()

        >>> # using password
        >>> gpu = rh.cluster(host='<hostname>',
        >>>                  ssh_creds={'ssh_user': '...', 'password':'*****'},
        >>>                  name='rh-a10x').save()

        >>> # Load cluster from above
        >>> reloaded_cluster = rh.cluster(name="rh-a10x")
    """
    if "ips" in kwargs:
        host = kwargs["ips"]
        warnings.warn(
            "``ips`` argument has been deprecated. Please use ``host`` to refer to the cluster IPs or host instead."
        )

    if name and host is None and ssh_creds is None and not kwargs:
        # If only the name is provided and dryrun is set to True
        return Cluster.from_name(name, dryrun)

    if name in RESERVED_SYSTEM_NAMES:
        raise ValueError(
            f"Cluster name {name} is a reserved name. Please use a different name which is not one of "
            f"{RESERVED_SYSTEM_NAMES}."
        )

    if "instance_type" in kwargs.keys():
        warnings.warn(
            "The `cluster` factory is intended to be used for static clusters. "
            "If you would like to create an on-demand cluster, please use `rh.ondemand_cluster()` instead."
        )
        return ondemand_cluster(name=name, **kwargs)

    if any(
        k in kwargs.keys()
        for k in [
            "role",
            "estimator",
            "instance_type",
            "autostop_mins",
            "connection_wait_time",
            "instance_count",
        ]
    ):
        warnings.warn(
            "The `cluster` factory is intended to be used for static clusters. "
            "If you would like to create a sagemaker cluster, please use `rh.sagemaker_cluster()` instead."
        )
        return sagemaker_cluster(name=name, **kwargs)

    return Cluster(ips=host, ssh_creds=ssh_creds, name=name, dryrun=dryrun)


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
        >>> reloaded_cluster = rh.ondemand_cluster(name="rh-4-a100s")
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
    role: str = None,
    profile: str = None,
    ssh_key_path: str = None,
    instance_type: str = None,
    instance_count: int = None,
    image_uri: str = None,
    autostop_mins: int = None,
    connection_wait_time: int = None,
    estimator: Union["sagemaker.estimator.EstimatorBase", Dict] = None,
    job_name: str = None,
    dryrun: bool = False,
) -> SageMakerCluster:
    """
    Builds an instance of :class:`SageMakerCluster`.

    Args:
        name (str): Name for the cluster, to re-use later on.
        role (str, optional): An AWS IAM role (either name or full ARN).
            Can be passed in explicitly as an argument or provided via an estimator. If not specified will try
            using the ``profile`` attribute or environment variable ``AWS_PROFILE`` to extract the relevant role ARN.
            More info on configuring an IAM role for SageMaker
            `here <https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-roles.html>`_.
        profile (str, optional): AWS profile to use for the cluster. If provided instead of a ``role``, will lookup
            the role ARN associated with the profile in the local AWS credentials.
            If not provided, will use the default profile.
        ssh_key_path (str, optional): Path to SSH key to use for connecting to the cluster. If not provided, will
            first look for the SageMaker default key store in path ``~/.ssh/sagemaker-ssh-gw`` before creating
            a new one.
        instance_type (str, optional): Type of AWS instance to use for the cluster. More info on supported
            instance options `here <https://aws.amazon.com/sagemaker/pricing/instance-types>`_.
            (Default: ``ml.m5.large``.)
        instance_count (int, optional): Number of instances to use for the cluster.
            (Default: ``1``.)
        image_uri (str, optional): Image to use for the cluster instead of using the default SageMaker image which
            will be based on the framework_version and py_version. Can be an ECR url or dockerhub image and tag.
        estimator (Union[str, sagemaker.estimator.EstimatorBase], optional): Estimator to use for a dedicated
            training job. Leave as ``None`` if launching the compute without running a dedicated job.
            More info on creating an estimator `here
            <https://sagemaker.readthedocs.io/en/stable/frameworks/pytorch/using_pytorch.html#create-an-estimator>`_.
        autostop_mins (int, optional): Number of minutes to keep the cluster up after inactivity,
            or ``-1`` to keep cluster up indefinitely. Note that this will keep the cluster up even if a dedicated
            job has finished running or failed.
        connection_wait_time (int, optional): Amount of time to wait inside the SageMaker cluster before
            continuing with normal execution. Useful if you want to connect before a dedicated job starts
            (e.g. training). If you don't want to wait, set it to ``0``.
            If no estimator is provided, will default to ``0``.
        job_name (str, optional): Name to provide for a training job. Only relevant if an estimator is provided.
            If not provided with an estimator, will generate a default job name based on the training
            image name and current timestamp.
        dryrun (bool): Whether to create the SageMakerCluster if it doesn't exist, or load a SageMakerCluster object
            as a dryrun.
            (Default: ``False``)

    Returns:
        SageMakerCluster: The resulting cluster.

    Example:
        >>> import runhouse as rh
        >>> # Launch a new SageMaker instance and keep it up indefinitely.
        >>> # Note: This will use Role ARN associated with the "sagemaker" profile defined in the local aws credentials
        >>> c = rh.sagemaker_cluster(name='sm-cluster', profile="sagemaker").save()

        >>> # Running a training job with a provided Estimator
        >>> c = rh.sagemaker_cluster(name='sagemaker-cluster',
        >>>                          estimator=PyTorch(entry_point='train.py',
        >>>                                            role='arn:aws:iam::123456789012:role/MySageMakerRole',
        >>>                                            source_dir='/Users/myuser/dev/sagemaker',
        >>>                                            framework_version='1.8.1',
        >>>                                            py_version='py36',
        >>>                                            instance_type='ml.p3.2xlarge'),
        >>>                          ).save()

        >>> # Load cluster from above
        >>> reloaded_cluster = rh.sagemaker_cluster(name="sagemaker-cluster")
    """
    if name:
        alt_options = dict(
            role=role,
            profile=profile,
            ssh_key_path=ssh_key_path,
            image_uri=image_uri,
            estimator=estimator,
            instance_type=instance_type,
            autostop_mins=autostop_mins,
            job_name=job_name,
            instance_count=instance_count,
        )
        # Filter out None/default values
        alt_options = {k: v for k, v in alt_options.items() if v is not None}
        try:
            c = SageMakerCluster.from_name(name, dryrun, alt_options=alt_options)
            if c:
                return c
        except ValueError:
            pass

    if name in RESERVED_SYSTEM_NAMES:
        raise ValueError(
            f"Cluster name {name} is a reserved name. Please use a different name which is not one of "
            f"{RESERVED_SYSTEM_NAMES}."
        )

    return SageMakerCluster(
        name=name,
        role=role,
        profile=profile,
        ssh_key_path=ssh_key_path,
        estimator=estimator,
        job_name=job_name,
        instance_type=instance_type,
        instance_count=instance_count,
        image_uri=image_uri,
        autostop_mins=autostop_mins,
        connection_wait_time=connection_wait_time,
        dryrun=dryrun,
    )
