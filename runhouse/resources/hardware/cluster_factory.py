import warnings
from typing import Dict, List, Optional, Union

from runhouse.resources.hardware.utils import RESERVED_SYSTEM_NAMES

from .cluster import Cluster, ServerConnectionType
from .on_demand_cluster import OnDemandCluster
from .sagemaker_cluster import SageMakerCluster


# Cluster factory method
def cluster(
    name: str,
    host: Union[str, List[str]] = None,
    ssh_creds: Optional[dict] = None,
    server_port: int = None,
    server_host: int = None,
    server_connection_type: Union[ServerConnectionType, str] = None,
    den_auth: bool = False,
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
        server_port (bool, optional): Port to use for the server. (Default: ``50052``).
        server_host (bool, optional): Host to use for the server. (Default: ``127.0.0.1``).
        server_connection_type (ServerConnectionType or str, optional): Type of connection to use for the Runhouse
            API server. ``ssh`` will use start with server with HTTP via port forwarding. ``tls`` will start the server
            with HTTPS using TLS certs. ``none`` will start the server with HTTP without using any
            port forwarding.
        den_auth (bool, optional): Whether to use Den authorization on the server. If ``True``, will validate incoming
            requests with a Runhouse token provided in the auth headers of the request with the format:
            ``{"Authorization": "Bearer <token>"}``. (Default: ``False``).
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

    if name and all(
        x is None
        for x in [
            host,
            ssh_creds,
            server_port,
            server_host,
            server_connection_type,
            den_auth,
            kwargs,
        ]
    ):
        # If only the name is provided
        return Cluster.from_name(name, dryrun)

    if name in RESERVED_SYSTEM_NAMES:
        raise ValueError(
            f"Cluster name {name} is a reserved name. Please use a different name which is not one of "
            f"{RESERVED_SYSTEM_NAMES}."
        )

    if "instance_type" in kwargs.keys():
        return ondemand_cluster(name=name, **kwargs)

    if any(
        k in kwargs.keys()
        for k in [
            "role",
            "estimator",
            "instance_type",
            "connection_wait_time",
            "instance_count",
        ]
    ):
        warnings.warn(
            "The `cluster` factory is intended to be used for static clusters. "
            "If you would like to create a sagemaker cluster, please use `rh.sagemaker_cluster()` instead."
        )
        return sagemaker_cluster(name=name, **kwargs)

    return Cluster(
        ips=host,
        ssh_creds=ssh_creds,
        name=name,
        server_host=server_host,
        server_port=server_port,
        server_connection_type=server_connection_type,
        den_auth=den_auth,
        dryrun=dryrun,
    )


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
    memory: Union[int, str, None] = None,
    disk_size: Union[int, str, None] = None,
    open_ports: Union[int, str, List[int], None] = None,
    server_port: int = None,
    server_host: int = None,
    server_connection_type: Union[ServerConnectionType, str] = None,
    den_auth: bool = False,
    dryrun: bool = False,
) -> OnDemandCluster:
    """
    Builds an instance of :class:`OnDemandCluster`. Note that image_id, region, memory, disk_size, and open_ports
    are all passed through to SkyPilot's Resource constructor:
    https://skypilot.readthedocs.io/en/latest/reference/api.html#resources

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
        memory (int or str, optional): Amount of memory to use for the cluster, e.g. "16" or "16+".
        disk_size (int or str, optional): Amount of disk space to use for the cluster, e.g. "100" or "100+".
        open_ports (int or str or List[int], optional): Ports to open in the cluster's security group. Note
            that you are responsible for ensuring that the applications listening on these ports are secure.
        server_port (bool, optional): Port to use for the server. (Default: ``50052``).
        server_host (bool, optional): Host to use for the server. (Default: ``127.0.0.1``).
        server_connection_type (ServerConnectionType or str, optional): Type of connection to use for the Runhouse
            API server. ``ssh`` will use start with server with HTTP via port forwarding. ``tls`` will start the server
            with HTTPS using TLS certs. ``none`` will start the server with HTTP without using any
            port forwarding.
        den_auth (bool, optional): Whether to use Den authorization on the server. If ``True``, will validate incoming
            requests with a Runhouse token provided in the auth headers of the request with the format:
            ``{"Authorization": "Bearer <token>"}``. (Default: ``False``).
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
    if name:
        alt_options = dict(
            instance_type=instance_type,
            num_instances=num_instances,
            provider=provider,
            region=region,
            image_id=image_id,
            memory=memory,
            disk_size=disk_size,
            open_ports=open_ports,
            server_host=server_host,
            server_port=server_port,
            server_connection_type=server_connection_type,
            den_auth=den_auth,
        )
        # Filter out None/default values
        alt_options = {k: v for k, v in alt_options.items() if v is not None}
        try:
            c = Cluster.from_name(name, dryrun, alt_options=alt_options)
            if c:
                return c
        except ValueError:
            pass

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
        memory=memory,
        disk_size=disk_size,
        open_ports=open_ports,
        server_host=server_host,
        server_port=server_port,
        server_connection_type=server_connection_type,
        den_auth=den_auth,
        name=name,
        dryrun=dryrun,
    )


def sagemaker_cluster(
    name: str,
    role: str = None,
    profile: str = None,
    ssh_key_path: str = None,
    instance_id: str = None,
    instance_type: str = None,
    instance_count: int = None,
    image_uri: str = None,
    autostop_mins: int = None,
    connection_wait_time: int = None,
    estimator: Union["sagemaker.estimator.EstimatorBase", Dict] = None,
    job_name: str = None,
    server_port: int = None,
    server_host: int = None,
    server_connection_type: Union[ServerConnectionType, str] = None,
    den_auth: bool = False,
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
            If not provided, will use the ``default`` profile.
        ssh_key_path (str, optional): Path (relative or absolute) to private SSH key to use for connecting to
            the cluster. If not provided, will look for the key in path ``~/.ssh/sagemaker-ssh-gw``.
            If not found will generate new keys and upload the public key to the default s3 bucket for the Role ARN.
        instance_id (str, optional): ID of the AWS instance to use for the cluster. SageMaker does not expose
            IP addresses of its instance, so we use an instance ID as a unique identifier for the cluster.
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
            or ``-1`` to keep cluster up indefinitely. *Note: this will keep the cluster up even if a dedicated
            training job has finished running or failed*.
        connection_wait_time (int, optional): Amount of time to wait inside the SageMaker cluster before
            continuing with normal execution. Useful if you want to connect before a dedicated job starts
            (e.g. training). If you don't want to wait, set it to ``0``.
            If no estimator is provided, will default to ``0``.
        job_name (str, optional): Name to provide for a training job. If not provided will generate a default name
            based on the image name and current timestamp (e.g. ``pytorch-training-2023-08-28-20-57-55-113``).
        server_port (bool, optional): Port to use for the server. (Default: ``50052``).
        server_host (bool, optional): Host to use for the server. (Default: ``127.0.0.1``).
        server_connection_type (ServerConnectionType or str, optional): Type of connection to use for the Runhouse
            API server. ``ssh`` will use start with server with HTTP via port forwarding. ``tls`` will start the server
            with HTTPS using TLS certs. ``none`` will start the server with HTTP without using any
            port forwarding.
        den_auth (bool, optional): Whether to use Den authorization on the server. If ``True``, will validate incoming
            requests with a Runhouse token provided in the auth headers of the request with the format:
            ``{"Authorization": "Bearer <token>"}``. (Default: ``False``).
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
    ssh_key_path = (
        SageMakerCluster._relative_ssh_path(ssh_key_path) if ssh_key_path else None
    )
    if name:
        alt_options = dict(
            role=role,
            profile=profile,
            ssh_key_path=ssh_key_path,
            instance_id=instance_id,
            image_uri=image_uri,
            estimator=estimator,
            instance_type=instance_type,
            job_name=job_name,
            instance_count=instance_count,
            server_host=server_host,
            server_port=server_port,
            server_connection_type=server_connection_type,
            den_auth=den_auth,
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
        instance_id=instance_id,
        instance_type=instance_type,
        instance_count=instance_count,
        image_uri=image_uri,
        autostop_mins=autostop_mins,
        connection_wait_time=connection_wait_time,
        server_host=server_host,
        server_port=server_port,
        server_connection_type=server_connection_type,
        den_auth=den_auth,
        dryrun=dryrun,
    )
