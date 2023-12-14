import subprocess
import warnings

from typing import Dict, List, Optional, Union

from runhouse.resources.hardware.utils import (
    RESERVED_SYSTEM_NAMES,
    ServerConnectionType,
)
from runhouse.rns.utils.api import relative_ssh_path

from .cluster import Cluster
from .on_demand_cluster import OnDemandCluster
from .sagemaker_cluster import SageMakerCluster


# Cluster factory method
def cluster(
    name: str,
    host: Union[str, List[str]] = None,
    ssh_creds: Optional[dict] = None,
    server_port: int = None,
    server_host: str = None,
    server_connection_type: Union[ServerConnectionType, str] = None,
    ssl_keyfile: str = None,
    ssl_certfile: str = None,
    den_auth: bool = False,
    dryrun: bool = False,
    **kwargs,
) -> Union[Cluster, OnDemandCluster, SageMakerCluster]:
    """
    Builds an instance of :class:`Cluster`.

    Args:
        name (str): Name for the cluster, to re-use later on.
        host (str or List[str], optional): Hostname (e.g. domain or name in .ssh/config), IP address, or list of IP
            addresses for the cluster (the first of which is the head node).
        ssh_creds (dict, optional): Dictionary mapping SSH credentials.
            Example: ``ssh_creds={'ssh_user': '...', 'ssh_private_key':'<path_to_key>'}``
        server_port (bool, optional): Port to use for the server. If not provided will use 80 for a
            ``server_connection_type`` of ``none``, 443 for ``tls`` and ``32300`` for all other SSH connection types.
        server_host (bool, optional): Host from which the server listens for traffic (i.e. the --host argument
            `runhouse start` run on the cluster). Defaults to "0.0.0.0" unless connecting to the server with an SSH
            connection, in which case ``localhost`` is used.
        server_connection_type (ServerConnectionType or str, optional): Type of connection to use for the Runhouse
            API server. ``ssh`` will use start with server via an SSH tunnel. ``tls`` will start the server
            with HTTPS on port 443 using TLS certs without an SSH tunnel. ``none`` will start the server with HTTP
            without an SSH tunnel. ``aws_ssm`` will start the server with HTTP using AWS SSM port forwarding.
            ``paramiko``will use paramiko to create an SSH tunnel to the cluster.
        ssl_keyfile(str, optional): Path to SSL key file to use for launching the API server with HTTPS.
        ssl_certfile(str, optional): Path to SSL certificate file to use for launching the API server with HTTPS.
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
        not x
        for x in [
            host,
            ssh_creds,
            server_port,
            server_host,
            server_connection_type,
            ssl_keyfile,
            ssl_certfile,
            den_auth,
            kwargs,
        ]
    ):
        # If only the name is provided
        return Cluster.from_name(name, dryrun)

    if host and ("localhost" in host or ":" in host):
        # If server_connection_type is not specified, we
        # assume we can hit the server directly via HTTP
        server_connection_type = server_connection_type or ServerConnectionType.NONE
        if ":" in host:
            # e.g. "localhost:23324" or <real_ip>:<custom port> (e.g. a port is already open to the server)
            host, client_port = host.split(":")
            kwargs["client_port"] = client_port

    server_connection_type = server_connection_type or (
        ServerConnectionType.TLS
        if ssl_certfile or ssl_keyfile
        else ServerConnectionType.SSH
    )

    if server_port is None:
        if server_connection_type == ServerConnectionType.TLS:
            server_port = Cluster.DEFAULT_HTTPS_PORT
        elif server_connection_type == ServerConnectionType.NONE:
            server_port = Cluster.DEFAULT_HTTP_PORT
        else:
            server_port = Cluster.DEFAULT_SERVER_PORT

    if name in RESERVED_SYSTEM_NAMES:
        raise ValueError(
            f"Cluster name {name} is a reserved name. Please use a different name which is not one of "
            f"{RESERVED_SYSTEM_NAMES}."
        )

    if "instance_type" in kwargs.keys():
        return ondemand_cluster(
            name=name,
            ssh_creds=ssh_creds,
            server_port=server_port,
            server_host=server_host,
            server_connection_type=server_connection_type,
            ssl_keyfile=ssl_keyfile,
            ssl_certfile=ssl_certfile,
            den_auth=den_auth,
            dryrun=dryrun,
            **kwargs,
        )

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
        return sagemaker_cluster(
            name=name,
            ssh_creds=ssh_creds,
            server_port=server_port,
            server_host=server_host,
            server_connection_type=server_connection_type,
            ssl_keyfile=ssl_keyfile,
            ssl_certfile=ssl_certfile,
            den_auth=den_auth,
            dryrun=dryrun,
            **kwargs,
        )

    if server_connection_type == ServerConnectionType.AWS_SSM:
        raise ValueError(
            f"Cluster does not support server connection type of {server_connection_type}"
        )

    if isinstance(host, str):
        host = [host]

    c = Cluster(
        ips=host,
        ssh_creds=ssh_creds,
        name=name,
        server_host=server_host,
        server_port=server_port,
        server_connection_type=server_connection_type,
        ssl_keyfile=ssl_keyfile,
        ssl_certfile=ssl_certfile,
        den_auth=den_auth,
        dryrun=dryrun,
        **kwargs,
    )

    if den_auth:
        c.save()

    return c


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
    ssl_keyfile: str = None,
    ssl_certfile: str = None,
    den_auth: bool = False,
    dryrun: bool = False,
    **kwargs,
) -> OnDemandCluster:
    """
    Builds an instance of :class:`OnDemandCluster`. Note that image_id, region, memory, disk_size, and open_ports
    are all passed through to SkyPilot's `Resource constructor
    <https://skypilot.readthedocs.io/en/latest/reference/api.html#resources>`_.

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
        server_port (bool, optional): Port to use for the server. If not provided will use 80 for a
            ``server_connection_type`` of ``none``, 443 for ``tls`` and ``32300`` for all other SSH connection types.
        server_host (bool, optional): Host from which the server listens for traffic (i.e. the --host argument
            `runhouse start` run on the cluster). Defaults to "0.0.0.0" unless connecting to the server with an SSH
            connection, in which case ``localhost`` is used.
        server_connection_type (ServerConnectionType or str, optional): Type of connection to use for the Runhouse
            API server. ``ssh`` will use start with server via an SSH tunnel. ``tls`` will start the server
            with HTTPS on port 443 using TLS certs without an SSH tunnel. ``none`` will start the server with HTTP
            without an SSH tunnel. ``aws_ssm`` will start the server with HTTP using AWS SSM port forwarding.
            ``paramiko``will use paramiko to create an SSH tunnel to the cluster.
        ssl_keyfile(str, optional): Path to SSL key file to use for launching the API server with HTTPS.
        ssl_certfile(str, optional): Path to SSL certificate file to use for launching the API server with HTTPS.
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
    if server_connection_type in [
        ServerConnectionType.AWS_SSM,
    ]:
        raise ValueError(
            f"OnDemandCluster does not support server connection type {server_connection_type}"
        )

    if not server_connection_type:
        if ssl_keyfile or ssl_certfile:
            server_connection_type = ServerConnectionType.TLS
        else:
            server_connection_type = ServerConnectionType.SSH

    if server_port is None:
        if server_connection_type == ServerConnectionType.TLS:
            server_port = Cluster.DEFAULT_HTTPS_PORT
        elif server_connection_type == ServerConnectionType.NONE:
            server_port = Cluster.DEFAULT_HTTP_PORT
        else:
            server_port = Cluster.DEFAULT_SERVER_PORT

    if (
        server_connection_type in [ServerConnectionType.TLS, ServerConnectionType.NONE]
        and server_host in Cluster.LOCAL_HOSTS
    ):
        warnings.warn(
            f"Server connection type set to {server_connection_type}, with server host set to {server_host}. "
            f"Note that this will require opening an SSH tunnel to forward traffic from {server_host} to the server."
        )

    open_ports = (
        []
        if open_ports is None
        else [open_ports]
        if isinstance(open_ports, (int, str))
        else open_ports
    )

    if open_ports:
        open_ports = [str(p) for p in open_ports]
        if str(server_port) in open_ports:
            if (
                server_connection_type
                in [ServerConnectionType.TLS, ServerConnectionType.NONE]
                and not den_auth
            ):
                warnings.warn(
                    "Server is insecure and must be inside a VPC or have `den_auth` enabled to secure it."
                )
        else:
            warnings.warn(
                f"Server port {server_port} not included in open ports. Note you are responsible for opening "
                f"the port or ensure you have access to it via a VPC."
            )
    else:
        # If using HTTP or HTTPS must enable traffic on the relevant port
        if server_connection_type in [
            ServerConnectionType.TLS,
            ServerConnectionType.NONE,
        ]:
            if server_port:
                warnings.warn(
                    f"No open ports specified. Make sure port {server_port} is open "
                    f"to {server_connection_type} traffic."
                )
            else:
                warnings.warn(
                    f"No open ports specified. Make sure the relevant port is open. "
                    f"HTTPS default: {Cluster.DEFAULT_HTTPS_PORT} and HTTP "
                    f"default: {Cluster.DEFAULT_HTTP_PORT}."
                )

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
            ssl_keyfile=ssl_keyfile,
            ssl_certfile=ssl_certfile,
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

    c = OnDemandCluster(
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
        ssl_keyfile=ssl_keyfile,
        ssl_certfile=ssl_certfile,
        den_auth=den_auth,
        name=name,
        dryrun=dryrun,
        **kwargs,
    )

    if den_auth:
        c.save()

    return c


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
    ssl_keyfile: str = None,
    ssl_certfile: str = None,
    den_auth: bool = False,
    dryrun: bool = False,
    **kwargs,
) -> SageMakerCluster:
    """
    Builds an instance of :class:`SageMakerCluster`. See SageMaker Hardware Setup section for more specific
    instructions and requirements for providing the role and setting up the cluster.

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
        server_port (bool, optional): Port to use for the server (Default: ``32300``).
        server_host (bool, optional): Host from which the server listens for traffic (i.e. the --host argument
            `runhouse start` run on the cluster).
            *Note: For SageMaker, since we connect to the Runhouse API server via an SSH tunnel, the only valid
            host is localhost.*
        server_connection_type (ServerConnectionType or str, optional): Type of connection to use for the Runhouse
            API server. *Note: For SageMaker, only ``aws_ssm`` is currently valid as the server connection type.*
        ssl_keyfile(str, optional): Path to SSL key file to use for launching the API server with HTTPS.
        ssl_certfile(str, optional): Path to SSL certificate file to use for launching the API server with HTTPS.
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
    if (
        "aws-cli/2."
        not in subprocess.run(
            ["aws", "--version"], capture_output=True, text=True
        ).stdout
    ):
        raise RuntimeError(
            "SageMaker SDK requires AWS CLI v2. You may also need to run `pip uninstall awscli` to ensure the right "
            "version is being used. For more info: https://www.run.house/docs/api/python/cluster#id2"
        )

    ssh_key_path = relative_ssh_path(ssh_key_path) if ssh_key_path else None

    if (
        server_connection_type is not None
        and server_connection_type != ServerConnectionType.AWS_SSM
    ):
        raise ValueError(
            "SageMaker Cluster currently requires a server connection type of `aws_ssm`."
        )
    server_connection_type = ServerConnectionType.AWS_SSM.value

    if server_host and server_host not in Cluster.LOCAL_HOSTS:
        raise ValueError(
            "SageMaker Cluster currently requires a server host of `localhost` or `127.0.0.1`"
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
            ssl_keyfile=ssl_keyfile,
            ssl_certfile=ssl_certfile,
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

    sm = SageMakerCluster(
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
        ssl_keyfile=ssl_keyfile,
        ssl_certfile=ssl_certfile,
        dryrun=dryrun,
        **kwargs,
    )

    if den_auth:
        sm.save()

    return sm
