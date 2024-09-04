import os
import subprocess
import warnings

from typing import Dict, List, Optional, Union

from runhouse.constants import RESERVED_SYSTEM_NAMES
from runhouse.globals import rns_client

from runhouse.logger import get_logger
from runhouse.resources.hardware.utils import ServerConnectionType

from .cluster import Cluster
from .on_demand_cluster import OnDemandCluster

logger = get_logger(__name__)

# Cluster factory method
def cluster(
    name: str,
    host: Union[str, List[str]] = None,
    ssh_creds: Union[Dict, str] = None,
    server_port: int = None,
    server_host: str = None,
    server_connection_type: Union[ServerConnectionType, str] = None,
    ssl_keyfile: str = None,
    ssl_certfile: str = None,
    domain: str = None,
    den_auth: bool = None,
    default_env: Union["Env", str] = None,
    load_from_den: bool = True,
    dryrun: bool = False,
    **kwargs,
) -> Union[Cluster, OnDemandCluster]:
    """
    Builds an instance of :class:`Cluster`.

    Args:
        name (str): Name for the cluster, to re-use later on.
        host (str or List[str], optional): Hostname (e.g. domain or name in .ssh/config), IP address, or list of IP
            addresses for the cluster (the first of which is the head node). (Default: ``None``).
        ssh_creds (dict or str, optional): SSH credentials, passed as dictionary or the name of an `SSHSecret` object.
            Example: ``ssh_creds={'ssh_user': '...', 'ssh_private_key':'<path_to_key>'}`` (Default: ``None``).
        server_port (bool, optional): Port to use for the server. If not provided will use 80 for a
            ``server_connection_type`` of ``none``, 443 for ``tls`` and ``32300`` for all other SSH connection types.
        server_host (bool, optional): Host from which the server listens for traffic (i.e. the --host argument
            `runhouse start` run on the cluster). Defaults to "0.0.0.0" unless connecting to the server with an SSH
            connection, in which case ``localhost`` is used. (Default: ``None``).
        server_connection_type (ServerConnectionType or str, optional): Type of connection to use for the Runhouse
            API server. ``ssh`` will use start with server via an SSH tunnel. ``tls`` will start the server
            with HTTPS on port 443 using TLS certs without an SSH tunnel. ``none`` will start the server with HTTP
            without an SSH tunnel. (Default: ``None``).
        ssl_keyfile(str, optional): Path to SSL key file to use for launching the API server with HTTPS.
            (Default: ``None``).
        ssl_certfile(str, optional): Path to SSL certificate file to use for launching the API server with HTTPS.
            (Default: ``None``).
        domain(str, optional): Domain name for the cluster. Relevant if enabling HTTPs on the cluster. (Default: ``None``).
        den_auth (bool, optional): Whether to use Den authorization on the server. If ``True``, will validate incoming
            requests with a Runhouse token provided in the auth headers of the request with the format:
            ``{"Authorization": "Bearer <token>"}``. (Default: ``None``).
        default_env (Env or str, optional): Environment that the Runhouse server is started on in the cluster. Used to
            specify an isolated environment (e.g. conda env) or any setup and requirements prior to starting the Runhouse
            server. (Default: ``None``)
        load_from_den (bool): Whether to try loading the Cluster resource from Den. (Default: ``True``)
        dryrun (bool): Whether to create the Cluster if it doesn't exist, or load a Cluster object as a dryrun.
            (Default: ``False``)

    Returns:
        Union[Cluster, OnDemandCluster]: The resulting cluster.

    Example:
        >>> # using private key
        >>> gpu = rh.cluster(host='<hostname>',
        >>>                  ssh_creds={'ssh_user': '...', 'ssh_private_key':'<path_to_key>'},
        >>>                  name='rh-a10x').save()

        >>> # using password
        >>> gpu = rh.cluster(host='<hostname>',
        >>>                  ssh_creds={'ssh_user': '...', 'password':'*****'},
        >>>                  name='rh-a10x').save()

        >>> # using the name of an SSHSecret object
        >>> gpu = rh.cluster(host='<hostname>',
        >>>                  ssh_creds="my_ssh_secret",
        >>>                  name='rh-a10x').save()

        >>> # Load cluster from above
        >>> reloaded_cluster = rh.cluster(name="rh-a10x")
    """
    if host and kwargs.get("ips"):
        raise ValueError(
            "Cluster factory method can only accept one of `host` or `ips` as an argument."
        )

    if name:
        alt_options = dict(
            host=host,
            ssh_creds=ssh_creds,
            server_port=server_port,
            server_host=server_host,
            server_connection_type=server_connection_type,
            ssl_keyfile=ssl_keyfile,
            ssl_certfile=ssl_certfile,
            domain=domain,
            den_auth=den_auth,
            default_env=default_env,
            kwargs=kwargs if len(kwargs) > 0 else None,
        )
        # Filter out None/default values
        alt_options = {k: v for k, v in alt_options.items() if v is not None}
        try:
            c = Cluster.from_name(
                name,
                load_from_den=load_from_den,
                dryrun=dryrun,
                _alt_options=alt_options,
            )
            if c:
                c.set_connection_defaults()
                if den_auth:
                    c.save()
                return c
        except ValueError as e:
            if not alt_options:
                raise e

    if ssh_creds:
        from runhouse.resources.secrets.utils import setup_cluster_creds

        ssh_creds_secret = setup_cluster_creds(ssh_creds, name)
    else:
        ssh_creds_secret = ssh_creds

    if "instance_type" in kwargs.keys():
        return ondemand_cluster(
            name=name,
            creds=ssh_creds_secret,
            server_port=server_port,
            server_host=server_host,
            server_connection_type=server_connection_type,
            ssl_keyfile=ssl_keyfile,
            ssl_certfile=ssl_certfile,
            domain=domain,
            den_auth=den_auth,
            default_env=default_env,
            dryrun=dryrun,
            **kwargs,
        )

    if isinstance(host, str):
        host = [host]

    c = Cluster(
        ips=kwargs.pop("ips", None) or host,
        creds=ssh_creds_secret,
        name=name,
        server_host=server_host,
        server_port=server_port,
        server_connection_type=server_connection_type,
        ssl_keyfile=ssl_keyfile,
        ssl_certfile=ssl_certfile,
        domain=domain,
        den_auth=den_auth,
        default_env=default_env,
        dryrun=dryrun,
        **kwargs,
    )
    c.set_connection_defaults(**kwargs)

    if den_auth or rns_client.autosave_resources():
        c.save()

    return c


def kubernetes_cluster(
    name: str,
    instance_type: str = None,
    namespace: str = None,
    kube_config_path: str = None,
    context: str = None,
    server_connection_type: Union[ServerConnectionType, str] = None,
    **kwargs,
) -> OnDemandCluster:

    # if user passes provider via kwargs to kubernetes_cluster
    provider_passed = kwargs.pop("provider", None)

    if provider_passed is not None and provider_passed != "kubernetes":
        raise ValueError(
            f"Runhouse K8s Cluster provider must be `kubernetes`. "
            f"You passed {provider_passed}."
        )

    # checking server_connection_type passed over from ondemand_cluster factory method
    if (
        server_connection_type is not None
        and server_connection_type != ServerConnectionType.SSH
    ):
        raise ValueError(
            f"Runhouse K8s Cluster server connection type must be set to `ssh`. "
            f"You passed {server_connection_type}."
        )

    if context is not None and namespace is not None:
        warnings.warn(
            "You passed both a context and a namespace. Ensure your namespace matches the one in your context.",
            UserWarning,
        )

    if namespace is not None:  # check if user passed a user-defined namespace
        cmd = f"kubectl config set-context --current --namespace={namespace}"
        try:
            process = subprocess.run(
                cmd,
                shell=True,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            logger.debug(process.stdout)
            logger.info(f"Kubernetes namespace set to {namespace}")

        except subprocess.CalledProcessError as e:
            logger.info(f"Error: {e}")

    if (
        kube_config_path is not None
    ):  # check if user passed a user-defined kube_config_path
        kube_config_dir = os.path.expanduser("~/.kube")
        kube_config_path_rl = os.path.join(kube_config_dir, "config")

        if not os.path.exists(
            kube_config_dir
        ):  # check if ~/.kube directory exists on local machine
            try:
                os.makedirs(
                    kube_config_dir
                )  # create ~/.kube directory if it doesn't exist
                logger.info(f"Created directory: {kube_config_dir}")
            except OSError as e:
                logger.info(f"Error creating directory: {e}")

        if os.path.exists(kube_config_path_rl):
            raise Exception(
                "A kubeconfig file already exists in ~/.kube directory. Aborting."
            )

        try:
            cmd = f"cp {kube_config_path} {kube_config_path_rl}"  # copy user-defined kube_config to ~/.kube/config
            subprocess.run(cmd, shell=True, check=True)
            logger.info(f"Copied kubeconfig to: {kube_config_path}")
        except subprocess.CalledProcessError as e:
            logger.info(f"Error copying kubeconfig: {e}")

    if context is not None:  # check if user passed a user-defined context
        try:
            cmd = f"kubectl config use-context {context}"  # set user-defined context as current context
            subprocess.run(cmd, shell=True, check=True)
            logger.info(f"Kubernetes context has been set to: {context}")
        except subprocess.CalledProcessError as e:
            logger.info(f"Error setting context: {e}")

    c = OnDemandCluster(
        name=name,
        instance_type=instance_type,
        provider="kubernetes",
        server_connection_type=server_connection_type,
        **kwargs,
    )
    c.set_connection_defaults()

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
    sky_kwargs: Dict = None,
    server_port: int = None,
    server_host: int = None,
    server_connection_type: Union[ServerConnectionType, str] = None,
    ssl_keyfile: str = None,
    ssl_certfile: str = None,
    domain: str = None,
    den_auth: bool = None,
    default_env: Union["Env", str] = None,
    load_from_den: bool = True,
    dryrun: bool = False,
    **kwargs,
) -> OnDemandCluster:
    """
    Builds an instance of :class:`OnDemandCluster`. Note that image_id, region, memory, disk_size, and open_ports
    are all passed through to SkyPilot's `Resource constructor
    <https://skypilot.readthedocs.io/en/latest/reference/api.html#resources>`__.

    Args:
        name (str): Name for the cluster, to re-use later on.
        instance_type (int, optional): Type of cloud instance to use for the cluster. This could
            be a Runhouse built-in type, or your choice of instance type.
        num_instances (int, optional): Number of instances to use for the cluster.
        provider (str, optional): Cloud provider to use for the cluster.
        autostop_mins (int, optional): Number of minutes to keep the cluster up after inactivity,
            or ``-1`` to keep cluster up indefinitely.
        use_spot (bool, optional): Whether or not to use spot instance.
        image_id (str, optional): Custom image ID for the cluster. If using a docker image, please use the following
            string format: "docker:<registry>/<image>:<tag>". See `user guide <https://www.run.house/docs/docker>`__
            for more information on Docker cluster setup.
        region (str, optional): The region to use for the cluster.
        memory (int or str, optional): Amount of memory to use for the cluster, e.g. "16" or "16+".
        disk_size (int or str, optional): Amount of disk space to use for the cluster, e.g. "100" or "100+".
        open_ports (int or str or List[int], optional): Ports to open in the cluster's security group. Note
            that you are responsible for ensuring that the applications listening on these ports are secure.
        sky_kwargs (dict, optional): Additional keyword arguments to pass to the SkyPilot `Resource` or
            `launch` APIs. Should be a dict of the form
            `{"resources": {<resources_kwargs>}, "launch": {<launch_kwargs>}}`, where resources_kwargs and
            launch_kwargs will be passed to the SkyPilot Resources API
            (See `SkyPilot docs <https://skypilot.readthedocs.io/en/latest/reference/api.html#resources>`__)
            and `launch` API (See
            `SkyPilot docs <https://skypilot.readthedocs.io/en/latest/reference/api.html#sky-launch>`__), respectively.
            Any arguments which duplicate those passed to the `ondemand_cluster` factory method will raise an error.
        server_port (bool, optional): Port to use for the server. If not provided will use 80 for a
            ``server_connection_type`` of ``none``, 443 for ``tls`` and ``32300`` for all other SSH connection types.
        server_host (bool, optional): Host from which the server listens for traffic (i.e. the --host argument
            `runhouse start` run on the cluster). Defaults to "0.0.0.0" unless connecting to the server with an SSH
            connection, in which case ``localhost`` is used.
        server_connection_type (ServerConnectionType or str, optional): Type of connection to use for the Runhouse
            API server. ``ssh`` will use start with server via an SSH tunnel. ``tls`` will start the server
            with HTTPS on port 443 using TLS certs without an SSH tunnel. ``none`` will start the server with HTTP
            without an SSH tunnel.
        ssl_keyfile(str, optional): Path to SSL key file to use for launching the API server with HTTPS.
        ssl_certfile(str, optional): Path to SSL certificate file to use for launching the API server with HTTPS.
        domain(str, optional): Domain name for the cluster. Relevant if enabling HTTPs on the cluster.
        den_auth (bool, optional): Whether to use Den authorization on the server. If ``True``, will validate incoming
            requests with a Runhouse token provided in the auth headers of the request with the format:
            ``{"Authorization": "Bearer <token>"}``. (Default: ``None``).
        default_env (Env or str, optional): Environment that the Runhouse server is started on in the cluster. Used to
            specify an isolated environment (e.g. conda env) or any setup and requirements prior to starting the Runhouse
            server. (Default: ``None``)
        load_from_den (bool): Whether to try loading the Cluster resource from Den. (Default: ``True``)
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

    if name in RESERVED_SYSTEM_NAMES:
        raise ValueError(
            f"Cluster name {name} is a reserved name. Please use a different name which is not one of "
            f"{RESERVED_SYSTEM_NAMES}."
        )

    if provider == "kubernetes":
        namespace = kwargs.pop("namespace", None)
        kube_config_path = kwargs.pop("kube_config_path", None)
        context = kwargs.pop("context", None)
        server_connection_type = kwargs.pop("server_connection_type", None)
        default_env = kwargs.pop("default_env", None)

        return kubernetes_cluster(
            name=name,
            instance_type=instance_type,
            namespace=namespace,
            kube_config_path=kube_config_path,
            context=context,
            server_connection_type=server_connection_type,
            default_env=default_env,
            autostop_mins=autostop_mins,
            num_instances=num_instances,
            provider=provider,
            use_spot=use_spot,
            image_id=image_id,
            region=region,
            memory=memory,
            disk_size=disk_size,
            open_ports=open_ports,
            sky_kwargs=sky_kwargs,
            server_port=server_port,
            server_host=server_host,
            ssl_keyfile=ssl_keyfile,
            ssl_certfile=ssl_certfile,
            domain=domain,
            den_auth=den_auth,
            dryrun=dryrun,
            **kwargs,
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
            domain=domain,
            den_auth=den_auth,
            default_env=default_env,
        )
        # Filter out None/default values
        alt_options = {k: v for k, v in alt_options.items() if v is not None}
        try:
            c = Cluster.from_name(
                name,
                load_from_den=load_from_den,
                dryrun=dryrun,
                _alt_options=alt_options,
            )
            if c:
                c.set_connection_defaults()
                if den_auth:
                    c.save()
                return c
        except ValueError as e:
            import sky

            state = sky.status(cluster_names=[name], refresh=False)
            if len(state) == 0 and not alt_options:
                raise e

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
        sky_kwargs=sky_kwargs,
        server_host=server_host,
        server_port=server_port,
        server_connection_type=server_connection_type,
        ssl_keyfile=ssl_keyfile,
        ssl_certfile=ssl_certfile,
        domain=domain,
        den_auth=den_auth,
        default_env=default_env,
        name=name,
        dryrun=dryrun,
        **kwargs,
    )
    c.set_connection_defaults()

    if den_auth or rns_client.autosave_resources():
        c.save()

    return c
