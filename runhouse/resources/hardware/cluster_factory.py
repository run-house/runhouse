from typing import Dict, List, Optional, Union

from runhouse.globals import rns_client

from runhouse.logger import get_logger
from runhouse.resources.hardware.cluster import Cluster
from runhouse.resources.hardware.constants import (
    KUBERNETES_CLUSTER_ARGS,
    ONDEMAND_COMPUTE_ARGS,
    RH_SERVER_ARGS,
    STATIC_CLUSTER_ARGS,
)
from runhouse.resources.hardware.on_demand_cluster import OnDemandCluster
from runhouse.resources.hardware.utils import (
    _config_and_args_mismatches,
    LauncherType,
    ServerConnectionType,
    setup_kubernetes,
)
from runhouse.resources.images.image import Image

logger = get_logger(__name__)


def cluster(
    name: str,
    host: Union[str, List[str]] = None,
    ssh_creds: Union[Dict, str] = None,
    ssh_port: Optional[int] = None,
    client_port: Optional[int] = None,
    server_port: Optional[int] = None,
    server_host: Optional[str] = None,
    server_connection_type: Union[ServerConnectionType, str] = None,
    ssl_keyfile: Optional[str] = None,
    ssl_certfile: Optional[str] = None,
    domain: Optional[str] = None,
    image: Optional[Image] = None,
    den_auth: bool = None,
    load_from_den: bool = True,
    dryrun: bool = False,
    **kwargs,
):
    """
    Builds an instance of :class:`Cluster`.

    * If Cluster with same name is found in Den and ``load_from_den`` is ``True``, load it down from Den
    * If arguments corresponding to ondemand clusters are provided, arguments are fed through to
      ``rh.ondemand_cluster`` factory function
    * If arguments are mismatched with loaded Cluster, return a new Cluster with the provided args

    Args:
        name (str): Name for the cluster.
        host (str or List[str], optional): Hostname (e.g. domain or name in .ssh/config), IP address, or list of IP
            addresses for the cluster (the first of which is the head node). (Default: ``None``).
        ssh_creds (Dict or str, optional): SSH credentials, passed as dictionary or the name of an ``SSHSecret`` object.
            Example: ``ssh_creds={'ssh_user': '...', 'ssh_private_key':'<path_to_key>'}`` (Default: ``None``).
        ssh_port (int, optional): Port to use for ssh. If not provided, will default to ``22``.
        client_port (int, optional): Port to use for the client. If not provided, will default to the server port.
        server_port (bool, optional): Port to use for the server. If not provided will use 80 for a
            ``server_connection_type`` or ``none``, 443 for ``tls`` and ``32300`` for all other SSH connection types.
        server_host (bool, optional): Host from which the server listens for traffic (i.e. the --host argument
            `runhouse server start` run on the cluster). Defaults to ``"0.0.0.0"`` unless connecting to the server
            with an SSH connection, in which case ``localhost`` is used. (Default: ``None``).
        server_connection_type (ServerConnectionType or str, optional): Type of connection to use for the Runhouse
            API server. ``ssh`` will use start with server via an SSH tunnel. ``tls`` will start the server
            with HTTPS on port 443 using TLS certs without an SSH tunnel. ``none`` will start the server with HTTP
            without an SSH tunnel. (Default: ``None``).
        ssl_keyfile(str, optional): Path to SSL key file to use for launching the API server with HTTPS.
            (Default: ``None``).
        ssl_certfile(str, optional): Path to SSL certificate file to use for launching the API server with HTTPS.
            (Default: ``None``).
        domain(str, optional): Domain name for the cluster. Relevant if enabling HTTPs on the cluster. (Default: ``None``).
        image (Image, optional): Default image containing setup steps to run during cluster setup. See :class:`Image`.
            (Default: ``None``)
        den_auth (bool, optional): Whether to use Den authorization on the server. If ``True``, will validate incoming
            requests with a Runhouse token provided in the auth headers of the request with the format:
            ``{"Authorization": "Bearer <token>"}``. (Default: ``None``).
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
    ips = kwargs.get("ips") or ([host] if isinstance(host, str) else host)

    cluster_args = locals().copy()
    cluster_args["ips"] = ips
    cluster_args["creds"] = cluster_args.pop("ssh_creds")
    cluster_args.pop("kwargs")
    cluster_args.pop("host")
    cluster_args = {k: v for k, v in cluster_args.items() if v is not None}

    # check for invalid args
    valid_kwargs = {*ONDEMAND_COMPUTE_ARGS, *KUBERNETES_CLUSTER_ARGS, *RH_SERVER_ARGS}
    unsupported_kwargs = kwargs.keys() - valid_kwargs
    if unsupported_kwargs:
        raise ValueError(
            f"Received unsupported kwargs {unsupported_kwargs}. "
            "Please refer to `rh.cluster` or `rh.ondemand_cluster` for valid input args."
        )

    if kwargs.keys() & {*ONDEMAND_COMPUTE_ARGS, *KUBERNETES_CLUSTER_ARGS}:
        if kwargs.keys() & STATIC_CLUSTER_ARGS:
            raise ValueError(
                "Received incompatible args specific to both a static and ondemand cluster."
            )
        return ondemand_cluster(
            **cluster_args,
            **kwargs,
        )

    try:
        new_cluster = Cluster.from_name(
            name, load_from_den=load_from_den, dryrun=dryrun
        )
        if isinstance(new_cluster, OnDemandCluster):
            # load from name, none of the other arguments were provided
            cluster_type = "ondemand"
        else:
            cluster_type = "static"
    except ValueError:
        new_cluster = None
        cluster_type = "unsaved"

    if cluster_type == "unsaved" and cluster_args.keys() == {
        "name",
        "load_from_den",
        "dryrun",
    }:
        raise ValueError(
            f"Cluster {name} not found in Den. Must provide cluster arguments to construct "
            "a new cluster object."
        )

    cluster_args["creds"] = cluster_args.get("creds", rns_client.default_ssh_key)
    if cluster_type == "unsaved":
        new_cluster = Cluster(**cluster_args)
    elif cluster_type == "static":
        mismatches = _config_and_args_mismatches(new_cluster.config(), cluster_args)
        server_mismatches = mismatches.keys() & RH_SERVER_ARGS
        if "ips" in mismatches:
            new_cluster = Cluster(**cluster_args)
        elif server_mismatches:
            if new_cluster.is_up():
                logger.warning(
                    "Runhouse server setting has been updated. Please run `cluster.restart_server()` "
                    f"to apply new server settings for {server_mismatches}"
                )
            new_cluster = Cluster(**cluster_args)

    new_cluster._set_connection_defaults()

    if den_auth:
        new_cluster.save()
    return new_cluster


def ondemand_cluster(
    name,
    # sky arguments
    instance_type: Optional[str] = None,
    num_nodes: Optional[int] = None,
    provider: Optional[str] = None,
    autostop_mins: Optional[int] = None,
    use_spot: bool = False,
    region: Optional[str] = None,
    memory: Union[int, str, None] = None,
    disk_size: Union[int, str, None] = None,
    num_cpus: Union[int, str, None] = None,
    accelerators: Union[int, str, None] = None,
    gpus: Union[int, str, None] = None,
    open_ports: Union[int, str, List[int], None] = None,
    vpc_name: Optional[str] = None,
    sky_kwargs: Dict = None,
    # kubernetes related arguments
    kube_namespace: Optional[str] = None,
    kube_config_path: Optional[str] = None,
    kube_context: Optional[str] = None,
    # runhouse server arguments
    server_port: int = None,
    server_host: str = None,
    server_connection_type: Union[ServerConnectionType, str] = None,
    ssl_keyfile: str = None,
    ssl_certfile: str = None,
    domain: str = None,
    image: Image = None,
    # misc arguments
    launcher: Union[LauncherType, str] = None,
    den_auth: bool = None,
    load_from_den: bool = True,
    dryrun: bool = False,
):
    """
    Builds an instance of :class:`OnDemandCluster`.

    * If Cluster with same name is found in Den and ``load_from_den`` is ``True``, load it down from Den
    * If launch arguments are mismatched with loaded Cluster, return a new Cluster with the provided args.
      These args are passed through to SkyPilot's `Resource constructor
      <https://skypilot.readthedocs.io/en/latest/reference/api.html#resources>`__: ``instance_type``,
      ``num_nodes``, ``provider``, ``use_spot``, ``region``, ``memory``, ``disk_size``, ``num_cpus``,
      ``gpus`` (``accelerators``), ``open_ports``, ``autostop_mins``, ``sky_kwargs``.
    * If runhouse related arguments are mismatched with loaded Cluster, override those Cluster properties


    Args:
        name (str): Name for the cluster, to re-use later on.
        instance_type (int, optional): Type of cloud VM type to use for the cluster, e.g. "r5d.xlarge".
            Optional, as may instead choose to specify resource requirements (e.g. memory, disk_size,
            num_cpus, gpus).
        num_nodes (int, optional): Number of nodes to use for the cluster.
        provider (str, optional): Cloud provider to use for the cluster.
        autostop_mins (int, optional): Number of minutes to keep the cluster up after inactivity,
            or ``-1`` to keep cluster up indefinitely. (Default: ``60``).
        use_spot (bool, optional): Whether or not to use spot instance. (Default: ``False``)
        region (str, optional): The region to use for the cluster. (Default: ``None``)
        memory (int or str, optional): Amount of memory to use for the cluster, e.g. `16` or "16+".
            (Default: ``None``)
        disk_size (int or str, optional): Amount of disk space to use for the cluster, e.g. `100` or "100+".
            (Default: ``None``)
        num_cpus (int or str, optional): Number of CPUs to use for the cluster, e.g. `4` or "4+". (Default: ``None``)
        gpus (int or str, optional): Type and number of GPU to use for the cluster e.g. "A101" or "L4:8".
            (Default: ``None``)
        open_ports (int or str or List[int], optional): Ports to open in the cluster's security group. Note
            that you are responsible for ensuring that the applications listening on these ports are secure.
            (Default: ``None``)
        vpc_name (str, optional): Specific VPC used for launching the cluster. If not specified,
            cluster will be launched in the default VPC.
        sky_kwargs (dict, optional): Additional keyword arguments to pass to the SkyPilot `Resource` or `launch`
            APIs. Should be a dict of the form `{"resources": {<resources_kwargs>}, "launch": {<launch_kwargs>}}`,
            where resources_kwargs and launch_kwargs will be passed to the SkyPilot Resources API (See
            `SkyPilot docs <https://skypilot.readthedocs.io/en/latest/reference/api.html#resources>`__) and `launch`
            API (See `SkyPilot docs <https://skypilot.readthedocs.io/en/latest/reference/api.html#sky-launch>`__),
            respectively. Duplicating arguments passed to the `ondemand_cluster` factory method will raise an error.
            (Default: ``None``)
        kube_namespace (str, optional): Namespace for kubernetes cluster, if applicable. (Default: ``None``)
        kube_config_path (str, optional): Path to the kube_config, for a kubernetes cluster. (Default: ``None``)
        kube_context (str, optional): Context for kubernetes cluster, if applicable. (Default: ``None``)
        server_port (bool, optional): Port to use for the server. If not provided will use 80 for a
            ``server_connection_type`` of ``none``, 443 for ``tls`` and ``32300`` for all other SSH connection types.
            (Default: ``None``)
        server_host (bool, optional): Host from which the server listens for traffic (i.e. the --host argument
            `runhouse server start` run on the cluster). Defaults to "0.0.0.0" unless connecting to the server with an SSH
            connection, in which case ``localhost`` is used. (Default: ``None``)
        server_connection_type (ServerConnectionType or str, optional): Type of connection to use for the Runhouse
            API server. ``ssh`` will use start with server via an SSH tunnel. ``tls`` will start the server
            with HTTPS on port 443 using TLS certs without an SSH tunnel. ``none`` will start the server with HTTP
            without an SSH tunnel. (Default: ``None``)
        launcher (LauncherType or str, optional): Method for launching the cluster. If set to `local`, will launch
            locally via Sky. If set to `den`, launching will be handled by Runhouse. If not provided, will be set
            to your configured default launcher, which defaults to ``local``. (Default: ``None``)
        ssl_keyfile(str, optional): Path to SSL key file to use for launching the API server with HTTPS. (Default:
            ``None``)
        ssl_certfile (str, optional): Path to SSL certificate file to use for launching the API server with HTTPS.
            (Default: ``None``)
        domain (str, optional): Domain name for the cluster. Relevant if enabling HTTPs on the cluster.
            (Default: ``None``)
        image (Image, optional): Default image containing setup steps to run during cluster setup. See :class:`Image`.
            (Default: ``None``)
        den_auth (bool, optional): Whether to use Den authorization on the server. If ``True``, will validate incoming
            requests with a Runhouse token provided in the auth headers of the request with the format:
            ``{"Authorization": "Bearer <token>"}``. (Default: ``None``).
        load_from_den (bool): Whether to try loading the Cluster resource from Den. (Default: ``True``)
        dryrun (bool): Whether to create the Cluster if it doesn't exist, or load a Cluster object as a dryrun.
            (Default: ``False``)

    Returns:
        OnDemandCluster: The resulting cluster.

    Example:
        >>> # On-Demand SkyPilot Cluster (OnDemandCluster)
        >>> gpu = rh.ondemand_cluster(name='rh-4-a100s',
        >>>                  instance_type='A100:4',
        >>>                  provider='gcp',
        >>>                  autostop_mins=-1,
        >>>                  use_spot=True,
        >>>                  region='us-east-1',
        >>>                  ).save()

        >>> # Load cluster from above
        >>> reloaded_cluster = rh.ondemand_cluster(name="rh-4-a100s")
    """
    if vpc_name and launcher == "local":
        raise ValueError(
            "Custom VPCs are not supported with local launching. To use a custom VPC, please use the "
            "Den launcher. For more information see "
            "https://www.run.house/docs/installation-setup#den-launcher"
        )

    cluster_args = locals().copy()
    cluster_args = {k: v for k, v in cluster_args.items() if v is not None}
    if "accelerators" in cluster_args:
        logger.warning(
            "``accelerators`` argument has been deprecated. Please use ``gpus`` argument instead."
        )
        cluster_args["gpus"] = cluster_args.pop("accelerators")

    try:
        new_cluster = Cluster.from_name(
            name, load_from_den=load_from_den, dryrun=dryrun
        )
        cluster_type = "ondemand"
    except ValueError:
        new_cluster = None
        cluster_type = "unsaved"

    if cluster_args.keys() & KUBERNETES_CLUSTER_ARGS:
        setup_kubernetes(**cluster_args)

    if cluster_type == "unsaved":
        if cluster_args.keys() == {"name", "use_spot", "load_from_den", "dryrun"}:
            raise ValueError(
                f"OndemandCluster {name} not found in Den. Must provide cluster arguments to construct "
                "a new cluster object."
            )
        new_cluster = OnDemandCluster(**cluster_args)
    elif cluster_type == "ondemand":
        mismatches = _config_and_args_mismatches(new_cluster.config(), cluster_args)
        compute_mismatches = {
            k: v
            for k, v in mismatches.items()
            if k in {*ONDEMAND_COMPUTE_ARGS, *KUBERNETES_CLUSTER_ARGS}
        }
        server_mismatches = {k: v for k, v in mismatches.items() if k in RH_SERVER_ARGS}
        new_autostop_mins = compute_mismatches.pop("autostop_mins", None)

        if mismatches and new_cluster.is_up():
            # cluster is up - throw error if launch compute mismatches, but allow server / autostop min updates
            if compute_mismatches:
                raise ValueError(
                    f"Cluster {name} is up, but received argument mismatches for compute: {compute_mismatches.keys()}. "
                    "Please construct a new cluster object or ensure that the arguments match."
                )
            if new_autostop_mins:
                logger.info(f"Updating autostop mins for cluster {name}")
                new_cluster.autostop_mins = new_autostop_mins
            if server_mismatches:
                new_cluster._update_values(server_mismatches)
                logger.warning(
                    "Runhouse server setting has been updated. Please run `cluster.restart_server()` "
                    f"to apply new server settings for {server_mismatches.keys()}"
                )
        else:
            # cluster is down
            # - construct new cluster if launch compute mismatches
            # - if compute matches/empty but server or autostop mismatches, override just those values
            if compute_mismatches:
                new_cluster = OnDemandCluster(**cluster_args)
            elif server_mismatches:
                new_cluster._update_values(server_mismatches)
            if new_autostop_mins:
                new_cluster._autostop_mins = new_autostop_mins

    new_cluster._set_connection_defaults()
    if den_auth:
        new_cluster.save()
    return new_cluster
