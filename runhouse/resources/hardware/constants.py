STATIC_CLUSTER_ARGS = {
    "host",
    "ssh_creds",
}

ONDEMAND_COMPUTE_ARGS = {
    "instance_type",
    "num_nodes",
    "provider",
    "use_spot",
    "region",
    "memory",
    "disk_size",
    "vpc_name",
    "num_cpus",
    "accelerators",
    "gpus",
    "sky_kwargs",
    "launcher",
    "autostop_mins",
}

KUBERNETES_CLUSTER_ARGS = {
    "kube_context",
    "kube_namespace",
    "kube_config_path",
}

RH_SERVER_ARGS = {
    "server_port",
    "server_host",
    "ssh_port",
    "open_ports",  # ondemand only
    "server_connection_type",
    "ssl_keyfile",
    "ssl_certfile",
    "domain",
    "image",
}
