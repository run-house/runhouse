import copy
import logging
import os
import subprocess
import warnings
from pathlib import Path

import sky
from sshtunnel import SSHTunnelForwarder

from runhouse.globals import open_cluster_tunnels
from runhouse.resources.hardware.utils import ServerConnectionType, SkySSHRunner
from runhouse.servers.http import HTTPClient

from .on_demand_cluster import OnDemandCluster

logger = logging.getLogger(__name__)


class KubernetesCluster(OnDemandCluster):
    RESOURCE_TYPE = "cluster"
    RECONNECT_TIMEOUT = 5

    def __init__(
        self,
        name,
        instance_type: str = None,
        namespace: str = None,
        kube_config_path: str = None,
        context: str = None,
        **kwargs,
    ):

        kwargs.pop("provider", None)
        kwargs.pop("server_connection_type", None)
        super().__init__(
            name=name,
            provider="kubernetes",
            instance_type=instance_type,
            server_connection_type=ServerConnectionType.SSH,
            **kwargs,
        )

        # TODO: extract namespace off context
        # Cases that need to be handled:
        # 1. User passes context and no namespace. Namespace needs to be extracted from context and set to it. 
        # 2. User passes namespace and no context. Namespace needs to be set with kubectl cmd (This should update the kubeconfig). Handled 
        # 3. User passes neither. Then, namespace needs to be extracted from current context
        # 4. User passes both namespace and context. Warn user. 

        self.namespace = namespace
        self.kube_config_path = kube_config_path
        self.context = context

    @staticmethod
    def from_config(config: dict, dryrun=False):
        return KubernetesCluster(**config, dryrun=dryrun)

    @property
    def config_for_rns(self):
        config = super().config_for_rns

        # Also store the ssh keys for the cluster in RNS
        config.update(
            {
                "instance_type": self.instance_type,
                "num_instances": self.num_instances,
                "provider": self.provider,
                "autostop_mins": self.autostop_mins,
                "open_ports": self.open_ports,
                "use_spot": self.use_spot,
                "image_id": self.image_id,
                "region": self.region,
                "live_state": self._get_sky_state(),
                "namespace": self.namespace,
                "kube_config_path": self.kube_config_path,
                "context": self.context,
            }
        )
        return config

    def ssh(self):
        """SSH into the cluster.

        Example:
            >>> cluster = rh.kubernetes_cluster (
                    name="cpu-cluster-05",
                    instance_type="1CPU--1GB",
                )
            >>> cluster.ssh()
        """
        # Get pod name
        command = f"kubectl get pods | grep {self.name}"

        try:

            output = subprocess.check_output(command, shell=True, text=True)

            lines = output.strip().split("\n")
            if lines:
                pod_name = lines[0].split()[0]
            else:
                logger.info("No matching pods found.")
        except subprocess.CalledProcessError as e:
            logger.info(f"Error: {e}")

        cmd = f"kubectl exec -it {pod_name} -- /bin/bash"
        subprocess.run(cmd, shell=True, check=True)
