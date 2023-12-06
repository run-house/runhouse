import sky
from pathlib import Path
import subprocess
import logging
import os
from runhouse.resources.hardware.utils import SkySSHRunner
import copy
import warnings 
from runhouse.globals import obj_store, open_cluster_tunnels, rns_client
from runhouse.servers.http import HTTPClient
from sshtunnel import HandlerSSHTunnelForwarderError, SSHTunnelForwarder
from runhouse.resources.hardware.utils import (
    ServerConnectionType,
    SkySSHRunner,
    SshMode,
)

from .on_demand_cluster import OnDemandCluster

logger = logging.getLogger(__name__)

class KubernetesCluster(OnDemandCluster):
    RESOURCE_TYPE = "cluster"
    RECONNECT_TIMEOUT = 5

    def __init__(
        self,
        name,
        instance_type: str = None,
        namespace: str = "default",
        kube_config_path: str = None,
        context: str = None,
        **kwargs,
    ):
        
        kwargs.pop("provider", None)
        super().__init__(
            name=name,
            provider="kubernetes",
            instance_type=instance_type,
            **kwargs,
        )

        # TODO: extract namespace off context 
        # Cases that need to be handled: 
        # 1. User passes context and no namespace. Namespace needs to be extracted from context and set to it. 
        # 2. User passes namespace and no context. Namespace needs to be set with kubectl cmd (This should update the kubeconfig). 
        # 3. User passes neither. Then, namespace needs to be extracted from current context
        # 4. User passes both namespace and context. Invalid. Warn user and ignore namespace argument. Set namespace to be value extracted from context. 

        self.namespace = namespace
        self.kube_config_path = kube_config_path
        self.context = context

        if self.context is not None and self.namespace is not None:
            warnings.warn("You passed both a context and a namespace. The namespace will be ignored.", UserWarning)
            self.namespace = None

        
        if self.namespace is not None and self.namespace != "default": # check if user passed a user-defined namespace
            cmd = f"kubectl config set-context --current --namespace={self.namespace}"
            try:
                process = subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                print(process.stdout)
                print(f"Kubernetes namespace set to {self.namespace}")

            except subprocess.CalledProcessError as e:   
                print(f"Error: {e}")


        if self.kube_config_path is not None:     # check if user passed a user-defined kube_config_path 
            kube_config_dir = os.path.expanduser("~/.kube")
            kube_config_path_rl = os.path.join(kube_config_dir, "config") 

            if not os.path.exists(kube_config_dir): # check if ~/.kube directory exists on local machine
                try:
                    os.makedirs(kube_config_dir)     # create ~/.kube directory if it doesn't exist
                    print(f"Created directory: {kube_config_dir}")
                except OSError as e:
                    print(f"Error creating directory: {e}")

            try:
                cmd = f"cp {self.kube_config_path} {kube_config_path_rl}"  # copy user-defined kube_config to ~/.kube/config
                subprocess.run(cmd, shell=True, check=True)
                print(f"Copied kubeconfig to: {kube_config_path}") # note: this will overwrite any existing kubeconfig in ~/.kube/config
            except subprocess.CalledProcessError as e:
                print(f"Error copying kubeconfig: {e}")


        if self.context is not None: # check if user passed a user-defined context
            try:
                cmd = f"kubectl config use-context {self.context}" # set user-defined context as current context
                subprocess.run(cmd, shell=True, check=True) 
                print(f"Kubernetes context has been set to: {self.context}") 
            except subprocess.CalledProcessError as e: 
                print(f"Error setting context: {e}") 



    def sky_up(self):

        task = sky.Task(
            num_nodes=1, # TODO: Add Multi-node support for Kubernetes in Runhouse. May need to use `setup` and `set_file_mounts` in Sky API 
        )
        cloud_provider = (
            sky.clouds.CLOUD_REGISTRY.from_str(self.provider)
        )
        task.set_resources(
            sky.Resources(
                cloud=cloud_provider,
                instance_type=self.instance_type
                if self.instance_type
                and "--" in self.instance_type 
                else None, 
            )
        )
        if Path("~/.rh").expanduser().exists():
            task.set_file_mounts(
                {
                    "~/.rh": "~/.rh",
                }
            )
            
        sky.launch(
            task,
            cluster_name=self.name,
            idle_minutes_to_autostop=self.autostop_mins,
            down=True,
        )


    def up(self):
        if self.on_this_cluster():
            return self
        
        self.sky_up()
    
        self._update_from_sky_status()
        self.restart_server(restart_ray=True)

        return self
        
    def _rsync(
        self,
        source: str,
        dest: str,
        up: bool,
        contents: bool = False,
        filter_options: str = None,
    ):
        """
        Sync the contents of the source directory into the destination.

        .. note:
            Ending `source` with a slash will copy the contents of the directory into dest,
            while omitting it will copy the directory itself (adding a directory layer).
        """
        # FYI, could be useful: https://github.com/gchamon/sysrsync
        if contents:
            source = source + "/" if not source.endswith("/") else source
            dest = dest + "/" if not dest.endswith("/") else dest

        ssh_credentials = copy.copy(self.ssh_creds())
        ssh_credentials.pop("ssh_host", self.address)

        if not ssh_credentials.get("password"):
            # Use SkyPilot command runner
            if not ssh_credentials.get("ssh_private_key"):
                ssh_credentials["ssh_private_key"] = None
            runner = SkySSHRunner(self.address, **ssh_credentials)
            if up:
                runner.run(["mkdir", "-p", dest], stream_logs=False)
            else:
                Path(dest).expanduser().parent.mkdir(parents=True, exist_ok=True)
            runner.rsync(
                source, dest, up=up, stream_logs=False # removed filter_options to work with new rsync in utils.py
            )
        else:
            if dest.startswith("~/"):
                dest = dest[2:]

            self._fsspec_sync(source, dest, up)
            
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
        command = f"kubectl get pods -n {self.namespace} | grep {self.name}"

        try:

            output = subprocess.check_output(command, shell=True, text=True)

            lines = output.strip().split('\n')
            if lines:
                pod_name = lines[0].split()[0]
            else:
                print("No matching pods found.")
        except subprocess.CalledProcessError as e:
            print(f"Error: {e}")

        cmd = f"kubectl exec -it {pod_name} -- /bin/bash"
        subprocess.run(cmd, shell=True, check=True)

    def connect_server_client(self, force_reconnect=False):
        if not self.address:
            raise ValueError(f"No address set for cluster <{self.name}>. Is it up?")

        if self._rpc_tunnel and force_reconnect:
            self._rpc_tunnel.close()

        ssh_tunnel = None
        connected_port = None
        if (self.address, self.ssh_port) in open_cluster_tunnels:
            ssh_tunnel, connected_port = open_cluster_tunnels[
                (self.address, self.ssh_port)
            ]
            if isinstance(ssh_tunnel, SSHTunnelForwarder):
                ssh_tunnel.check_tunnels()
                if ssh_tunnel.tunnel_is_up[ssh_tunnel.local_bind_address]:
                    self._rpc_tunnel = ssh_tunnel

        if (
            self.server_connection_type
            not in [ServerConnectionType.NONE, ServerConnectionType.TLS]
            and ssh_tunnel is None
        ):
            # Case 3: server connection requires SSH tunnel, but we don't have one up yet
            self._rpc_tunnel, connected_port = self.ssh_tunnel(
                local_port=self.server_port,
                remote_port=self.server_port,
                num_ports_to_try=10,
            )

        open_cluster_tunnels[(self.address, self.ssh_port)] = (
            self._rpc_tunnel,
            connected_port,
        )

        if self._rpc_tunnel:
            logger.info(
                f"Connecting to server via SSH, port forwarding via port {connected_port}."
            )

        self.client_port = connected_port or self.client_port or self.server_port
        use_https = self._use_https
        cert_path = self.cert_config.cert_path if use_https else None

        # Connecting to localhost because it's tunneled into the server at the specified port.
        creds = self.ssh_creds()
        if self.server_connection_type in [
            ServerConnectionType.SSH,
            ServerConnectionType.AWS_SSM,
        ]:
            ssh_user = creds.get("ssh_user")
            password = creds.get("password")
            auth = (ssh_user, password) if ssh_user and password else None
            self.client = HTTPClient(
                host=self.LOCALHOST,
                port=self.client_port,
                auth=auth,
                cert_path=cert_path,
                use_https=use_https,
            )
        else:
            self.client = HTTPClient(
                host=self.LOCALHOST,   # k8s needs this to be localhost and not the server address. Server address is address of k8s pod
                port=self.client_port,
                cert_path=cert_path,
                use_https=use_https,
            )

