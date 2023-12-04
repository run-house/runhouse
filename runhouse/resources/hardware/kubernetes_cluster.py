import sky
from pathlib import Path
import subprocess
import logging
import os
from runhouse.resources.hardware.utils import SkySSHRunner
import copy
import warnings 

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

        self.namespace = namespace
        self.kube_config_path = kube_config_path
        self.context = context

        if self.instance_type is None:
            raise ValueError("You must specify an instance type")
        
        if self.name is None:
            raise ValueError("You must specify a name for your cluster")

        if self.context is not None and self.namespace is not None:
            warnings.warn("You passed both a context and a namespace. The namespace will be ignored.", UserWarning)
            self.namespace = None


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
