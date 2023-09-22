import sky
from pathlib import Path
import subprocess
import logging
import time

from runhouse.resources.hardware.on_demand_cluster import OnDemandCluster

logger = logging.getLogger(__name__)

class KubernetesCluster(OnDemandCluster):

    def up(self):
        """Up the cluster.

        Example:
            >>> rh.kubernetes_cluster(
            >>>     name="cpu-cluster-test",
            >>>     instance_type="CPU:1",
            >>>     provider="kubernetes",      
            >>> )
        """
        if self.on_this_cluster():
            return self

        if self.provider in ["kubernetes"]:
            task = sky.Task(
                num_nodes=self.num_instances
                if self.instance_type and ":" not in self.instance_type
                else None,
                # docker_image=image,  # Zongheng: this is experimental, don't use it
                # envs=None,
            )
            cloud_provider = (
                sky.clouds.CLOUD_REGISTRY.from_str(self.provider)
                if self.provider != "cheapest"
                else None
            )
            task.set_resources(
                sky.Resources(
                    cloud=cloud_provider,
                    instance_type=self.instance_type
                    if self.instance_type
                    and ":" not in self.instance_type
                    and "CPU" not in self.instance_type
                    else None,
                    accelerators=self.instance_type
                    if self.instance_type
                    and ":" in self.instance_type
                    and "CPU" not in self.instance_type
                    else None,
                    cpus=self.instance_type.rsplit(":", 1)[1]
                    if self.instance_type
                    and ":" in self.instance_type
                    and "CPU" in self.instance_type
                    else None,
                    region=self.region,
                    image_id=self.image_id,
                    use_spot=self.use_spot,
                )
            )
            if Path("~/.rh").expanduser().exists():
                task.set_file_mounts(
                    {
                        "~/.rh": "~/.rh",
                    }
                )
            # If we choose to reduce collisions of cluster names:
            # cluster_name = self.rns_address.strip('~/').replace("/", "-")
            sky.launch(
                task,
                cluster_name=self.name,
                idle_minutes_to_autostop=self.autostop_mins,
                down=True,
            )
        else:
            raise ValueError(f"Cluster provider {self.provider} not supported.")

        self._update_from_sky_status()
        
        self.restart_server(restart_ray=True)

        return self

    def restart_server(
        self,
        _rh_install_url: str = None,
        resync_rh: bool = True,
        restart_ray: bool = True,
        env_activate_cmd: str = None,
    ):
        """Restart the RPC server.

        Args:
            resync_rh (bool): Whether to resync runhouse. (Default: ``True``)
            restart_ray (bool): Whether to restart Ray. (Default: ``True``)
            env_activate_cmd (str, optional): Command to activate the environment on the server. (Default: ``None``)
        Example:
            >>> rh.kubernetes_cluster("rh-cpu").restart_server()
        """
        logger.info(f"Restarting HTTP server on {self.name}.")

        if resync_rh:

            # sync_rh_cmd = "kubectl cp ../runhouse/ default/cpu-cluster-42-dc19-ray-head:runhouse"

            # consider modifying this so that the kubernetes namespace is parameterized 
            # also consider extracting the kubernetes pod name 
            sync_rh_cmd = f"kubectl cp ../runhouse/ default/{self.name}-dc19-ray-head:runhouse"

            cmd = f"{sync_rh_cmd}"
            try:
                subprocess.run(cmd, shell=True, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error: {e}")

        install_rh_cmd = "pip install ./runhouse"

        cmd_1 = self.CLI_RESTART_CMD + (" --no-restart-ray" if not restart_ray else "")
        cmd_1 = f"{install_rh_cmd} && {cmd_1}"

        status_codes = self.run(commands=[cmd_1])
        if not status_codes[0][0] == 0:
            raise ValueError(f"Failed to restart server {self.name}.")
        # As of 2023-15-May still seems we need this.
        time.sleep(5)
        return status_codes