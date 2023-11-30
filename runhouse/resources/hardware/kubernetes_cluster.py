import sky
from pathlib import Path
import subprocess
import logging
import time
import pkgutil
import yaml

from .on_demand_cluster import OnDemandCluster

logger = logging.getLogger(__name__)

class KubernetesCluster(OnDemandCluster):
    RESOURCE_TYPE = "cluster"
    RECONNECT_TIMEOUT = 5
    
    def __init__(
        self,
        name,
        instance_type: str = None,
        **kwargs,
    ):
        
        super().__init__(
            name=name,
            provider="kubernetes",
        )

        self.instance_type = instance_type

    def sky_up(self, local_rh_package_path):

        task = sky.Task(
            num_nodes=1, # TODO: Add Multi-node support for Kubernetes in Runhouse
            setup='python3 -m pip install ../runhouse',
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
                    "~/runhouse": f"{local_rh_package_path}"
                }
            )
            
        sky.launch(
            task,
            cluster_name=self.name,
            # idle_minutes_to_autostop=self.autostop_mins,
            # down=True,
        )


    def up(self):
        if self.on_this_cluster():
            return self
        
        local_rh_package_path = Path(pkgutil.get_loader("runhouse").path).parent

        if (
            local_rh_package_path.parent.name == "runhouse"
            and (local_rh_package_path.parent / "setup.py").exists()
        ):
            # Package is installed in editable mode
            local_rh_package_path = local_rh_package_path.parent
        
        self.sky_up(local_rh_package_path)
    
        self._update_from_sky_status()
        self.restart_server(restart_ray=True, resync_rh=False)

        return self
    
    def _sync_runhouse_to_cluster(self, _install_url=None, env=None):
        if not self.address:
            raise ValueError(f"No address set for cluster <{self.name}>. Is it up?")

        local_rh_package_path = Path(pkgutil.get_loader("runhouse").path).parent

        # Check if runhouse is installed from source and has setup.py
        if (
            not _install_url
            and local_rh_package_path.parent.name == "runhouse"
            and (local_rh_package_path.parent / "setup.py").exists()
        ):
            # Package is installed in editable mode
            local_rh_package_path = local_rh_package_path.parent

            rh_install_cmd = "python3 -m pip install ./runhouse"

            self.sky_up(local_rh_package_path)

        else:
            # Package is installed in site-packages
            # status_codes = self.run(['pip install runhouse-nightly==0.0.2.20221202'], stream_logs=True)
            # rh_package = 'runhouse_nightly-0.0.1.dev20221202-py3-none-any.whl'
            # rh_download_cmd = f'curl https://runhouse-package.s3.amazonaws.com/{rh_package} --output {rh_package}'
            if not _install_url:
                import runhouse

                _install_url = f"runhouse=={runhouse.__version__}"
            rh_install_cmd = f"python3 -m pip install {_install_url}"

        install_cmd = f"{env._run_cmd} {rh_install_cmd}" if env else rh_install_cmd

        status_codes = self.run([install_cmd], stream_logs=True)

        if status_codes[0][0] != 0:
            raise ValueError(f"Error installing runhouse on cluster <{self.name}>")
        
    # def restart_server(
    #     self,
    #     _rh_install_url: str = None,
    #     resync_rh: bool = True,
    #     restart_ray: bool = True,
    #     env_activate_cmd: str = None,
    #     restart_proxy: bool = False,
    # ):
    #     """Restart the RPC server.

    #     Args:
    #         resync_rh (bool): Whether to resync runhouse. (Default: ``True``)
    #         restart_ray (bool): Whether to restart Ray. (Default: ``True``)
    #         env_activate_cmd (str, optional): Command to activate the environment on the server. (Default: ``None``)
    #         restart_proxy (bool): Whether to restart nginx on the cluster, if configured. (Default: ``False``)
    #     Example:
    #         >>> rh.cluster("rh-cpu").restart_server()
    #     """
    #     logger.info(f"Restarting Runhouse API server on {self.name}.")

    #     if resync_rh:
    #         self._sync_runhouse_to_cluster(_install_url=_rh_install_url)

    #     # Update the cluster config on the cluster
    #     self.save_config_to_cluster()

    #     cmd = (
    #         self.CLI_RESTART_CMD
    #         + (" --no-restart-ray" if not restart_ray else "")
    #     )

    #     cmd = f"{env_activate_cmd} && {cmd}" if env_activate_cmd else cmd

    #     status_codes = self.run(commands=[cmd])
    #     if not status_codes[0][0] == 0:
    #         raise ValueError(f"Failed to restart server {self.name}.")

    #     return status_codes








