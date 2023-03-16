import contextlib
import logging
import subprocess
from pathlib import Path
from typing import Any, Dict

import ray

import sky
import yaml
from sky.backends import backend_utils, CloudVmRayBackend

from runhouse.rh_config import configs, rns_client

from runhouse.rns.hardware.cluster import Cluster
from runhouse.rns.obj_store import _current_cluster

logger = logging.getLogger(__name__)


class OnDemandCluster(Cluster):
    RESOURCE_TYPE = "cluster"
    RECONNECT_TIMEOUT = 5

    def __init__(
        self,
        name,
        instance_type: str = None,
        num_instances: int = None,
        provider: str = None,
        dryrun=True,
        autostop_mins=None,
        use_spot=False,
        image_id=None,
        region=None,
        sky_state=None,
        **kwargs,  # We have this here to ignore extra arguments when calling from from_config
    ):
        """
        On-demand `SkyPilot <https://github.com/skypilot-org/skypilot/>`_ Cluster.

        .. note::
            To build a cluster, please use the factory method :func:`cluster`.
        """

        super().__init__(name=name, dryrun=dryrun)

        self.instance_type = instance_type
        self.num_instances = num_instances
        self.provider = provider or configs.get("default_provider")
        self.autostop_mins = (
            autostop_mins
            if autostop_mins is not None
            else configs.get("default_autostop")
        )
        self.use_spot = use_spot if use_spot is not None else configs.get("use_spot")
        self.image_id = image_id
        self.region = region

        self.address = None
        self._grpc_tunnel = None
        self.client = None
        self.sky_state = sky_state

        # Checks if state info is in local sky db, populates if so.
        status_dict = self.status(refresh=False)
        if status_dict:
            self._populate_connection_from_status_dict(status_dict)
        elif self.sky_state:
            self._save_sky_state()

        if not self.address and not dryrun:
            # Cluster status is set to INIT in the Sky DB right after starting, so we need to refresh once
            self.update_from_sky_status(dryrun=False)

    @staticmethod
    def from_config(config: dict, dryrun=False):
        return OnDemandCluster(**config, dryrun=dryrun)

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
                "use_spot": self.use_spot,
                "image_id": self.image_id,
                "region": self.region,
                "sky_state": self._get_sky_state(),
            }
        )
        return config

    def _get_sky_state(self):
        config = sky.global_user_state.get_cluster_from_name(self.name)
        if not config:
            return None
        config["status"] = config[
            "status"
        ].name  # ClusterStatus enum is not json serializable
        if config["handle"]:
            # with open(config["handle"].cluster_yaml, mode="r") as f:
            #     config["ray_config"] = yaml.safe_load(f)
            config["public_key"] = self.ssh_creds()["ssh_private_key"] + ".pub"
            config["handle"] = {
                "cluster_name": config["handle"].cluster_name,
                # This is saved as an absolute path - convert it to relative
                "cluster_yaml": self.relative_yaml_path(
                    yaml_path=config["handle"]._cluster_yaml
                ),
                "head_ip": config["handle"].head_ip,
                "launched_nodes": config["handle"].launched_nodes,
                "launched_resources": config[
                    "handle"
                ].launched_resources.to_yaml_config(),
            }
            config["handle"]["launched_resources"].pop("spot_recovery", None)

            config["ssh_creds"] = self.ssh_creds()
        return config

    def _copy_sky_yaml_from_cluster(self, abs_yaml_path: str):
        if not Path(abs_yaml_path).exists():
            Path(abs_yaml_path).parent.mkdir(parents=True, exist_ok=True)
            self.rsync("~/.sky/sky_ray.yml", abs_yaml_path, up=False)

            # Save SSH info to the ~/.ssh/config
            ray_yaml = yaml.safe_load(open(abs_yaml_path, "r"))
            backend_utils.SSHConfigHelper.add_cluster(
                self.name, [self.address], ray_yaml["auth"]
            )

    def _save_sky_state(self):
        if not self.sky_state:
            raise ValueError("No sky state to save")

        # if we're on this cluster, no need to save sky state
        current_cluster_name = _current_cluster("cluster_name")
        if self.sky_state.get("handle", {}).get("cluster_name") == current_cluster_name:
            return

        handle_info = self.sky_state.get("handle", {})

        # If we already have the cluster in local sky db,
        # we don't need to save the state, just populate the connection info from the status
        if not sky.global_user_state.get_cluster_from_name(self.name):
            # Try running a command on the cluster before saving down the state into sky db
            self.address = handle_info.get("head_ip")
            self._ssh_creds = self.sky_state["ssh_creds"]

            try:
                self.ping(timeout=self.RECONNECT_TIMEOUT)
            except TimeoutError:
                self.address = None
                self._ssh_creds = None
                print(
                    f"Timeout when trying to connect to cluster {self.name}, treating cluster as down."
                )
                return

            # TODO [JL] we need to call `ray.shutdown()` as sky does a ray init here on the cluster
            ray.shutdown()

            resources = sky.Resources.from_yaml_config(
                handle_info["launched_resources"]
            )
            # Need to convert to relative to find the yaml file in a new environment
            yaml_path = self.relative_yaml_path(handle_info.get("cluster_yaml"))
            handle = CloudVmRayBackend.ResourceHandle(
                cluster_name=self.name,
                cluster_yaml=str(Path(yaml_path).expanduser()),
                launched_nodes=handle_info["launched_nodes"],
                # head_ip=handle_info['head_ip'], # deprecated
                launched_resources=resources,
            )
            sky.global_user_state.add_or_update_cluster(
                cluster_name=self.name,
                cluster_handle=handle,
                requested_resources=[resources],
                is_launch=True,
                ready=False,
            )

        # Now try loading in the status from the sky DB
        status = self.status(refresh=False)

        abs_yaml_path = status["handle"].cluster_yaml
        try:
            if not Path(abs_yaml_path).exists():
                # This is also a good way to check if the cluster is still up
                self._copy_sky_yaml_from_cluster(abs_yaml_path)
            else:
                # We still should check if the cluster is up, since the status/yaml file could be stale
                self.ping(timeout=self.RECONNECT_TIMEOUT)
        except Exception:
            # Refresh the cluster status before saving the ssh info so SkyPilot has a chance to wipe the .ssh/config if
            # the cluster went down
            self.update_from_sky_status(dryrun=self.dryrun)

    def __getstate__(self):
        """Make sure sky_state is loaded in before pickling."""
        self.sky_state = self._get_sky_state()
        return super().__getstate__()

    @staticmethod
    def relative_yaml_path(yaml_path):
        if Path(yaml_path).is_absolute():
            yaml_path = "~/.sky/generated/" + Path(yaml_path).name
        return yaml_path

    # ----------------- Launch/Lifecycle Methods -----------------

    def is_up(self) -> bool:
        """Whether the cluster is up."""
        self.update_from_sky_status(dryrun=False)
        return self.address is not None

    def status(self, refresh: bool = True):
        """
        Get status of Sky cluster.

        Return dict looks like:

        .. code-block::

            {'name': 'sky-cpunode-donny',
             'launched_at': 1662317201,
             'handle': ResourceHandle(
                          cluster_name=sky-cpunode-donny,
                          head_ip=54.211.97.164,
                          cluster_yaml=/Users/donny/.sky/generated/sky-cpunode-donny.yml,
                          launched_resources=1x AWS(m6i.2xlarge),
                          tpu_create_script=None,
                          tpu_delete_script=None),
             'last_use': 'sky cpunode',
             'status': <ClusterStatus.UP: 'UP'>,
             'autostop': -1,
             'metadata': {}}

        .. note::
            For more information:
            https://github.com/skypilot-org/skypilot/blob/0c2b291b03abe486b521b40a3069195e56b62324/sky/backends/cloud_vm_ray_backend.py#L1457
        """  # noqa
        # return backend_utils._refresh_cluster_record(
        #     self.name, force_refresh=refresh, acquire_per_cluster_status_lock=False
        # )
        if not sky.global_user_state.get_cluster_from_name(self.name):
            return None

        state = sky.status(cluster_names=[self.name], refresh=refresh)
        # We still need to check if the cluster present in case the cluster went down and was removed from the DB
        if len(state) == 0:
            return None
        return state[0]

    def _populate_connection_from_status_dict(self, cluster_dict: Dict[str, Any]):
        if cluster_dict and cluster_dict["status"].name in ["UP", "INIT"]:
            self.address = cluster_dict["handle"].head_ip
            yaml_path = cluster_dict["handle"].cluster_yaml
            if Path(yaml_path).exists():
                self._ssh_creds = backend_utils.ssh_credential_from_yaml(yaml_path)
        else:
            self.address = None
            self._ssh_creds = None

    def update_from_sky_status(self, dryrun: bool = False):
        # Try to get the cluster status from SkyDB
        cluster_dict = self.status(refresh=not dryrun)
        self._populate_connection_from_status_dict(cluster_dict)

    def up(self):
        """Up the cluster."""
        if self.provider in ["aws", "gcp", "azure", "lambda", "cheapest"]:
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
            sky.launch(
                task,
                cluster_name=self.name,
                idle_minutes_to_autostop=self.autostop_mins,
                down=True,
            )
        elif self.provider == "k8s":
            raise NotImplementedError("Kubernetes Cluster provider not yet supported")
        else:
            raise ValueError(f"Cluster provider {self.provider} not supported.")

        self.update_from_sky_status()
        self.restart_grpc_server()

    def keep_warm(self, autostop_mins: int = -1):
        """Keep the cluster warm for given number of minutes after inactivity. If `autostop_mins` is set
        to -1, keep cluster warm indefinitely."""
        sky.autostop(self.name, autostop_mins, down=True)
        self.autostop_mins = autostop_mins

    def teardown(self):
        """Teardown cluster."""
        # Stream logs
        sky.down(self.name)
        self.address = None

    def teardown_and_delete(self):
        """Teardown cluster and delete it from configs."""
        self.teardown()
        rns_client.delete_configs()

    @contextlib.contextmanager
    def pause_autostop(self):
        sky.autostop(self.name, idle_minutes=-1)
        yield
        sky.autostop(self.name, idle_minutes=self.autostop_mins)

    # ----------------- SSH Methods ----------------- #

    @staticmethod
    def cluster_ssh_key(path_to_file):
        """Retrieve SSH key for the cluster."""
        try:
            f = open(path_to_file, "r")
            private_key = f.read()
            return private_key
        except FileNotFoundError:
            raise Exception(f"File with ssh key not found in: {path_to_file}")

    def ssh_creds(self):
        if self._ssh_creds:
            return self._ssh_creds

        if not self.status(refresh=False) and self.sky_state:
            # If this cluster was serialized and sent over the wire, it will have sky_state (we make sure of that
            # in __getstate__) but no yaml, and we need to save down the sky data to the sky db and local yaml
            self._save_sky_state()
        else:
            # To avoid calling this twice (once in save_sky_data)
            self.update_from_sky_status(dryrun=True)

        return self._ssh_creds

    def ssh(self):
        """SSH into the cluster."""
        subprocess.run(["ssh", f"{self.name}"])
