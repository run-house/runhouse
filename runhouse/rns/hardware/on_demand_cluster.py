import contextlib
import json
import logging
import subprocess
from pathlib import Path

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
        self._yaml_path = None
        self._grpc_tunnel = None
        self.client = None
        self.sky_state = sky_state

        # Checks local SkyDB if cluster is up, and loads connection info if so.
        self.update_from_sky_status(dryrun=True)

        if not self.address and not dryrun:
            if self.sky_state is not None:
                self._save_sky_state()
            else:
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

    def _copy_sky_yaml_from_cluster(self):
        if not Path(self._yaml_path).expanduser().exists():
            Path(self._yaml_path).expanduser().parent.mkdir(parents=True, exist_ok=True)
            self.rsync("~/.sky/sky_ray.yml", self._yaml_path, up=False)

    def _save_sky_state(self):
        # if we're on this cluster, no need to save sky state
        current_cluster_name = (
            _current_cluster().rsplit("/", 1)[-1] if _current_cluster() else None
        )
        if self.sky_state.get("handle", {}).get("cluster_name") == current_cluster_name:
            return

        yaml_path = self._yaml_path or self.sky_state.get("handle", {}).get("cluster_yaml")

        # convert to relative path
        yaml_path = self.relative_yaml_path(yaml_path)

        if (
            sky.global_user_state.get_cluster_from_name(self.name)
            and Path(yaml_path).expanduser().exists()
        ):
            self.update_from_sky_status(dryrun=True)
            return

        handle_info = self.sky_state.pop("handle", {})

        # cluster_abs_path = str(Path(yaml_path).expanduser())
        # cloud_provider = sky.clouds.CLOUD_REGISTRY.from_str(
        #     handle_info["launched_resources"]["cloud"]
        # )
        # backend_utils._add_auth_to_cluster_config(cloud_provider, cluster_abs_path)

        # TODO [JL] we need to call `ray.shutdown()` as sky does a ray init here on the cluster
        ray.shutdown()

        resources = sky.Resources.from_yaml_config(handle_info["launched_resources"])
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
        self._yaml_path = yaml_path
        self.address = handle_info["head_ip"]
        self._ssh_creds = self.sky_state.get("ssh_creds", {})
        self._copy_sky_yaml_from_cluster()

        # Refresh the cluster status before saving the ssh info so SkyPilot has a chance to wipe the .ssh/config if
        # the cluster went down
        self.update_from_sky_status(dryrun=False)
        ray_yaml = yaml.safe_load(open(self._yaml_path, "r"))
        if self.address:  # If cluster is found to still be up after refresh
            backend_utils.SSHConfigHelper.add_cluster(
                self.name, [handle_info["head_ip"]], ray_yaml["auth"]
            )

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

    def update_from_sky_status(self, dryrun: bool = False):
        # Try to get the cluster status from SkyDB
        cluster_dict = self.status(refresh=not dryrun)
        if not cluster_dict:
            return

        self._yaml_path = cluster_dict["handle"].cluster_yaml

        if not cluster_dict["status"].name == "UP":
            self.address = None
        else:
            ip = cluster_dict["handle"].head_ip
            if self.address is None or self.address != ip:
                self.address = ip
                self._ssh_creds = backend_utils.ssh_credential_from_yaml(self._yaml_path)

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
        if (
            not self._yaml_path
            or not Path(self._yaml_path).expanduser().exists()
            and self.sky_state
        ):
            # If this cluster was serialized and sent over the wire, it will have sky_state (we make sure of that
            # in __getstate__) but no yaml, and we need to save down the sky data to the sky db and local yaml
            self._save_sky_state()
        else:
            # To avoid calling this twice (once in save_sky_data)
            self.update_from_sky_status(dryrun=self.dryrun)

        return self._ssh_creds

    def ssh(self):
        """SSH into the cluster."""
        subprocess.run(["ssh", f"{self.name}"])
