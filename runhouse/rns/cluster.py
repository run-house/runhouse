from pathlib import Path
import subprocess
import json

import yaml
from ray.autoscaler._private.commands import get_head_node_ip, get_worker_node_ips

from .rns_client import RNSClient

default_yaml = Path(__file__).parent / "rh-minimal.yaml"
default_clusters_dir = Path.home() / ".rh/clusters"
default_cluster_name = "default"

class Cluster:

    def __init__(self,
                 name=None,
                 yaml_path=None,
                 address=None,
                 create=True,
                 clusters_dir=None):
        self.name = name or default_cluster_name
        self.yaml_path = yaml_path
        self.address = address

        # Assumes that if the user passed a path and an address then they know the cluster is up,
        # so we're not pinging the cluster with every instantiation
        if self.yaml_path is None or self.address is None:
            # Create the cluster config directory, e.g. ~/.rh/clusters/<my_cluster_name>
            self.cluster_dir = Path(clusters_dir or default_clusters_dir, self.name)
            config = RNSClient().load_config_from_name(self.name,
                                                       resource_dir=self.cluster_dir,
                                                       resource_type="cluster")

            self.yaml_path = yaml_path or config.get('yaml_path', None)
            self.address = address or config.get('address', None)

            # Still no yaml, create one from template
            if self.yaml_path is None:
                # Check if yaml file is in directory in case the config didn't save,
                # sometimes the python client times out while the cluster is starting
                # If it exists, assume it's the right one, don't make a new one
                self.yaml_path = self.cluster_dir / f"{self.name}_ray_conf.yaml"
                if not self.yaml_path.exists():
                    cluster_yaml = yaml.safe_load(default_yaml.open('r'))
                    cluster_yaml['cluster_name'] = self.name
                    # TODO fix python mismatch business here too?
                    yaml.dump(cluster_yaml, self.yaml_path.open('w'))

            if self.address is None:
                # Also ensures cluster is up, and creates it if not
                self.address = self.get_cluster_address(create=create)

            config = {'name': self.name,
                      # TODO save full yaml file, not just path
                      'yaml_path': str(self.yaml_path),
                      'address': self.address}
            RNSClient().save_config_for_name(self.name,
                                             config,
                                             resource_dir=self.cluster_dir,
                                             resource_type="cluster")

    def get_cluster_address(self, create=True):
        try:
            # Private fn - Ray looks at tags of active EC2 instances through boto to find a node
            # with tags ray-node-type==head and ray-cluster-name==<name>
            # https://github.com/ray-project/ray/blob/releases/1.13.1/python/ray/autoscaler/_private/commands.py#L1264
            ip = get_head_node_ip(self.yaml_path)
        except RuntimeError as e:
            if create:
                # Cluster not up or not found, start new one
                subprocess.run(['ray', 'up', '-y', '--no-restart', self.yaml_path])
                ip = get_head_node_ip(self.yaml_path)
            else:
                raise e
        return f'ray://{ip}:10001'

    def ssh_into_head(self):
        subprocess.run(["ray", "attach", f"{self.yaml_path}"])

    # TODO untested strawman
    def ssh_into_worker(self, worker_index):
        # Private fn
        # https://github.com/ray-project/ray/blob/releases/1.13.1/python/ray/autoscaler/_private/commands.py#L1283
        ips = get_worker_node_ips(self.yaml_path)
        pem_path = None
        user = 'ubuntu'
        subprocess.run([f'ssh -tt -o IdentitiesOnly=yes -i {pem_path} {user}@{ips[worker_index]} '
                        f'docker exec -it ray_container /bin/bash'])

    def teardown(self):
        subprocess.run(["ray", "down", f"{self.yaml_path}"])
        self.address = None

    def teardown_and_delete(self):
        self.teardown()
        RNSClient().delete_configs(self.name, self.cluster_dir, 'cluster')
