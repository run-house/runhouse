from typing import List, Optional, Union
import subprocess
import logging
from pathlib import Path
import pkgutil
import contextlib
import yaml

import sky
from sky.backends import backend_utils, CloudVmRayBackend
from sky.utils import command_runner
from sshtunnel import SSHTunnelForwarder, HandlerSSHTunnelForwarderError
import ray.cloudpickle as pickle

from runhouse.rns.resource import Resource
from runhouse.rns.packages.package import Package
from runhouse.grpc_handler.unary_client import UnaryClient
from runhouse.grpc_handler.unary_server import UnaryService
from runhouse.rh_config import rns_client, configs, open_grpc_tunnels

logger = logging.getLogger(__name__)


class Cluster(Resource):
    RESOURCE_TYPE = "cluster"
    DEFAULT_AUTOSTOP_MINS = 10

    def __init__(self,
                 name,
                 instance_type: str = None,
                 num_instances: int = None,
                 provider: str = None,
                 dryrun=True,
                 autostop_mins=None,
                 use_spot=False,
                 image_id=None,
                 sky_data=None,
                 **kwargs  # We have this here to ignore extra arguments when calling from from_config
                 ):
        """
        Args:
            name:
            instance_type: Type of cloud instance to use for the cluster
            num_instances: Number of instances to use for the cluster
            provider: Cloud provider to use for the cluster
            dryrun:
            autostop_mins: Number of minutes to keep the cluster up for following inactivity, or
                -1 to keep up indefinitely.
        """

        super().__init__(name=name, dryrun=dryrun)

        self.instance_type = instance_type
        self.num_instances = num_instances
        self.provider = provider or configs.get('default_provider')
        self.autostop_mins = autostop_mins if autostop_mins is not None \
            else configs.get('default_autostop')
        self.use_spot = use_spot if use_spot is not None else configs.get('use_spot')
        self.image_id = image_id

        self.address = None
        self._yaml_path = None
        self._grpc_tunnel = None
        self._secrets_sent = False
        self.client = None
        self.sky_data = sky_data
        if self.sky_data is not None:
            self._save_sky_data()

        # Checks local SkyDB if cluster is up, and loads connection info if so.
        self.populate_vars_from_status(dryrun=True)

        # Cluster status is set to INIT in the Sky DB right after starting, so we need to refresh once
        if not self.address:
            self.populate_vars_from_status(dryrun=False)

    @staticmethod
    def from_config(config: dict, dryrun=False):
        return Cluster(**config, dryrun=dryrun)

    @property
    def config_for_rns(self):
        config = super().config_for_rns

        # Also store the ssh keys for the cluster in RNS
        config.update({'instance_type': self.instance_type,
                       'num_instances': self.num_instances,
                       'provider': self.provider,
                       'autostop_mins': self.autostop_mins,
                       'use_spot': self.use_spot,
                       'image_id': self.image_id,
                       'sky_data': self._get_sky_data(),
                       })
        return config

    def _get_sky_data(self):
        config = sky.global_user_state.get_cluster_from_name(self.name)
        if not config:
            return None
        config['status'] = config['status'].name  # ClusterStatus enum is not json serializable
        if config['handle']:
            with open(config['handle'].cluster_yaml, mode='r') as f:
                config['ray_config'] = yaml.safe_load(f)
            config['public_key'] = self.ssh_creds()['ssh_private_key'] + '.pub'
            config['handle'] = {'cluster_name': config['handle'].cluster_name,
                                # This is saved as an absolute path - convert it to relative
                                'cluster_yaml': self.relative_yaml_path(yaml_path=config['handle']._cluster_yaml),
                                'head_ip': config['handle'].head_ip,
                                'launched_nodes': config['handle'].launched_nodes,
                                'launched_resources': config['handle'].launched_resources.to_yaml_config()
                                }
            config['handle']['launched_resources'].pop('spot_recovery', None)

            pub_key_path = self.ssh_creds()['ssh_private_key'] + '.pub'
            if Path(pub_key_path).exists():
                with open(pub_key_path, mode='r') as f:
                    config['public_key'] = f.read()
        return config

    def _save_sky_data(self):
        # If we already have an entry for this cluster in the local sky files, ignore the new config
        # TODO [DG] when this is more stable maybe we shouldn't.
        if sky.global_user_state.get_cluster_from_name(self.name) and self._yaml_path and \
                Path(self._yaml_path).absolute().exists():
            return

        ray_config = self.sky_data.pop('ray_config', {})
        handle_info = self.sky_data.pop('handle', {})
        if not ray_config or not handle_info:
            raise Exception('Expecting both `ray_config` and `handle` attributes in sky data')

        yaml_path = handle_info['cluster_yaml']
        if not Path(yaml_path).expanduser().parent.exists():
            Path(yaml_path).expanduser().parent.mkdir(parents=True, exist_ok=True)

        with Path(yaml_path).expanduser().open(mode='w+') as f:
            yaml.safe_dump(ray_config, f)

        cluster_abs_path = str(Path(yaml_path).expanduser())
        cloud_provider = sky.clouds.CLOUD_REGISTRY.from_str(handle_info['launched_resources']['cloud'])
        backend_utils._add_auth_to_cluster_config(cloud_provider, cluster_abs_path)

        handle = CloudVmRayBackend.ResourceHandle(
            cluster_name=self.name,
            cluster_yaml=cluster_abs_path,
            launched_nodes=handle_info['launched_nodes'],
            # head_ip=handle_info['head_ip'], # deprecated
            launched_resources=sky.Resources.from_yaml_config(handle_info['launched_resources']),
        )
        sky.global_user_state.add_or_update_cluster(self.name,
                                                    cluster_handle=handle,
                                                    is_launch=True,
                                                    ready=False)
        backend_utils.SSHConfigHelper.add_cluster(
            self.name, [handle_info['head_ip']], ray_config['auth'])

    def __getstate__(self):
        """ Make sure sky_data is loaded in before pickling. """
        self.sky_data = self._get_sky_data()
        state = self.__dict__.copy()
        return state

    @staticmethod
    def relative_yaml_path(yaml_path):
        if Path(yaml_path).is_absolute():
            yaml_path = '~/.sky/generated/' + Path(yaml_path).name
        return yaml_path

    # ----------------- Launch/Lifecycle Methods -----------------

    # TODO [DG] this sometimes returns True when cluster is not up
    def is_up(self) -> bool:
        self.populate_vars_from_status(dryrun=False)
        return self.address is not None

    def status(self, refresh=True):
        """
        Return dict looks like:
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
        More: https://github.com/skypilot-org/skypilot/blob/0c2b291b03abe486b521b40a3069195e56b62324/sky/backends/cloud_vm_ray_backend.py#L1457
        """
        return self.get_sky_statuses(cluster_name=self.name, refresh=refresh)

    def populate_vars_from_status(self, dryrun=False):
        # Try to get the cluster status from SkyDB
        cluster_dict = self.status(refresh=not dryrun)
        if not cluster_dict:
            return
        self.address = cluster_dict['handle'].head_ip
        self._yaml_path = cluster_dict['handle'].cluster_yaml
        self.autostop_mins = cluster_dict['autostop']
        self.provider = str(cluster_dict['handle'].launched_resources.cloud).lower()
        if not cluster_dict['status'].name == 'UP':
            self.address = None

    @staticmethod
    def get_sky_statuses(cluster_name: str = None, refresh: bool = True):
        """
        Get status dicts for all Sky clusters.
        Args:
            cluster_name (str): Return status dict for only specific cluster.

        Returns:

        """
        # TODO [DG] just get status for this cluster
        all_clusters_status = sky.status(refresh=refresh)
        if not cluster_name:
            return all_clusters_status
        for cluster_dict in all_clusters_status:
            if cluster_dict['name'] == cluster_name:
                return cluster_dict

    def up_if_not(self):
        if not self.is_up():
            self.up()
        return self

    def up(self,
           region: str = None,
           ):
        if self.provider in ['aws', 'gcp', 'azure', 'cheapest']:
            task = sky.Task(  # run=f'echo SkyPilot cluster {self.name} launched.',
                num_nodes=self.num_instances if ':' not in self.instance_type else None,
                # workdir=package,
                # docker_image=image,  # Zongheng: this is experimental, don't use it
                # envs=None,
            )
            cloud_provider = sky.clouds.CLOUD_REGISTRY.from_str(self.provider) \
                if self.provider != 'cheapest' else None
            task.set_resources(
                sky.Resources(
                    cloud=cloud_provider,
                    instance_type=self.instance_type if ':' not in self.instance_type else None,
                    accelerators=self.instance_type if ':' in self.instance_type else None,
                    region=region,  # TODO
                    image_id=self.image_id,
                    use_spot=self.use_spot
                )
            )
            if Path('~/.rh').expanduser().exists():
                task.set_file_mounts({
                    '~/.rh': '~/.rh',
                })
            sky.launch(task,
                       cluster_name=self.name,
                       idle_minutes_to_autostop=self.autostop_mins,
                       down=True,
                       )
        elif self.provider == 'k8s':
            # TODO ssh in and do for real
            # https://docs.ray.io/en/latest/cluster/kubernetes/user-guides/config.html#kuberay-config
            # subprocess.Popen('kubectl apply -f raycluster.yaml'.split(' '))
            # self.address = cluster_dict['handle'].head_ip
            # self._yaml_path = cluster_dict['handle'].cluster_yaml
            raise NotImplementedError(f'Kubernetes Cluster provider not yet supported')
        else:
            raise ValueError(f'Cluster provider {self.provider} not supported.')

        self.populate_vars_from_status()
        self.restart_grpc_server()

    def sync_runhouse_to_cluster(self, _install_url=None):
        if not self.address:
            raise ValueError(f'No address set for cluster <{self.name}>. Is it up?')
        local_rh_package_path = Path(pkgutil.get_loader('runhouse').path).parent

        # Check if runhouse is installed from source and has setup.py
        if not _install_url and \
                local_rh_package_path.parent.name == 'runhouse' and \
                (local_rh_package_path.parent / 'setup.py').exists():
            # Package is installed in editable mode
            local_rh_package_path = local_rh_package_path.parent
            rh_package = Package.from_string(f'reqs:{local_rh_package_path}', dryrun=True)
            rh_package.to_cluster(self, mount=False)
            status_codes = self.run(['pip install ./runhouse'], stream_logs=True)
        # elif local_rh_package_path.parent.name == 'site-packages':
        else:
            # Package is installed in site-packages
            # status_codes = self.run(['pip install runhouse-nightly==0.0.2.20221202'], stream_logs=True)
            # rh_package = 'runhouse_nightly-0.0.1.dev20221202-py3-none-any.whl'
            # rh_download_cmd = f'curl https://runhouse-package.s3.amazonaws.com/{rh_package} --output {rh_package}'
            _install_url = _install_url or 'git+https://github.com/run-house/runhouse.git@latest_patch'
            rh_install_cmd = f'pip install {_install_url}'
            status_codes = self.run([rh_install_cmd], stream_logs=True)

        if status_codes[0][0] != 0:
            raise ValueError(f'Error installing runhouse on cluster <{self.name}>')

    def install_packages(self, reqs: List[Union[Package, str]]):
        if not self.is_connected():
            self.connect_grpc()
        to_install = []
        # TODO [DG] validate package strings
        for package in reqs:
            if isinstance(package, str):
                # If the package is a local folder, we need to create the package to sync it over to the cluster
                pkg_obj = Package.from_string(package, dryrun=False)
            else:
                pkg_obj = package

            from runhouse.rns.folders.folder import Folder
            if isinstance(pkg_obj.install_target, Folder) and \
                    pkg_obj.install_target.is_local():
                pkg_str = pkg_obj.name or Path(pkg_obj.install_target.url).name
                logging.info(f'Copying local package {pkg_str} to cluster <{self.name}>')
                remote_package = pkg_obj.to_cluster(self, mount=False, return_dest_folder=True)
                to_install.append(remote_package)
            else:
                to_install.append(package)  # Just appending the string!
        # TODO replace this with figuring out how to stream the logs when we install
        logging.info(f'Installing packages on cluster {self.name}: '
                     f'{[req if isinstance(req, str) else str(req) for req in reqs]}')
        self.client.install_packages(pickle.dumps(to_install))

    def flush_pins(self, pins: Optional[List[str]] = None):
        if not self.is_connected():
            self.connect_grpc()
        self.client.flush_pins(pins)
        logger.info(f'Clearing pins on cluster {pins or ""}')

    def keep_warm(self, autostop_mins=-1):
        sky.autostop(self.name, autostop_mins, down=True)
        self.autostop_mins = autostop_mins

    def teardown(self):
        # Stream logs
        sky.down(self.name)
        self.address = None

    def teardown_and_delete(self):
        self.teardown()
        rns_client.delete_configs()

    # ----------------- gRPC Methods ----------------- #

    def connect_grpc(self, force_reconnect=False):
        # FYI based on: https://sshtunnel.readthedocs.io/en/latest/#example-1
        # FYI If we ever need to do this from scratch, we can use this example:
        # https://github.com/paramiko/paramiko/blob/main/demos/rforward.py#L74
        if not self.address:
            raise ValueError(f'No address set for cluster <{self.name}>. Is it up?')

        # TODO [DG] figure out how to ping to see if tunnel is already up
        if self._grpc_tunnel and force_reconnect:
            self._grpc_tunnel.close()

        # TODO Check if port is already open instead of refcounting?
        # status = subprocess.run(['nc', '-z', self.address, str(self.grpc_port)], capture_output=True)
        # if not self.check_port(self.address, UnaryClient.DEFAULT_PORT):

        tunnel_refcount = 0
        if self.address in open_grpc_tunnels:
            ssh_tunnel, connected_port, tunnel_refcount = open_grpc_tunnels[self.address]
            ssh_tunnel.check_tunnels()
            if ssh_tunnel.tunnel_is_up[ssh_tunnel.local_bind_address]:
                self._grpc_tunnel = ssh_tunnel
        else:
            self._grpc_tunnel, connected_port = self.ssh_tunnel(UnaryClient.DEFAULT_PORT,
                                                                remote_port=UnaryService.DEFAULT_PORT,
                                                                num_ports_to_try=5)
        open_grpc_tunnels[self.address] = (self._grpc_tunnel, connected_port, tunnel_refcount + 1)

        # Connecting to localhost because it's tunneled into the server at the specified port.
        self.client = UnaryClient(host='127.0.0.1', port=connected_port)

    @staticmethod
    def check_port(ip_address, port):
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        return s.connect_ex(('127.0.0.1', int(port)))

    def ssh_tunnel(self,
                   local_port,
                   remote_port=None,
                   num_ports_to_try: int = 0) -> (SSHTunnelForwarder, int):
        # Debugging cmds (mac):
        # netstat -vanp tcp | grep 5005
        # lsof -i :5005_
        # kill -9 <pid>

        creds: dict = self.ssh_creds()
        connected = False
        ssh_tunnel = None
        while not connected:
            try:
                if local_port > local_port + num_ports_to_try:
                    raise Exception(f'Failed to create SSH tunnel after {num_ports_to_try} attempts')

                ssh_tunnel = SSHTunnelForwarder(
                    self.address,
                    ssh_username=creds['ssh_user'],
                    ssh_pkey=creds['ssh_private_key'],
                    local_bind_address=('', local_port),
                    remote_bind_address=('127.0.0.1', remote_port or local_port),
                    set_keepalive=1,
                    # mute_exceptions=True,
                )
                ssh_tunnel.start()
                connected = True
            except HandlerSSHTunnelForwarderError as e:
                # try connecting with a different port - most likely the issue is the port is already taken
                local_port += 1
                pass

        return ssh_tunnel, local_port

    # TODO [DG] Remove this for now, for some reason it was causing execution to hang after programs completed
    # def __del__(self):
    # if self.address in open_grpc_tunnels:
    #     tunnel, port, refcount = open_grpc_tunnels[self.address]
    #     if refcount == 1:
    #         tunnel.stop(force=True)
    #         open_grpc_tunnels.pop(self.address)
    #     else:
    #         open_grpc_tunnels[self.address] = (tunnel, port, refcount - 1)
    # elif self._grpc_tunnel:  # Not sure why this would be reached but keeping it just in case
    #     self._grpc_tunnel.stop(force=True)

    # import paramiko
    # ssh = paramiko.SSHClient()
    # ssh.load_system_host_keys()
    # ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    # from pathlib import Path
    # ssh.connect(self.address,
    #             username=creds['ssh_user'],
    #             key_filename=str(Path(creds['ssh_private_key']).expanduser())
    #             )
    # transport = ssh.get_transport()
    # transport.request_port_forward('', local_port)
    # ssh_tunnel = transport.open_channel("direct-tcpip", ("localhost", local_port),
    #                                     (self.address, remote_port or local_port))
    # if ssh_tunnel.is_active():
    #     connected = True
    #     print(f"SSH tunnel is open to {self.address}:{local_port}")

    def restart_grpc_server(self, _rh_install_url=None, resync_rh=True):
        # TODO how do we capture errors if this fails?
        if resync_rh:
            self.sync_runhouse_to_cluster(_install_url=_rh_install_url)
        grpc_server_cmd = f'screen -dm python3 -m runhouse.grpc_handler.unary_server'
        kill_proc_cmd = f'pkill -f "python3 -m runhouse.grpc_handler.unary_server"'

        # If we need different commands for debian or ubuntu, we can use this:
        # Need to get actual provider in case provider == 'cheapest'
        # handle = sky.global_user_state.get_cluster_from_name(self.name)['handle']
        # cloud_provider = str(handle.launched_resources.cloud)
        # ubuntu_kill_proc_cmd = f'fuser -k {UnaryService.DEFAULT_PORT}/tcp'
        # debian_kill_proc_cmd = "kill -9 $(netstat -anp | grep 50052 | grep -o '[0-9]*/' | sed 's+/$++')"
        # f'kill -9 $(lsof -t -i:{UnaryService.DEFAULT_PORT})'
        # kill_proc_at_port_cmd = debian_kill_proc_cmd if cloud_provider == 'GCP' \
        #     else ubuntu_kill_proc_cmd

        status_codes = self.run(commands=[kill_proc_cmd,
                                          grpc_server_cmd],
                                stream_logs=True,
                                )
        # As of 2022-27-Dec still seems we need this.
        import time
        time.sleep(2)
        return status_codes

    @contextlib.contextmanager
    def pause_autostop(self):
        sky.autostop(self.name, idle_minutes=-1)
        yield
        sky.autostop(self.name, idle_minutes=self.autostop_mins)

    def call_grpc(self, serialized_func):
        # TODO check if this is actually avoiding creating a duplicate tunnel when creating one send after another
        if not self.is_connected():
            self.connect_grpc()

        # TODO would be great to pause autostop here but it's too slow, about 2 seconds
        return self.client.call_fn_remotely(message=serialized_func)

    def is_connected(self):
        return self.client is not None and self.client.is_connected()

    def disconnect(self):
        if self._grpc_tunnel:
            self._grpc_tunnel.stop()
        # if self.client:
        #     self.client.shutdown()

    # ----------------- SSH Methods ----------------- #

    @staticmethod
    def cluster_ssh_key(path_to_file):
        try:
            f = open(path_to_file, 'r')
            private_key = f.read()
            return private_key
        except FileNotFoundError:
            raise Exception(f'File with ssh key not found in: {path_to_file}')

    @staticmethod
    def path_to_cluster_ssh_key(path_to_file) -> str:
        user_path = Path(path_to_file).expanduser()
        return str(user_path)

    def ssh_creds(self):
        # TODO [DG] handle if sky_data is empty (which shouldn't be possible).
        if not Path(self._yaml_path).exists():
            self._save_sky_data()
            self.populate_vars_from_status(dryrun=self.dryrun)

        return backend_utils.ssh_credential_from_yaml(self._yaml_path)

    def rsync(self, source, dest, up, contents=False):
        """ Note that ending `source` with a slash will copy the contents of the directory into dest,
        while omitting it will copy the directory itself (adding a directory layer)."""
        # FYI, could be useful: https://github.com/gchamon/sysrsync
        if contents:
            source = source + '/'
            dest = dest + '/'
        ssh_credentials = self.ssh_creds()
        runner = command_runner.SSHCommandRunner(self.address, **ssh_credentials)
        runner.rsync(source, dest, up=up, stream_logs=False)

    def ssh(self):
        subprocess.run(["ssh", f"{self.name}"])

    def run(self, commands: list, stream_logs=True, port_forward=None, require_outputs=True):
        """ Run a list of shell commands on the cluster. """
        # TODO add name parameter to create Run object, and use sky.exec (after updating to sky 2.0):
        # sky.exec(commands, cluster_name=self.name, stream_logs=stream_logs, detach=False)
        runner = command_runner.SSHCommandRunner(self.address, **self.ssh_creds())
        return_codes = []
        for command in commands:
            logger.info(f"Running command on {self.name}: {command}")
            ret_code = runner.run(command,
                                  require_outputs=require_outputs,
                                  stream_logs=stream_logs,
                                  port_forward=port_forward)
            return_codes.append(ret_code)
        return return_codes

    def run_python(self, commands: list, stream_logs=True, port_forward=None):
        """ Run a list of python commands on the cluster. """
        command_str = '; '.join(commands)
        self.run([f'python3 -c "{command_str}"'], stream_logs=stream_logs, port_forward=port_forward)

    def send_secrets(self, reload=False, providers: Optional[List[str]] = None):
        if providers is not None:
            # Send secrets for specific providers from local configs rather than trying to load from Vault
            from runhouse import Secrets
            secrets: list = Secrets.load_provider_secrets(providers=providers)
            # TODO [JL] change this API so we don't have to convert the list to a dict
            secrets: dict = {s['provider']: {k: v for k, v in s.items() if k != 'provider'} for s in secrets}
            load_secrets_cmd = ['import runhouse as rh',
                                f'rh.Secrets.save_provider_secrets(secrets={secrets})']
        elif not self._secrets_sent or reload:
            load_secrets_cmd = ['import runhouse as rh',
                                'rh.Secrets.download_into_env()']
        else:
            # Secrets already sent and not reloading
            return

        self.run_python(load_secrets_cmd, stream_logs=True)
        # TODO [JL] change this to a list to make sure new secrets get sent when the user wants to
        self._secrets_sent = True

    def ipython(self):
        # TODO tunnel into python interpreter in cluster
        pass

    def notebook(self, persist=False, sync_package_on_close=None, port_forward=8888):
        tunnel, port_fwd = self.ssh_tunnel(local_port=port_forward, num_ports_to_try=10)
        try:
            install_cmd = "pip install jupyterlab"
            jupyter_cmd = f'jupyter lab --port {port_fwd} --no-browser'
            # port_fwd = '-L localhost:8888:localhost:8888 '  # TOOD may need when we add docker support
            with self.pause_autostop():
                self.run(commands=[install_cmd, jupyter_cmd], stream_logs=True)

        finally:
            if sync_package_on_close:
                if sync_package_on_close == './':
                    sync_package_on_close = rns_client.locate_working_dir()
                pkg = Package.from_string('local:' + sync_package_on_close)
                self.rsync(source=f'~/{pkg.name}', dest=pkg.local_path, up=False)
            if not persist:
                tunnel.stop(force=True)
                kill_jupyter_cmd = f'jupyter notebook stop {port_fwd}'
                self.run(commands=[kill_jupyter_cmd])


# Cluster factory method
def cluster(name: str,
            instance_type: str = None,
            num_instances: int = None,
            provider: str = None,
            autostop_mins: int = None,
            dryrun: bool = False,
            use_spot: bool = None,
            image_id: str = None,
            ) -> Cluster:
    config = rns_client.load_config(name)
    config['name'] = name or config.get('rns_address', None) or config.get('name')

    config['instance_type'] = instance_type or config.get('instance_type', None)
    config['num_instances'] = num_instances or config.get('num_instances', None)
    config['provider'] = provider or config.get('provider', None)
    config['autostop_mins'] = autostop_mins if autostop_mins is not None else config.get('autostop_mins', None)
    config['use_spot'] = use_spot if use_spot is not None else config.get('use_spot', None)
    config['image_id'] = image_id if image_id is not None else config.get('image_id', None)

    new_cluster = Cluster.from_config(config, dryrun=dryrun)

    return new_cluster
