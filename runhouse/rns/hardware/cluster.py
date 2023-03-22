import contextlib
import logging
import pkgutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import grpc
import ray.cloudpickle as pickle
import sshtunnel

from sky.utils import command_runner
from sshtunnel import HandlerSSHTunnelForwarderError, SSHTunnelForwarder

from runhouse.rh_config import open_grpc_tunnels, rns_client
from runhouse.rns.packages.package import Package

from runhouse.rns.resource import Resource

from runhouse.servers.grpc.unary_client import UnaryClient
from runhouse.servers.grpc.unary_server import UnaryService

logger = logging.getLogger(__name__)


class Cluster(Resource):
    RESOURCE_TYPE = "cluster"
    GRPC_TIMEOUT = 5  # seconds

    def __init__(
        self,
        name,
        ips: List[str] = None,
        ssh_creds: Dict = None,
        dryrun=True,
        **kwargs,  # We have this here to ignore extra arguments when calling from from_config
    ):
        """
        The Runhouse cluster, or system. This is where you can run Functions or access/transfer data
        between. You can BYO (bring-your-own) cluster by providing cluster IP and ssh_creds, or
        this can be an on-demand cluster that is spun up/down through
        `SkyPilot <https://github.com/skypilot-org/skypilot>`_, using your cloud credentials.

        .. note::
            To build a cluster, please use the factory method :func:`cluster`.
        """

        super().__init__(name=name, dryrun=dryrun)

        self.address = ips[0] if ips else None
        self._ssh_creds = ssh_creds
        self.ips = ips
        self._grpc_tunnel = None
        self.client = None

        if not dryrun and self.address:
            # OnDemandCluster will start ray itself, but will also set address later, so won't reach here.
            self.start_ray()
            self.save_config_to_cluster()

    def save_config_to_cluster(self):
        import json

        config = self.config_for_rns
        if "sky_state" in config.keys():
            # a bunch of setup commands that mess up json dump
            del config["sky_state"]
        json_config = f"{json.dumps(config)}"

        self.run(
            [
                f"mkdir -p ~/.rh; touch ~/.rh/cluster_config.yaml; echo '{json_config}' > ~/.rh/cluster_config.yaml"
            ]
        )

    @staticmethod
    def from_config(config: dict, dryrun=False):
        # TODO use 'resource_subtype' in config?
        if "ips" in config:
            return Cluster(**config, dryrun=dryrun)
        else:
            from runhouse.rns.hardware import OnDemandCluster

            return OnDemandCluster(**config, dryrun=dryrun)

    @property
    def config_for_rns(self):
        config = super().config_for_rns
        self.save_attrs_to_config(config, ["ips"])
        if self.ips is not None:
            config["ssh_creds"] = self.ssh_creds()
        return config

    def is_up(self) -> bool:
        """Check if the cluster is up."""
        return self.address is not None

    def up_if_not(self):
        """Bring up the cluster if it is not up. No-op if cluster is already up."""
        if not self.is_up():
            if not hasattr(self, "up"):
                raise NotImplementedError(
                    f"Cluster <{self.name}> does not have an up method."
                )
            self.up()
        return self

    def keep_warm(self):
        logger.info(
            f"cluster.keep_warm will have no effect on self-managed cluster {self.name}."
        )

    def start_ray(self):
        if self.is_up():
            res = self.run(["ray start --head"], stream_logs=False)
            if res[0] == 0:
                return
            if any(
                line.startswith("ConnectionError: Ray is trying to start at")
                for line in res[0][1].splitlines()
            ):
                # Ray is already started
                return
            # Check if ray is installed
            if "ray" not in self._get_pip_installs(strip_versions=True):
                self.run(["pip install ray"])
                res = self.run(["ray start --head"])
                if not res[0][0]:
                    raise RuntimeError(
                        f"Failed to start ray on cluster <{self.name}>. "
                        f"Error: {res[0][1]}"
                    )
        else:
            raise ValueError(f"Cluster <{self.name}> is not up.")

    def _get_pip_installs(self, strip_versions=False):
        packages = self.run(["pip freeze"], stream_logs=False)[0][1].splitlines()
        if strip_versions:
            packages = [p.split("==")[0] for p in packages]
        return packages

    def sync_runhouse_to_cluster(self, _install_url=None):
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
            rh_package = Package.from_string(
                f"reqs:{local_rh_package_path}", dryrun=True
            )
            rh_package.to_cluster(self, mount=False)
            status_codes = self.run(["pip install ./runhouse"], stream_logs=True)
        # elif local_rh_package_path.parent.name == 'site-packages':
        else:
            # Package is installed in site-packages
            # status_codes = self.run(['pip install runhouse-nightly==0.0.2.20221202'], stream_logs=True)
            # rh_package = 'runhouse_nightly-0.0.1.dev20221202-py3-none-any.whl'
            # rh_download_cmd = f'curl https://runhouse-package.s3.amazonaws.com/{rh_package} --output {rh_package}'
            # TODO need to check user's current version and install same version?
            _install_url = _install_url or "runhouse"
            rh_install_cmd = f"pip install {_install_url}"
            status_codes = self.run([rh_install_cmd], stream_logs=True)

        if status_codes[0][0] != 0:
            raise ValueError(f"Error installing runhouse on cluster <{self.name}>")

    def install_packages(self, reqs: List[Union[Package, str]]):
        """Install the given packages on the cluster."""
        self.check_grpc()
        to_install = []
        for package in reqs:
            if isinstance(package, str):
                # If the package is a local folder, we need to create the package to sync it over to the cluster
                pkg_obj = Package.from_string(package, dryrun=False)
            else:
                pkg_obj = package

            from runhouse.rns.folders.folder import Folder

            if (
                isinstance(pkg_obj.install_target, Folder)
                and pkg_obj.install_target.is_local()
            ):
                pkg_str = pkg_obj.name or Path(pkg_obj.install_target.path).name
                logging.info(
                    f"Copying local package {pkg_str} to cluster <{self.name}>"
                )
                remote_package = pkg_obj.to_cluster(self, mount=False)
                to_install.append(remote_package)
            else:
                to_install.append(package)  # Just appending the string!
        logging.info(
            f"Installing packages on cluster {self.name}: "
            f"{[req if isinstance(req, str) else str(req) for req in reqs]}"
        )
        self.client.install_packages(to_install)

    def get(self, key: str, default: Any = None, stream_logs: bool = False):
        """Get the object at the given key from the cluster's object store."""
        self.check_grpc()
        return self.client.get_object(key, stream_logs=stream_logs) or default

    def add_secrets(self, provider_secrets: dict):
        """Copy secrets from current environment onto the cluster"""
        self.check_grpc()
        return self.client.add_secrets(pickle.dumps(provider_secrets))

    def put(self, key: str, obj: Any):
        """Put the given object on the cluster's object store at the given key."""
        self.check_grpc()
        return self.client.put_object(key, obj)

    # TODO [DG] add a method to list all the keys in the cluster

    def cancel(self, key, force=False):
        """Cancel the given run on cluster."""
        self.check_grpc()
        return self.client.cancel_runs(key, force=force)

    def clear_pins(self, pins: Optional[List[str]] = None):
        """Remove the given pinned items from the cluster. If `pins` is set to ``None``, then
        all pinned objects will be cleared."""
        self.check_grpc()
        self.client.clear_pins(pins)
        logger.info(f'Clearing pins on cluster {pins or ""}')

    def on_same_cluster(self, resource: Resource):
        """Whether the given resource is on the cluster."""
        if hasattr(resource, "system") and isinstance(resource.system, Resource):
            return resource.system.rns_address == self.rns_address
        return False

    # ----------------- gRPC Methods ----------------- #

    def connect_grpc(self, force_reconnect=False):
        # FYI based on: https://sshtunnel.readthedocs.io/en/latest/#example-1
        # FYI If we ever need to do this from scratch, we can use this example:
        # https://github.com/paramiko/paramiko/blob/main/demos/rforward.py#L74
        if not self.address:
            raise ValueError(f"No address set for cluster <{self.name}>. Is it up?")

        # TODO [DG] figure out how to ping to see if tunnel is already up
        if self._grpc_tunnel and force_reconnect:
            self._grpc_tunnel.close()

        # TODO Check if port is already open instead of refcounting?
        # status = subprocess.run(['nc', '-z', self.address, str(self.grpc_port)], capture_output=True)
        # if not self.check_port(self.address, UnaryClient.DEFAULT_PORT):

        tunnel_refcount = 0
        if self.address in open_grpc_tunnels:
            ssh_tunnel, connected_port, tunnel_refcount = open_grpc_tunnels[
                self.address
            ]
            ssh_tunnel.check_tunnels()
            if ssh_tunnel.tunnel_is_up[ssh_tunnel.local_bind_address]:
                self._grpc_tunnel = ssh_tunnel
        else:
            self._grpc_tunnel, connected_port = self.ssh_tunnel(
                UnaryClient.DEFAULT_PORT,
                remote_port=UnaryService.DEFAULT_PORT,
                num_ports_to_try=5,
            )
        open_grpc_tunnels[self.address] = (
            self._grpc_tunnel,
            connected_port,
            tunnel_refcount + 1,
        )

        # Connecting to localhost because it's tunneled into the server at the specified port.
        self.client = UnaryClient(host="127.0.0.1", port=connected_port)
        waited = 0
        while not self.is_connected() and waited <= self.GRPC_TIMEOUT:
            time.sleep(0.25)
            waited += 0.25

    def check_grpc(self, restart_grpc_server=True):
        if not self.address:
            # For OnDemandCluster, this initial check doesn't trigger a sky.status, which is slow.
            # If cluster simply doesn't have an address we likely need to up it.
            if not hasattr(self, "up"):
                raise ValueError(
                    "Cluster must have an ip address (i.e. be up) or have a reup_cluster method "
                    "(e.g. OnDemandCluster)."
                )
            if not self.is_up():
                # If this is a OnDemandCluster, before we up the cluster, run a sky.status to see if the cluster
                # is already up but doesn't have an address assigned yet.
                self.up_if_not()

        if not self.client:
            try:
                self.connect_grpc()
            except (
                grpc.RpcError,
                sshtunnel.BaseSSHTunnelForwarderError,
            ):
                # It's possible that the cluster went down while we were trying to install packages.
                if not self.is_up():
                    self.up_if_not()
                else:
                    self.restart_grpc_server(resync_rh=False)

        if self.is_connected():
            return

        self.connect_grpc()
        if self.is_connected():
            return

        if restart_grpc_server:
            self.restart_grpc_server(resync_rh=False)
            self.connect_grpc()
            if self.is_connected():
                return

            self.restart_grpc_server(resync_rh=True)
            self.connect_grpc()
            if self.is_connected():
                return

        raise ValueError(f"Could not connect to cluster <{self.name}>")

        # try:
        #     self.client.ping()
        # except Exception as e:
        #     if restart_if_down:
        #         self.restart_grpc_server(resync_rh=resync_rh)
        #         self.connect_grpc(force_reconnect=True)
        #         self.client.ping()
        #     else:
        #         raise e

    @staticmethod
    def check_port(ip_address, port):
        import socket

        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        return s.connect_ex(("127.0.0.1", int(port)))

    def ssh_tunnel(
        self, local_port, remote_port=None, num_ports_to_try: int = 0
    ) -> Tuple[SSHTunnelForwarder, int]:
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
                    raise Exception(
                        f"Failed to create SSH tunnel after {num_ports_to_try} attempts"
                    )

                ssh_tunnel = SSHTunnelForwarder(
                    self.address,
                    ssh_username=creds["ssh_user"],
                    ssh_pkey=creds["ssh_private_key"],
                    local_bind_address=("", local_port),
                    remote_bind_address=("127.0.0.1", remote_port or local_port),
                    set_keepalive=1,
                    # mute_exceptions=True,
                )
                ssh_tunnel.start()
                connected = True
            except HandlerSSHTunnelForwarderError:
                # try connecting with a different port - most likely the issue is the port is already taken
                local_port += 1
                num_ports_to_try -= 1
                pass

        return ssh_tunnel, local_port

    # TODO [DG] Remove this for now, for some reason it was causing execution to hang after programs completed
    # def __del__(self):
    #     if self.address in open_grpc_tunnels:
    #         tunnel, port, refcount = open_grpc_tunnels[self.address]
    #         if refcount == 1:
    #             tunnel.close()
    #             open_grpc_tunnels.pop(self.address)
    #         else:
    #             open_grpc_tunnels[self.address] = (tunnel, port, refcount - 1)
    #     elif self._grpc_tunnel:  # Not sure why this would be reached but keeping it just in case
    #         self._grpc_tunnel.close()

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

    def restart_grpc_server(
        self,
        _rh_install_url: str = None,
        resync_rh: bool = True,
        restart_ray: bool = False,
    ):
        """Restart the GRPC server."""
        # TODO how do we capture errors if this fails?
        if resync_rh:
            self.sync_runhouse_to_cluster(_install_url=_rh_install_url)
            self.save_config_to_cluster()
        kill_proc_cmd = 'pkill -f "python3 -m runhouse.servers.grpc.unary_server"'
        logfile = f"{self.name}_grpc_server.log"
        grpc_server_cmd = "python3 -m runhouse.servers.grpc.unary_server"
        # 2>&1 redirects stderr to stdout
        screen_cmd = f"screen -dm bash -c '{grpc_server_cmd} >> ~/.rh/{logfile} 2>&1'"
        cmds = [kill_proc_cmd]
        if restart_ray:
            cmds.append("ray stop")
            cmds.append(
                "ray start --head"
            )  # Need to set gpus or Ray will block on cpu-only clusters
        cmds.append(screen_cmd)

        # If we need different commands for debian or ubuntu, we can use this:
        # Need to get actual provider in case provider == 'cheapest'
        # handle = sky.global_user_state.get_cluster_from_name(self.name)['handle']
        # cloud_provider = str(handle.launched_resources.cloud)
        # ubuntu_kill_proc_cmd = f'fuser -k {UnaryService.DEFAULT_PORT}/tcp'
        # debian_kill_proc_cmd = "kill -9 $(netstat -anp | grep 50052 | grep -o '[0-9]*/' | sed 's+/$++')"
        # f'kill -9 $(lsof -t -i:{UnaryService.DEFAULT_PORT})'
        # kill_proc_at_port_cmd = debian_kill_proc_cmd if cloud_provider == 'GCP' \
        #     else ubuntu_kill_proc_cmd

        status_codes = self.run(
            commands=cmds,
            stream_logs=True,
        )
        # As of 2022-27-Dec still seems we need this.
        time.sleep(2)
        return status_codes

    @contextlib.contextmanager
    def pause_autostop(self):
        """Context manager to temporarily pause autostop. Mainly for OnDemand clusters, for BYO cluster
        there is no autostop."""
        pass

    def run_module(self, relative_path, module_name, fn_name, fn_type, args, kwargs):
        self.check_grpc()
        return self.client.run_module(
            relative_path, module_name, fn_name, fn_type, args, kwargs
        )

    def is_connected(self):
        return self.client is not None and self.client.is_connected()

    def disconnect(self):
        if self._grpc_tunnel:
            self._grpc_tunnel.stop()
        # if self.client:
        #     self.client.shutdown()

    def __getstate__(self):
        """Delete non-serializable elements (e.g. thread locks) before pickling."""
        state = self.__dict__.copy()
        state["client"] = None
        state["_grpc_tunnel"] = None
        return state

    # ----------------- SSH Methods ----------------- #

    def ssh_creds(self):
        return self._ssh_creds

    def rsync(self, source: str, dest: str, up: bool, contents: bool = False):
        """
        Sync the contents of the source directory into the destination.

        .. note:
            Ending `source` with a slash will copy the contents of the directory into dest,
            while omitting it will copy the directory itself (adding a directory layer)."""
        # FYI, could be useful: https://github.com/gchamon/sysrsync
        if contents:
            source = source + "/" if not source.endswith("/") else source
            dest = dest + "/" if not dest.endswith("/") else dest
        ssh_credentials = self.ssh_creds()
        runner = command_runner.SSHCommandRunner(self.address, **ssh_credentials)
        if up:
            runner.run(["mkdir", "-p", dest], stream_logs=False)
        else:
            Path(dest).expanduser().parent.mkdir(parents=True, exist_ok=True)
        runner.rsync(source, dest, up=up, stream_logs=False)

    def ssh(self):
        creds = self.ssh_creds()
        subprocess.run(
            f"ssh {creds['ssh_user']}:{self.address} -i {creds['ssh_private_key']}".split(
                " "
            )
        )

    def ping(self, timeout=5):
        ssh_call = threading.Thread(target=lambda: self.run(['echo "hello"']))
        ssh_call.start()
        ssh_call.join(timeout=timeout)
        if ssh_call.is_alive():
            raise TimeoutError("SSH call timed out")
        return True

    def run(
        self,
        commands: List[str],
        stream_logs: bool = True,
        port_forward: Optional[int] = None,
        require_outputs: bool = True,
    ):
        """Run a list of shell commands on the cluster."""
        # TODO [DG] suspect autostop while running?
        runner = command_runner.SSHCommandRunner(self.address, **self.ssh_creds())
        return_codes = []
        for command in commands:
            logger.info(f"Running command on {self.name}: {command}")
            ret_code = runner.run(
                command,
                require_outputs=require_outputs,
                stream_logs=stream_logs,
                port_forward=port_forward,
            )
            return_codes.append(ret_code)
        return return_codes

    def run_python(
        self,
        commands: List[str],
        stream_logs: bool = True,
        port_forward: Optional[int] = None,
    ):
        """Run a list of python commands on the cluster."""
        command_str = "; ".join(commands)
        return_codes = self.run(
            [f'python3 -c "{command_str}"'],
            stream_logs=stream_logs,
            port_forward=port_forward,
        )
        return return_codes

    def send_secrets(self, providers: Optional[List[str]] = None):
        """Send secrets for the given providers. If none provided will send secrets for providers that have been
        configured in the environment."""
        from runhouse import Secrets

        Secrets.to(hardware=self, providers=providers)

    def ipython(self):
        # TODO tunnel into python interpreter in cluster
        pass

    def notebook(
        self,
        persist: bool = False,
        sync_package_on_close: Optional[str] = None,
        port_forward: int = 8888,
    ):
        """Tunnel into and launch notebook from the cluster."""
        tunnel, port_fwd = self.ssh_tunnel(local_port=port_forward, num_ports_to_try=10)
        try:
            install_cmd = "pip install jupyterlab"
            jupyter_cmd = f"jupyter lab --port {port_fwd} --no-browser"
            with self.pause_autostop():
                self.run(commands=[install_cmd, jupyter_cmd], stream_logs=True)

        finally:
            if sync_package_on_close:
                if sync_package_on_close == "./":
                    sync_package_on_close = rns_client.locate_working_dir()
                pkg = Package.from_string("local:" + sync_package_on_close)
                self.rsync(source=f"~/{pkg.name}", dest=pkg.local_path, up=False)
            if not persist:
                tunnel.stop()
                kill_jupyter_cmd = f"jupyter notebook stop {port_fwd}"
                self.run(commands=[kill_jupyter_cmd])
