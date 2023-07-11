import contextlib
import logging
import pkgutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import requests.exceptions
import sshtunnel

from sky.utils import command_runner
from sshtunnel import HandlerSSHTunnelForwarderError, SSHTunnelForwarder

from runhouse.rh_config import open_cluster_tunnels, rns_client
from runhouse.rns.folders.folder import Folder
from runhouse.rns.packages.package import Package
from runhouse.rns.resource import Resource
from runhouse.rns.utils.hardware import _current_cluster

from runhouse.servers.http import DEFAULT_SERVER_PORT, HTTPClient

logger = logging.getLogger(__name__)


class Cluster(Resource):
    RESOURCE_TYPE = "cluster"
    REQUEST_TIMEOUT = 5  # seconds

    def __init__(
        self,
        name,
        ips: List[str] = None,
        ssh_creds: Dict = None,
        dryrun=False,
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
        self._rpc_tunnel = None
        self.client = None

        if not dryrun and self.address:
            self.check_server()
            # OnDemandCluster will start ray itself, but will also set address later, so won't reach here.
            self.start_ray()

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
        resource_subtype = config.get("resource_subtype")
        if resource_subtype == "Cluster":
            return Cluster(**config, dryrun=dryrun)
        elif resource_subtype == "OnDemandCluster":
            from runhouse.rns.hardware import OnDemandCluster

            return OnDemandCluster(**config, dryrun=dryrun)
        else:
            raise ValueError(f"Unknown cluster type {resource_subtype}")

    @property
    def config_for_rns(self):
        config = super().config_for_rns
        self.save_attrs_to_config(config, ["ips"])
        if self.ips is not None:
            config["ssh_creds"] = self.ssh_creds()
        return config

    def is_up(self) -> bool:
        """Check if the cluster is up.

        Example:
            >>> rh.cluster("rh-cpu").is_up()
        """
        return self.address is not None

    def up_if_not(self):
        """Bring up the cluster if it is not up. No-op if cluster is already up.
        This only applies to on-demand clusters, and has no effect on self-managed clusters.

        Example:
            >>> rh.cluster("rh-cpu").up_if_not()
        """
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
        """Start Ray on the cluster.

        Example:
            >>> rh.cluster("rh-cpu").start_ray()
        """
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
                self.run(
                    ["pip install ray==2.4.0"]
                )  # pin to SkyPilot's Ray requirement
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
            dest_path = f"~/{local_rh_package_path.name}"

            # temp update rsync filters to exclude docs, when syncing over runhouse folder
            org_rsync_filter = command_runner.RSYNC_FILTER_OPTION
            command_runner.RSYNC_FILTER_OPTION = (
                "--filter='dir-merge,- .gitignore,- docs/'"
            )
            self._rsync(
                source=str(local_rh_package_path),
                dest=dest_path,
                up=True,
                contents=True,
            )
            command_runner.RSYNC_FILTER_OPTION = org_rsync_filter

            rh_install_cmd = "pip install ./runhouse"
        # elif local_rh_package_path.parent.name == 'site-packages':
        else:
            # Package is installed in site-packages
            # status_codes = self.run(['pip install runhouse-nightly==0.0.2.20221202'], stream_logs=True)
            # rh_package = 'runhouse_nightly-0.0.1.dev20221202-py3-none-any.whl'
            # rh_download_cmd = f'curl https://runhouse-package.s3.amazonaws.com/{rh_package} --output {rh_package}'
            if not _install_url:
                import runhouse

                _install_url = f"runhouse=={runhouse.__version__}"
            rh_install_cmd = f"pip install {_install_url}"

        install_cmd = (
            f"{env._activate_cmd} && {rh_install_cmd}" if env else rh_install_cmd
        )
        status_codes = self.run([install_cmd], stream_logs=True)

        if status_codes[0][0] != 0:
            raise ValueError(f"Error installing runhouse on cluster <{self.name}>")

    def install_packages(
        self, reqs: List[Union[Package, str]], env: Union["Env", str] = None
    ):
        """Install the given packages on the cluster.

        Args:
            reqs (List[Package or str): List of packages to install on cluster and env
            env (Env or str): Environment to install package on. If left empty, defaults to base environment.
                (Default: ``None``)

        Example:
            >>> cluster.install_packages(reqs=["accelerate", "diffusers"])
            >>> cluster.install_packages(reqs=["accelerate", "diffusers"], env="my_conda_env")
        """
        self.check_server()
        to_install = []
        for package in reqs:
            if isinstance(package, str):
                pkg_obj = Package.from_string(package, dryrun=False)
            else:
                pkg_obj = package

            if isinstance(pkg_obj, dict):
                pkg_obj = Package.from_config(pkg_obj)
                to_install.append(pkg_obj)
            elif isinstance(pkg_obj.install_target, Folder):
                if not pkg_obj.install_target.system == self:
                    pkg_str = pkg_obj.name or Path(pkg_obj.install_target.path).name
                    logging.info(
                        f"Copying local package {pkg_str} to cluster <{self.name}>"
                    )
                    pkg_obj = pkg_obj.to(self)
                to_install.append(pkg_obj)
            else:
                to_install.append(package)  # Just appending the string!
        logging.info(
            f"Installing packages on cluster {self.name}: "
            f"{[req if isinstance(req, str) else str(req) for req in reqs]}"
        )
        self.client.install(to_install, env)

    def get(self, key: str, default: Any = None, stream_logs: bool = True):
        """Get the result for a given key from the cluster's object store."""
        self.check_server()
        res = self.client.get_object(key, stream_logs=stream_logs)
        return res if res is not None else default

    def get_run(self, run_name: str, folder_path: str = None):
        self.check_server()
        return self.client.get_run_object(run_name, folder_path)

    def add_secrets(self, provider_secrets: dict):
        """Copy secrets from current environment onto the cluster"""
        self.check_server()
        return self.client.add_secrets(provider_secrets)

    def put(self, key: str, obj: Any):
        """Put the given object on the cluster's object store at the given key."""
        self.check_server()
        return self.client.put_object(key, obj)

    def list_keys(self):
        """List all keys in the cluster's object store."""
        self.check_server()
        res = self.client.list_keys()
        return res

    def cancel(self, key: str, force=False):
        """Cancel a given run on cluster by its key."""
        self.check_server()
        return self.client.cancel_runs(key, force=force)

    def cancel_all(self, force=False):
        """Cancel all runs on cluster."""
        self.check_server()
        return self.client.cancel_runs("all", force=force)

    def clear_pins(self, pins: Optional[List[str]] = None):
        """Remove the given pinned items from the cluster. If `pins` is set to ``None``, then
        all pinned objects will be cleared."""
        self.check_server()
        self.client.clear_pins(pins)
        logger.info(f'Clearing pins on cluster {pins or ""}')

    def on_this_cluster(self):
        """Whether this function is being called on the same cluster."""
        return _current_cluster("name") == self.rns_address

    # ----------------- RPC Methods ----------------- #

    def connect_server_client(self, tunnel=True, force_reconnect=False):
        # FYI based on: https://sshtunnel.readthedocs.io/en/latest/#example-1
        # FYI If we ever need to do this from scratch, we can use this example:
        # https://github.com/paramiko/paramiko/blob/main/demos/rforward.py#L74
        if not self.address:
            raise ValueError(f"No address set for cluster <{self.name}>. Is it up?")

        # TODO [DG] figure out how to ping to see if tunnel is already up
        if self._rpc_tunnel and force_reconnect:
            self._rpc_tunnel.close()

        # TODO Check if port is already open instead of refcounting?
        # status = subprocess.run(['nc', '-z', self.address, str(self.grpc_port)], capture_output=True)
        # if not self.check_port(self.address, UnaryClient.DEFAULT_PORT):

        tunnel_refcount = 0
        if self.address in open_cluster_tunnels:
            ssh_tunnel, connected_port, tunnel_refcount = open_cluster_tunnels[
                self.address
            ]
            ssh_tunnel.check_tunnels()
            if ssh_tunnel.tunnel_is_up[ssh_tunnel.local_bind_address]:
                self._rpc_tunnel = ssh_tunnel
        else:
            self._rpc_tunnel, connected_port = self.ssh_tunnel(
                HTTPClient.DEFAULT_PORT,
                remote_port=DEFAULT_SERVER_PORT,
                num_ports_to_try=5,
            )
        open_cluster_tunnels[self.address] = (
            self._rpc_tunnel,
            connected_port,
            tunnel_refcount + 1,
        )

        # Connecting to localhost because it's tunneled into the server at the specified port.
        self.client = HTTPClient(host="127.0.0.1", port=connected_port)

    def check_server(self, restart_server=True):
        if self.name == _current_cluster("name"):
            return

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
                self.connect_server_client()
                cluster_config = self.config_for_rns
                if "sky_state" in cluster_config.keys():
                    # a bunch of setup commands that mess up json dump
                    del cluster_config["sky_state"]
                logger.info(f"Checking server {self.name}")
                self.client.check_server(cluster_config=cluster_config)
                logger.info(f"Server {self.name} is up.")
            except (
                requests.exceptions.ConnectionError,
                sshtunnel.BaseSSHTunnelForwarderError,
            ):
                # It's possible that the cluster went down while we were trying to install packages.
                if not self.is_up():
                    logger.info(f"Server {self.name} is down.")
                    self.up_if_not()
                elif restart_server:
                    logger.info(
                        f"Server {self.name} is up, but the HTTP server may not be up."
                    )
                    self.restart_server(resync_rh=False)
                    logger.info(f"Checking server {self.name} again.")
                    self.client.check_server(cluster_config=cluster_config)
                else:
                    raise ValueError(f"Could not connect to cluster <{self.name}>")
        return

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
                    ssh_username=creds.get("ssh_user"),
                    ssh_pkey=creds.get("ssh_private_key"),
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

    def restart_server(
        self,
        _rh_install_url: str = None,
        resync_rh: bool = True,
        restart_ray: bool = False,
    ):
        """Restart the RPC server.

        Args:
            resync_rh (bool): Whether to resync runhouse. (Default: True)
            restart_ray (bool): Whether to restart Ray. (Default: False)

        Example:
            >>> rh.cluster("rh-cpu").restart_server()
        """
        logger.info(f"Restarting HTTP server on {self.name}.")

        # TODO how do we capture errors if this fails?
        if resync_rh:
            self._sync_runhouse_to_cluster(_install_url=_rh_install_url)
        logfile = f"cluster_server_{self.name}.log"
        http_server_cmd = "python -m runhouse.servers.http.http_server"
        kill_proc_cmd = f'pkill -f "{http_server_cmd}"'
        # 2>&1 redirects stderr to stdout
        screen_cmd = (
            f"screen -dm bash -c '{http_server_cmd} |& tee -a ~/.rh/{logfile} 2>&1'"
        )
        cmds = [kill_proc_cmd]
        if restart_ray:
            ray_start_cmd = "ray start --head --port 6379 --autoscaling-config=~/ray_bootstrap_config.yaml"
            # We need to use this instead of ray stop to make sure we don't stop the SkyPilot ray server,
            # which runs on other ports but is required to preserve autostop and correct cluster status.
            kill_ray_cmd = 'pkill -f ".*ray.*6379.*"'
            if self.ips and len(self.ips) > 1:
                raise NotImplementedError(
                    "Starting Ray on a cluster with multiple nodes is not yet supported."
                    "In the meantime, you can simply start the Ray cluster via the following instructions, "
                    "and pass *only* the head node ip to the cluster constructor: \n"
                    "https://docs.ray.io/en/latest/cluster/vms/user-guides/launching-clusters/on-premises.html#manually-set-up-a-ray-cluster"
                )
            cmds.append(kill_ray_cmd)
            cmds.append(ray_start_cmd)
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
        # As of 2023-15-May still seems we need this.
        time.sleep(5)
        return status_codes

    @contextlib.contextmanager
    def pause_autostop(self):
        """Context manager to temporarily pause autostop. Mainly for OnDemand clusters, for BYO cluster
        there is no autostop."""
        pass

    def _run_module(
        self,
        relative_path,
        module_name,
        fn_name,
        fn_type,
        resources,
        conda_env,
        env_vars,
        run_name,
        args,
        kwargs,
    ):
        self.check_server()
        return self.client.run_module(
            relative_path,
            module_name,
            fn_name,
            fn_type,
            resources,
            conda_env,
            env_vars,
            run_name,
            args,
            kwargs,
        )

    def is_connected(self):
        """Whether the RPC tunnel is up.

        Example:
            >>> connected = cluster.is_connected()
        """
        return self.client is not None

    def disconnect(self):
        """Disconnect the RPC tunnel.

        Example:
            >>> cluster.disconnect()
        """
        if self._rpc_tunnel:
            self._rpc_tunnel.stop()
        # if self.client:
        #     self.client.shutdown()

    def __getstate__(self):
        """Delete non-serializable elements (e.g. thread locks) before pickling."""
        state = self.__dict__.copy()
        state["client"] = None
        state["_rpc_tunnel"] = None
        return state

    # ----------------- SSH Methods ----------------- #

    def ssh_creds(self):
        """Retrieve SSH credentials."""
        return self._ssh_creds

    def _rsync(self, source: str, dest: str, up: bool, contents: bool = False):
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
        ssh_credentials = self.ssh_creds()
        runner = command_runner.SSHCommandRunner(self.address, **ssh_credentials)
        if up:
            runner.run(["mkdir", "-p", dest], stream_logs=False)
        else:
            Path(dest).expanduser().parent.mkdir(parents=True, exist_ok=True)
        runner.rsync(source, dest, up=up, stream_logs=False)

    def ssh(self):
        """SSH into the cluster

        Example:
            >>> rh.cluster("rh-cpu").ssh()
        """
        creds = self.ssh_creds()
        subprocess.run(
            f"ssh {creds['ssh_user']}:{self.address} -i {creds['ssh_private_key']}".split(
                " "
            )
        )

    def _ping(self, timeout=5):
        ssh_call = threading.Thread(target=lambda: self.run(['echo "hello"']))
        ssh_call.start()
        ssh_call.join(timeout=timeout)
        if ssh_call.is_alive():
            raise TimeoutError("SSH call timed out")
        return True

    def run(
        self,
        commands: List[str],
        env: Union["Env", str] = None,
        stream_logs: bool = True,
        port_forward: Optional[int] = None,
        require_outputs: bool = True,
        run_name: Optional[str] = None,
    ) -> list:
        """Run a list of shell commands on the cluster. If `run_name` is provided, the commands will be
        sent over to the cluster before being executed and a Run object will be created.

        Example:
            >>> cpu.run(["pip install numpy"])
            >>> cpu.run(["pip install numpy", env="my_conda_env"])
            >>> cpu.run(["python script.py"], run_name="my_exp")
        """
        # TODO [DG] suspect autostop while running?
        from runhouse import run

        cmd_prefix = ""
        if env:
            if isinstance(env, str):
                from runhouse.rns.envs import Env

                env = Env.from_name(env)
            cmd_prefix = env._run_cmd

        if not run_name:
            # If not creating a Run then just run the commands via SSH and return
            return self._run_commands_with_ssh(
                commands, cmd_prefix, stream_logs, port_forward, require_outputs
            )

        # Create and save the Run locally
        with run(name=run_name, cmds=commands, overwrite=True) as r:
            return_codes = self._run_commands_with_ssh(
                commands, cmd_prefix, stream_logs, port_forward, require_outputs
            )

        # Register the completed Run
        r._register_cmd_run_completion(return_codes)
        logger.info(f"Saved Run to path: {r.folder.path}")
        return return_codes

    def _run_commands_with_ssh(
        self,
        commands: list,
        cmd_prefix: str,
        stream_logs: bool,
        port_forward: int = None,
        require_outputs: bool = True,
    ):
        return_codes = []

        runner = command_runner.SSHCommandRunner(self.address, **self.ssh_creds())
        for command in commands:
            command = f"{cmd_prefix} {command}" if cmd_prefix else command
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
        env: Union["Env", str] = None,
        stream_logs: bool = True,
        port_forward: Optional[int] = None,
        run_name: Optional[str] = None,
    ):
        """Run a list of python commands on the cluster.

        Example:
            >>> cpu.run_python(['import numpy', 'print(numpy.__version__)'])([""])
        """
        cmd_prefix = "python3 -c"
        if env:
            if isinstance(env, str):
                from runhouse.rns.envs import Env

                env = Env.from_name(env)
            cmd_prefix = f"{env._run_cmd} {cmd_prefix}"
        command_str = "; ".join(commands)
        # If invoking a run as part of the python commands also return the Run object
        return_codes = self.run(
            [f'{cmd_prefix} "{command_str}"'],
            stream_logs=stream_logs,
            port_forward=port_forward,
            run_name=run_name,
        )
        return return_codes

    def sync_secrets(self, providers: Optional[List[str]] = None):
        """Send secrets for the given providers.

        Args:
            providers(List[str] or None): List of providers to send secrets for.
                If `None`, all providers configured in the environment will by sent.

        Example:
            >>> cpu.sync_secrets(providers=["aws", "lambda"])
        """
        from runhouse import Secrets

        Secrets.to(system=self, providers=providers)

    def ipython(self):
        # TODO tunnel into python interpreter in cluster
        pass

    def notebook(
        self,
        persist: bool = False,
        sync_package_on_close: Optional[str] = None,
        port_forward: int = 8888,
    ):
        """Tunnel into and launch notebook from the cluster.

        Example:
            >>> rh.cluster("test-cluster").notebook()
        """
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
                self._rsync(source=f"~/{pkg.name}", dest=pkg.local_path, up=False)
            if not persist:
                tunnel.stop()
                kill_jupyter_cmd = f"jupyter notebook stop {port_fwd}"
                self.run(commands=[kill_jupyter_cmd])

    def remove_conda_env(
        self,
        env: Union[str, "CondaEnv"],
    ):
        """Remove conda env from the cluster.

        Example:
            >>> rh.cluster("rh-cpu").remove_conda_env("my_conda_env")
        """
        env_name = env if isinstance(env, str) else env.env_name
        self.run([f"conda env remove -n {env_name}"])
