import contextlib
import logging
import os
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

from runhouse.rh_config import obj_store, open_cluster_tunnels, rns_client
from runhouse.rns.packages.package import Package
from runhouse.rns.resource import Resource
from runhouse.rns.utils.env import _get_env_from
from runhouse.rns.utils.hardware import _current_cluster

from runhouse.servers.http import DEFAULT_SERVER_PORT, HTTPClient

logger = logging.getLogger(__name__)


class Cluster(Resource):
    RESOURCE_TYPE = "cluster"
    REQUEST_TIMEOUT = 5  # seconds

    SERVER_LOGFILE = "~/.rh/server.log"
    CLI_RESTART_CMD = "runhouse restart"
    SERVER_START_CMD = "python3 -m runhouse.servers.http.http_server"
    SERVER_STOP_CMD = f'pkill -f "{SERVER_START_CMD}"'
    # 2>&1 redirects stderr to stdout
    START_SCREEN_CMD = (
            f"screen -dm bash -c \"{SERVER_START_CMD} |& tee -a '{SERVER_LOGFILE}' 2>&1\""
        )
    RAY_START_CMD = "ray start --head --port 6379"
    # RAY_BOOTSTRAP_FILE = "~/ray_bootstrap_config.yaml"
    # --autoscaling-config=~/ray_bootstrap_config.yaml
    # We need to use this instead of ray stop to make sure we don't stop the SkyPilot ray server,
    # which runs on other ports but is required to preserve autostop and correct cluster status.
    RAY_KILL_CMD = 'pkill -f ".*ray.*6379.*"'

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
            # OnDemandCluster will start ray itself, but will also set address later, so won't reach here.
            self.check_server()

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
        elif resource_subtype == "SageMakerCluster":
            from runhouse.rns.hardware import SageMakerCluster

            return SageMakerCluster(**config, dryrun=dryrun)
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

        install_cmd = f"{env._run_cmd} {rh_install_cmd}" if env else rh_install_cmd

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
        from runhouse.rns.envs.env import Env

        self.check_server()
        env = _get_env_from(env) or Env(name=env)
        env.reqs = env._reqs + reqs
        env.to(self)

    def get(
        self, key: str, default: Any = None, remote=False, stream_logs: bool = False
    ):
        """Get the result for a given key from the cluster's object store."""
        self.check_server()
        if self.on_this_cluster():
            return obj_store.get(key, default=default)
        res = self.client.call_module_method(
            key, None, remote=remote, stream_logs=stream_logs
        )
        return res if res is not None else default

    # TODO deprecate
    def get_run(self, run_name: str, folder_path: str = None):
        self.check_server()
        return self.get(run_name, remote=True).provenance

    # TODO this doesn't need to be a dedicated rpc, it can just flow through Secrets.to and put_resource,
    #  like installing packages. Also, it should accept an env (for env var secrets and docker envs).
    def add_secrets(self, provider_secrets: dict):
        """Copy secrets from current environment onto the cluster"""
        self.check_server()
        return self.client.add_secrets(provider_secrets)

    def put(self, key: str, obj: Any, env=None):
        """Put the given object on the cluster's object store at the given key."""
        self.check_server()
        if self.on_this_cluster():
            return obj_store.put(key, obj, env=env)
        return self.client.put_object(key, obj, env=env)

    def put_resource(self, resource: Resource, state=None, dryrun=False):
        """Put the given resource on the cluster's object store. Returns the key (important if name is not set)."""
        self.check_server()
        env = (
            resource.env
            if hasattr(resource, "env")
            else resource.name
            if resource.RESOURCE_TYPE == "env"
            else None
        )
        if self.on_this_cluster():
            return obj_store.put(key=resource.name, value=resource, env=env)
        return self.client.put_resource(
            resource, state=state or {}, env=env, dryrun=dryrun
        )

    def rename(self, old_key: str, new_key: str):
        """Rename a key in the cluster's object store."""
        self.check_server()
        if self.on_this_cluster():
            return obj_store.rename(old_key, new_key)
        return self.client.rename_object(old_key, new_key)

    def list_keys(self, env=None):
        """List all keys in the cluster's object store."""
        self.check_server()
        if self.on_this_cluster():
            return obj_store.keys()
        res = self.client.list_keys(env=env)
        return res

    def cancel(self, key: str, force=False):
        """Cancel a given run on cluster by its key."""
        self.check_server()
        if self.on_this_cluster():
            return obj_store.cancel(key, force=force)
        return self.client.cancel(key, force=force)

    def cancel_all(self, force=False):
        """Cancel all runs on cluster."""
        self.check_server()
        if self.on_this_cluster():
            return obj_store.cancel_all(force=force)
        return self.client.cancel("all", force=force)

    def delete_keys(self, keys: Union[None, str, List[str]] = None):
        """Delete the given keys from the cluster's object store."""
        self.check_server()
        if isinstance(keys, str):
            keys = [keys]
        return self.client.delete_keys(keys)

    def on_this_cluster(self):
        """Whether this function is being called on the same cluster."""
        return _current_cluster("name") == self.rns_address

    # ----------------- RPC Methods ----------------- #

    def connect_server_client(self, tunnel=True, force_reconnect=False):
        if not self.address:
            raise ValueError(f"No address set for cluster <{self.name}>. Is it up?")

        if self._rpc_tunnel and force_reconnect:
            self._rpc_tunnel.close()

        tunnel_refcount = 0
        ssh_tunnel = None
        if self.address in open_cluster_tunnels:
            ssh_tunnel, connected_port, tunnel_refcount = open_cluster_tunnels[
                self.address
            ]
            if isinstance(ssh_tunnel, SSHTunnelForwarder):
                ssh_tunnel.check_tunnels()
                if ssh_tunnel.tunnel_is_up[ssh_tunnel.local_bind_address]:
                    self._rpc_tunnel = ssh_tunnel

        if "localhost" in self.address or ":" in self.address:
            if ":" in self.address:
                # Case 1: "localhost:23324" or <real_ip>:<custom port> (e.g. a port is already open to the server)
                self._rpc_tunnel, connected_port = self.address.split(":")
            else:
                # Case 2: "localhost"
                self._rpc_tunnel, connected_port = self.address, HTTPClient.DEFAULT_PORT
        elif not ssh_tunnel:
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
        creds = self.ssh_creds()
        if creds.get("password") and creds.get("ssh_user"):
            self.client = HTTPClient(
                host="127.0.0.1",
                port=connected_port,
                auth=(creds.get("ssh_user"), creds.get("password")),
            )
        else:
            self.client = HTTPClient(host="127.0.0.1", port=connected_port)

    def check_server(self, restart_server=True):
        if self.on_this_cluster():
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
                    self.restart_server()
                    for i in range(5):
                        logger.info(f"Checking server {self.name} again [{i + 1}/5].")
                        try:
                            self.client.check_server(cluster_config=cluster_config)
                            logger.info(f"Server {self.name} is up.")
                            break
                        except (
                            requests.exceptions.ConnectionError,
                            requests.exceptions.ReadTimeout,
                        ) as error:
                            if i == 5:
                                print(error)
                            time.sleep(5)
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
                    ssh_password=creds.get("password"),
                    local_bind_address=("", local_port),
                    remote_bind_address=(
                        "127.0.0.1",
                        remote_port or local_port,
                    ),
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

    def restart_server(
        self,
        _rh_install_url: str = None,
        resync_rh: bool = True,
        restart_ray: bool = True,
    ):
        """Restart the RPC server.

        Args:
            resync_rh (bool): Whether to resync runhouse. (Default: ``True``)
            restart_ray (bool): Whether to restart Ray. (Default: ``True``)

        Example:
            >>> rh.cluster("rh-cpu").restart_server()
        """
        logger.info(f"Restarting HTTP server on {self.name}.")

        if resync_rh:
            self._sync_runhouse_to_cluster(_install_url=_rh_install_url)

        cmd = self.CLI_RESTART_CMD + (" --no-restart-ray" if not restart_ray else "")
        status_codes = self.run(commands=[cmd])
        if not status_codes[0][0] == 0:
            raise ValueError(f"Failed to restart server {self.name}.")
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

    def call_module_method(
        self,
        module_name,
        method_name,
        *args,
        stream_logs=True,
        run_name=None,
        remote=False,
        run_async=False,
        **kwargs,
    ):
        """Call a method on a module that is installed on the cluster.

        Args:
            module_name (str): Name of the module saved on system.
            method_name (str): Name of the method.
            stream_logs (bool): Whether to stream logs from the method call.
            run_name (str): Name for the run.
            remote (bool): Return a remote object from the function, rather than the result proper.
            run_async (bool): Run the method asynchronously and retun a run_key to retreive results and logs later.
            *args: Positional arguments to pass to the method.
            **kwargs: Keyword arguments to pass to the method.

        Example:
            >>> cluster.call_module_method("my_module", "my_method", arg1, arg2, kwarg1=kwarg1)
        """
        self.check_server()
        # Note: might be single value, might be a generator!
        if self.on_this_cluster():
            # TODO
            pass
        return self.client.call_module_method(
            module_name,
            method_name,
            stream_logs=stream_logs,
            run_name=run_name,
            remote=remote,
            run_async=run_async,
            args=args,
            kwargs=kwargs,
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

    def _fsspec_sync(self, source: str, dest: str, up: bool):
        from runhouse.rns.folders import folder

        logger.info(f"syncing files from {source} to {dest} using fsspec")

        f = folder(system=self, path="", dryrun=True)
        fs = f.fsspec_fs

        if up:  # local to cluster
            if (Path(source) / ".gitignore").exists():
                tracked_files = (
                    subprocess.check_output(
                        "git ls-files --cached --exclude-standard",
                        cwd=source,
                        shell=True,
                    )
                    .decode("utf-8")
                    .split()
                )
                # exclude docs/ when syncing over runhouse
                if Path(source).name == "runhouse":
                    tracked_files = [
                        file for file in tracked_files if "docs/" not in file
                    ]
                untracked_files = (
                    subprocess.check_output(
                        "git ls-files --other --exclude-standard",
                        cwd=source,
                        shell=True,
                    )
                    .decode("utf-8")
                    .split()
                )
                files = [
                    Path(source) / file for file in tracked_files + untracked_files
                ]
                fs.put(files, dest, recursive=True, create_dir=True)
            else:
                fs.put(source, dest, recursive=True, create_dir=True)
        else:  # cluster to local
            if fs.exists(str(Path(source) / ".gitignore")):
                files = self.run(
                    [f"cd {source} && git ls-files --cached --exclude-standard"]
                )[0][1]

                files = files.split()
                fs.get(files, dest, recursive=True, create_dir=True)
            else:
                # the following errors if {source} is a directory so currently working around by extracting files
                # fs.get(source, dest, recursive=True, create_dir=True)
                files = fs.find(source)
                fs.get(files, dest, recursive=True, create_dir=True)

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
        if not ssh_credentials.get("password"):
            # Use SkyPilot command runner
            if not ssh_credentials.get("ssh_private_key"):
                ssh_credentials["ssh_private_key"] = None
            runner = command_runner.SSHCommandRunner(self.address, **ssh_credentials)
            if up:
                runner.run(["mkdir", "-p", dest], stream_logs=False)
            else:
                Path(dest).expanduser().parent.mkdir(parents=True, exist_ok=True)
            runner.rsync(source, dest, up=up, stream_logs=False)
        else:
            if dest.startswith("~/"):
                dest = dest[2:]

            self._fsspec_sync(source, dest, up)

    def ssh(self):
        """SSH into the cluster

        Example:
            >>> rh.cluster("rh-cpu").ssh()
        """
        creds = self.ssh_creds()

        if creds.get("ssh_private_key"):
            cmd = (
                f"ssh {creds['ssh_user']}:{self.address} -i {creds['ssh_private_key']}"
            )
        else:
            # if needs a password, will prompt for manual password
            cmd = f"ssh {creds['ssh_user']}@{self.address}"

        subprocess.run(cmd.split(" "))

    def _ping(self, timeout=5):
        ssh_call = threading.Thread(target=lambda: self.run(['echo "hello"']))
        ssh_call.start()
        ssh_call.join(timeout=timeout)
        if ssh_call.is_alive():
            raise TimeoutError("SSH call timed out")
        return True

    def _logfile_path(self, logfile):
        return f"~/.rh/{logfile}"

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
        # TODO [DG] suspend autostop while running
        from runhouse.rns.run import run

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

        ssh_credentials = self.ssh_creds()

        if not ssh_credentials.get("password"):
            runner = command_runner.SSHCommandRunner(self.address, **ssh_credentials)
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
        else:
            import paramiko

            log_path = os.devnull
            log_dir = os.path.expanduser(os.path.dirname(log_path))
            os.makedirs(log_dir, exist_ok=True)

            with paramiko.SSHClient() as ssh:
                ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                ssh.connect(
                    self.address,
                    username=ssh_credentials.get("ssh_user"),
                    # key_filename=ssh_credentials.get("ssh_private_key", None),
                    password=ssh_credentials.get("password"),
                )

                # bash warnings to remove from stderr
                skip_err = [
                    "bash: cannot set terminal process group",
                    "bash: no job control in this shell",
                ]

                for command in commands:
                    logger.info(f"Running command on {self.name}: {command}")

                    if stream_logs:
                        command += f"| tee {log_path} "
                        command += "; exit ${PIPESTATUS[0]}"
                    else:
                        command += f"> {log_path}"

                    # adapted from skypilot's ssh command runner
                    command = (
                        "bash --login -c -i $'true && source ~/.bashrc"
                        "&& export OMP_NUM_THREADS=1 PYTHONWARNINGS=ignore"
                        f" && ({cmd_prefix} {command})'"
                    )

                    transport = ssh.get_transport()
                    channel = transport.open_session()
                    channel.exec_command(command)

                    stdout = channel.recv(-1).decode()
                    exit_code = channel.recv_exit_status()
                    stderr = channel.recv_stderr(-1).decode()

                    stderr = stderr.split("\n")
                    stderr = [
                        err
                        for err in stderr
                        if not any(skip in err for skip in skip_err)
                    ]
                    stderr = "\n".join(stderr)

                    channel.close()

                    if require_outputs:
                        return_codes.append((exit_code, stdout, stderr))
                    else:
                        return_codes.append(exit_code)

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
            >>> cpu.run_python(['import numpy', 'print(numpy.__version__)'])
            >>> cpu.run_python(["print('hello')"])

        Note:
            Running Python commands with nested quotes can be finicky. If using nested quotes,
            try to wrap the outer quote with double quotes (") and the inner quotes with a single quote (').
        """
        cmd_prefix = "python3 -c"
        if env:
            if isinstance(env, str):
                from runhouse.rns.envs import Env

                env = Env.from_name(env)
            cmd_prefix = f"{env._run_cmd} {cmd_prefix}"
        command_str = "; ".join(commands)
        command_str_repr = (
            repr(repr(command_str))[2:-2]
            if self.ssh_creds().get("password")
            else command_str
        )
        formatted_command = f'{cmd_prefix} "{command_str_repr}"'

        # If invoking a run as part of the python commands also return the Run object
        return_codes = self.run(
            [formatted_command],
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
            >>> rh.ondemand_cluster("rh-cpu").remove_conda_env("my_conda_env")
        """
        env_name = env if isinstance(env, str) else env.env_name
        self.run([f"conda env remove -n {env_name}"])
