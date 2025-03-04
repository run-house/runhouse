import shlex
import subprocess
import warnings

from pathlib import Path
from typing import Dict, Optional

from runhouse.constants import DEFAULT_SERVER_PORT, LOCALHOST
from runhouse.resources.hardware.cluster import Cluster
from runhouse.resources.hardware.utils import ServerConnectionType

from runhouse.resources.images import Image

from runhouse.utils import run_with_logs

# Filter out DeprecationWarnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from runhouse.logger import get_logger

logger = get_logger(__name__)


class DockerCluster(Cluster):
    RESOURCE_TYPE = "cluster"

    def __init__(
        self,
        # Name will almost always be provided unless a "local" cluster is created
        name: Optional[str] = None,
        container_name: str = None,
        server_host: str = None,
        server_port: int = None,
        client_port: int = None,
        den_auth: bool = False,
        dryrun: bool = False,
        image: Optional["Image"] = None,
        home_dir: Optional[str] = None,
        **kwargs,  # We have this here to ignore extra arguments when calling from from_config
    ):
        """
        Docker Cluster.
        """
        super().__init__(
            name=name,
            ips=[container_name],
            server_host=server_host or "0.0.0.0",
            server_port=server_port or DEFAULT_SERVER_PORT,
            client_port=client_port,
            den_auth=den_auth,
            image=image,
            dryrun=dryrun,
        )
        self.server_connection_type = ServerConnectionType.DOCKER
        self._home_dir = home_dir

    @property
    def container_client(self):
        import docker

        client = docker.from_env()

        return client.containers.get(self.head_ip)

    @property
    def home_dir(self):
        if not self._home_dir:
            # self._home_dir = self.container_client.exec_run("pwd").output.decode().strip()
            self._home_dir = (
                self.container_client.exec_run("bash -c 'echo $HOME'")
                .output.decode()
                .strip()
            )

        return self._home_dir

    def config(self, condensed: bool = True):
        config = super().config(condensed)
        self.save_attrs_to_config(
            config,
            [
                "home_dir",
            ],
        )
        return config

    @property
    def server_address(self):
        """Address to use in the requests made to the cluster. If creating an SSH tunnel with the cluster,
        ths will be set to localhost, otherwise will use the cluster's domain (if provided), or its
        public IP address."""
        return LOCALHOST

    def ssh_tunnel(
        self, local_port, remote_port=None, num_ports_to_try: int = 0
    ) -> "SshTunnel":
        # We can't actually forward new ports after the container is started, so raise an error
        raise NotImplementedError()

    # ----------------- SSH Methods ----------------- #
    def rsync(
        self,
        source: str,
        dest: str,
        up: bool = True,
        node: str = None,
        src_node: str = None,
        contents: bool = False,
        filter_options: str = None,
        stream_logs: bool = False,
        ignore_existing: bool = False,
        parallel: bool = False,
    ):
        """
        Sync the contents of the source directory into the destination.

        Args:
            source (str): The source path.
            dest (str): The target path.
            up (bool): The direction of the sync. If ``True``, will rsync from local to cluster. If ``False``
              will rsync from cluster to local.
            node (str, optional): Specific cluster node to rsync to. If not specified will use the
                address of the cluster's head node.
            src_node (str, optional): Specific cluster node to rsync from, for node-to-node rsyncs.
            contents (bool, optional): Whether the contents of the source directory or the directory
                itself should be copied to destination. If ``True`` the contents of the source directory are
                copied to the destination, and the source directory itself is not created at the destination.
                If ``False`` the source directory along with its contents are copied ot the destination, creating
                an additional directory layer at the destination. (Default: ``False``).
            filter_options (str, optional): The filter options for rsync.
            stream_logs (bool, optional): Whether to stream logs to the stdout/stderr. (Default: ``False``).
            ignore_existing (bool, optional): Whether the rsync should skip updating files that already exist
                on the destination. (Default: ``False``).

        .. note::
            Ending ``source`` with a slash will copy the contents of the directory into dest,
            while omitting it will copy the directory itself (adding a directory layer).
        """
        # FYI, could be useful: https://github.com/gchamon/sysrsync
        if node not in [None, 0, self.ips[0], "all"]:
            raise ValueError("DockerCluster only supports syncing to the head node.")
        if src_node or parallel:
            raise ValueError(
                "DockerCluster does not support node-to-node rsyncs or parallel rsyncs."
            )

        if contents:
            source = source + "/" if not source.endswith("/") else source
            dest = dest + "/" if not dest.endswith("/") else dest

        if "~" in dest:
            dest = dest.replace("~", self.home_dir)

        if up:
            logger.info(f"Rsyncing {source} to {dest}")
            self.run_bash_over_ssh([f"mkdir -p {dest}"], stream_logs=stream_logs)
        else:
            logger.info(f"Rsyncing {source} to {dest}")
            Path(dest).expanduser().parent.mkdir(parents=True, exist_ok=True)

        rsync_cmd = shlex.split(
            "rsync -avz --filter='dir-merge,- .gitignore' "
            f"-e 'docker exec -i ' {source} {self.head_ip}:{dest}"
        )
        logger.info(f"Running rsync command: {' '.join(rsync_cmd)}")
        if ignore_existing:
            rsync_cmd += ["--ignore-existing"]
        subprocess.run(rsync_cmd, check=True, capture_output=not stream_logs, text=True)

    def _ping(self, timeout=5, retry=False):
        if not self.ips:
            return False

        import docker

        client = docker.from_env()
        try:
            container = client.containers.get(self.ips[0])
        except docker.errors.NotFound:
            return False

        if container.status != "running":
            return False

        return True

    def up(self):
        """Start the cluster"""
        # import docker
        #
        # client = docker.from_env()
        #
        # container = client.containers.run("runhouse",
        #                                   detach=True,
        #                                   ports={"32300": 32300},
        #                                   name="runhouse-test-container"
        #                                   )
        # container.start()
        raise NotImplementedError(
            "Docker clusters are started when the container is created."
        )

    def ssh(self):
        """SSH into the cluster

        Example:
            >>> rh.cluster("rh-cpu").ssh()
        """
        shell_cmd = f"docker exec -it {self.head_ip} /bin/bash"
        subprocess.run(shell_cmd, shell=True)

    def _run_commands_with_runner(
        self,
        commands: list,
        env_vars: Dict = {},
        stream_logs: bool = True,
        node: str = None,
        require_outputs: bool = True,
        _ssh_mode: str = "interactive",  # Note, this only applies for non-password SSH
    ):
        if isinstance(commands, str):
            commands = [commands]

        for i, command in enumerate(commands):
            if "~" in command:
                commands[i] = command.replace("~", self.home_dir)

        return_codes = []

        env_var_prefix = (
            " ".join(f"{key}={val}" for key, val in env_vars.items())
            if env_vars
            else ""
        )

        for command in commands:
            logger.info(f"Running command on {self.name}: {command}")

            # set env vars after log statement
            command = f"{env_var_prefix} {command}" if env_var_prefix else command

            docker_exec_cmd = (
                f"docker exec -i {self.head_ip} bash -c {shlex.quote(command)}"
            )
            ret_code = run_with_logs(
                docker_exec_cmd,
                stream_logs=stream_logs,
                require_outputs=require_outputs,
            )
            return_codes.append(ret_code)

        return return_codes
