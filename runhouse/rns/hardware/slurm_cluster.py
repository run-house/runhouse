import logging
import re
from typing import List, Tuple, Union

import paramiko
from sky.utils import command_runner

from runhouse.rns.api_utils.utils import resolve_absolute_path
from runhouse.rns.hardware.cluster import Cluster

logger = logging.getLogger(__name__)


class SlurmConnectionManager:
    def __init__(
        self,
        jumpbox_client: paramiko.SSHClient,
        jumpbox_transport: paramiko.Transport,
        target_client: paramiko.SSHClient,
    ):
        self.jumpbox_client = jumpbox_client
        self.jumpbox_transport = jumpbox_transport
        self.target_client = target_client

    def close(self):
        self.target_client.close()
        self.jumpbox_transport.close()
        self.jumpbox_client.close()


class SlurmCluster(Cluster):
    RESOURCE_TYPE = "cluster"
    DEFAULT_PORT = 22

    def __init__(
        self,
        name: str,
        ip: str,
        ssh_creds: dict = None,
        port: int = None,
        **kwargs,  # We have this here to ignore extra arguments when calling from from_config
    ):
        """
        .. note::
            To build a slurm cluster, please use the factory method :func:`slurm_cluster`.
        """
        self.port = port or self.DEFAULT_PORT
        self.scm = None

        # Note: dryrun is ignored here, since we don't want to start a ray instance on the slurm cluster
        super().__init__(name=name, dryrun=True, ips=[ip], ssh_creds=ssh_creds)

    @staticmethod
    def from_config(config: dict, dryrun=True, **kwargs):
        cluster_config = {**config, **kwargs}
        return SlurmCluster(**cluster_config)

    def submit_job(self, job_name: str, commands: list) -> List[str]:
        """Run the submit command on the cluster. Return the IP(s) of the worker(s)
        running the job. If the submitted job name is not found, return None.

        Args:
            job_name (str): Name of the job. To be used later for retrieving the job id.
            commands (list): Commands to run on the cluser to submit the job (ex: srun / sbatch).

        Returns:
            List of IP addresses of the worker nodes running the job.
        """

        # TODO [JL] can we reliably get the worker nodes from these commands, or also run an squeue command below?
        ret = self.run(commands=commands)

        resp_code = ret[0][0]
        if resp_code != 0:
            raise Exception(f"Failed to submit job: {ret}")

        logger.info(f"Ran command on cluster with IP: {self.address}")

        # TODO [JL] can also filter by partition, user, etc. - make this more dynamic?
        ret = self.run([f"squeue --name={job_name} -o %i"])
        if ret[0][0] != 0:
            raise Exception(
                f"Failed to get job id for submitted job with name: {job_name}"
            )

        job_ids = self.format_cmd_output(ret)

        # Take the most recent job id if there are multiple jobs with the same name
        job_id = job_ids.split("\n")[-1]
        worker_ips = self._get_worker_node_ips(job_id)

        return worker_ips

    def ssh_tunnel_to_target_host(
        self,
        target_host: str,
        target_port: int = 22,
        target_username: str = "ubuntu",
        local_port: int = 9000,
    ) -> None:
        """
        Create an SSH tunnel to a target host (i.e. worker node) via the jumpbox server / login node.

        Args:
            target_host (str, optional): IP of the target host.
            target_port (int, optional): Port to connect to on the target host. Defaults to 22.
            target_username (str, optional): Username to connect to target host. Defaults to ``ubuntu``.
            local_port (int, optional): Local port to bind the tunnel. Defaults to 9000.
        """
        # Connect to the jumpbox server / login node
        ssh_creds = self.ssh_creds()
        key_filename = ssh_creds.get("ssh_private_key")
        if key_filename is None:
            raise ValueError("ssh_private_key must be specified in ssh_creds")
        key_filename = resolve_absolute_path(key_filename)

        jumpbox_client = paramiko.SSHClient()
        jumpbox_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        jumpbox_client.load_system_host_keys()
        jumpbox_client.connect(
            self.address,
            port=self.port,
            username=ssh_creds.get("ssh_user"),
            key_filename=key_filename,
        )

        # Create a transport channel through the jumpbox server
        jumpbox_transport = jumpbox_client.get_transport()

        # Open a new SSH session to the target server via the jumpbox
        target_channel = jumpbox_transport.open_channel(
            "direct-tcpip", (target_host, target_port), ("localhost", local_port)
        )

        # Connect to the target server through the tunnel
        target_client = paramiko.SSHClient()
        target_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        target_client.connect(
            "localhost",
            port=local_port,
            username=target_username,
            sock=target_channel,
            key_filename=key_filename,
        )

        self.scm = SlurmConnectionManager(
            jumpbox_client=jumpbox_client,
            jumpbox_transport=jumpbox_transport,
            target_client=target_client,
        )

        logger.info(f"Successfully created SSH tunnel to {target_host}:{target_port}")

    def run_commands_on_target_host(self, commands: list) -> Tuple[str, str]:
        """Execute commands on the target worker node via SSH. Return the stdout and stderr."""
        if self.scm is None:
            raise ValueError(
                "No SSH connection to worker host has been established. "
                "Please call `ssh_tunnel_to_target()` to create one."
            )
        stdin, stdout, stderr = self.scm.target_client.exec_command("\n".join(commands))

        self.scm.close()

        return self._decode_streams(stdout), self._decode_streams(stderr)

    def sync_data_to_cluster(self, source: str, target: str) -> None:
        """Sync data from local machine to the jumpbox or login node.
        Note: We assume data stored on the jump server will be automatically synced with the requested resources'
        node where the job will be run.
        """
        runner = command_runner.SSHCommandRunner(ip=self.address, **self.ssh_creds())

        # Up: indicates that we are syncing from local to the cluster
        runner.rsync(source=source, target=target, up=True)

    # TODO [JL] - add these back in
    def status(self, job_id: int) -> str:
        """Get the status of a job."""
        raise NotImplementedError

    def result(self, job_id: int) -> str:
        """Get the result of a job. Returns the result as a string."""
        raise NotImplementedError

    def stderr(self, job_id: int) -> str:
        """Get the stderr of a job."""
        raise NotImplementedError

    def stdout(self, job_id: int) -> str:
        """Get the stdout of a job."""
        raise NotImplementedError

    # -------------------------------------
    @staticmethod
    def _decode_streams(stream):
        return stream.read().decode("utf-8")

    @staticmethod
    def format_cmd_output(output):
        return output[0][1].strip("\n")

    def _get_worker_node_ips(self, job_id: Union[str, int]) -> list:
        """Get the worker node(s) where a specified job is running."""
        ret = self.run(commands=[f"scontrol show jobid -dd {job_id} | grep NodeList"])
        if ret[0][0] != 0:
            raise Exception(f"Failed to get info on job: {ret}")

        # Extract the value(s) of NodeList (i.e. the nodes where the job is running)
        node_data = ret[0][1]
        reg_exp = re.search(r"NodeList=([^ ]+)", str(node_data))
        node_list = reg_exp.group(1).strip()

        if node_list == "(null)":
            raise Exception(f"No node list found: {node_list}")

        worker_ips = []
        for ip_or_name in node_list:
            resp = self.run_python(
                ["import socket", f"print(socket.inet_aton('{ip_or_name}'))"]
            )
            if resp[0][0] == 0:
                worker_ips.append(self.format_cmd_output(resp))
                continue

            resp = self.run_python(
                ["import socket", f"print(socket.gethostbyname('{ip_or_name}'))"]
            )
            if resp[0][0] != 0:
                raise Exception(f"Failed to get IP address for {ip_or_name}: {resp}")
            worker_ips.append(self.format_cmd_output(resp))

        if not worker_ips:
            raise Exception(f"Failed to get IP address from node list")

        return worker_ips
