import logging
from typing import Callable, List, Union

from sky.utils import command_runner

from runhouse.rns.hardware.cluster import Cluster

logger = logging.getLogger(__name__)


class SlurmCluster(Cluster):
    RESOURCE_TYPE = "cluster"
    PARTITION_SERVER_JOB = "partition_server"

    def __init__(
        self,
        name: str,
        ip: str,
        ssh_creds: dict = None,
        partition: str = None,
        dryrun: bool = False,
        **kwargs,  # We have this here to ignore extra arguments when calling from from_config
    ):
        """
        .. note::
            To build a slurm cluster, please use the factory method :func:`slurm_cluster`.
        """
        self.partition = partition

        super().__init__(name=name, dryrun=dryrun, ips=[ip], ssh_creds=ssh_creds)

        if not dryrun:
            self._launch_partition_server()

    @staticmethod
    def from_config(config: dict, dryrun=True, **kwargs):
        cluster_config = {**config, **kwargs}
        return SlurmCluster(**cluster_config)

    @property
    def config_for_rns(self):
        """Metadata to store in RNS for the Slurm Cluster."""
        config = super().config_for_rns
        self.save_attrs_to_config(config, ["partition"])
        return config

    def jump_run(
        self,
        fn: Callable = None,
        commands: List[str] = None,
        env: Union["Env", str] = None,
        mail_type: str = None,
        mail_user: str = None,
        *args,
        **kwargs,
    ):
        """Submit a function or command(s) to run on the slurm cluster. Runhouse will send an RPC to the jump server
        and push the job into a queue on the cluster for execution.

        Args:
            fn (Callable, optional): A function to run on the cluster. If not provided, ``commands`` must be provided.
            commands (List[str], optional): A list of commands to run on the cluster.
                If not provided, ``fn`` must be provided.
            env (Union[Env, str], optional): Environment to install package on.
                If left empty, defaults to base environment.
            mail_type (str, optional): The type of email to send.
                Options include: ``NONE``, ``BEGIN``, ``END``, ``FAIL``, ``REQUEUE``, ``ALL``.
            mail_user (str, optional): The email address to send the email to. If not provided, no email will be sent.
            *args: Arguments to pass to the function.
            **kwargs: Keyword arguments to pass to the function.
        """
        from runhouse import function

        fn_obj = function(fn=fn, name=self.name, env=env, system=self) if fn else None

        resp = self.client.submit_job(
            name=self.name,
            fn_obj=fn_obj,
            partition=self.partition,
            commands=commands,
            env=env,
            mail_type=mail_type,
            mail_user=mail_user,
            args=args,
            kwargs=kwargs,
        )

        if resp.status_code != 200:
            raise Exception(f"Failed to submit job to {self.name}: {resp.json()}")

    def srun(
        self,
        commands: List[str],
        env: Union["Env", str] = None,
        stream_logs: bool = True,
    ):
        """Run a command (e.g. srun / sbatch) on the slurm cluster, without respecting queueing."""
        return super().run(commands, env, stream_logs)

    def sync_data_to_cluster(self, source: str, target: str) -> None:
        """Sync data from local machine to the jumpbox or login node.
        Note: We assume data stored on the jump server will be automatically synced with the requested resources'
        node where the job will be run.
        """
        runner = command_runner.SSHCommandRunner(ip=self.address, **self.ssh_creds())

        # Up: indicates that we are syncing from local to the cluster
        runner.rsync(source=source, target=target, up=True)

    def _launch_partition_server(self, job_name: str = None):
        job_name = job_name or self.PARTITION_SERVER_JOB
        if self._partition_server_is_running(job_name):
            return

        if self.partition:
            partition_server_cmd = (
                f"srun -J {job_name} -o %j.out -e %j.err "
                "python3 -m runhouse.servers.http.slurm.partition_server"
            )
        else:
            #  If not using slurm to launch the partition server (e.g. on a single node cluster), then
            #  run the partition server as a python process
            python_cmd = "python3 -m runhouse.servers.http.slurm.partition_server"
            partition_server_cmd = (
                f"screen -dm bash -c '{python_cmd} |& tee "
                f"-a ~/.rh/{self.PARTITION_SERVER_JOB}.log 2>&1'"
            )

        status_codes = self.run(
            commands=[partition_server_cmd],
            stream_logs=True,
        )

        if status_codes[0][0] != 0:
            raise Exception(f"Failed to launch partition server for {self.name}.")

    def _partition_server_is_running(self, job_name: str) -> bool:
        if self.partition:
            status_codes = self.run([f"squeue --name {job_name}"], stream_logs=True)
            if job_name in status_codes[0][1]:
                return True
        else:
            status_codes = self.run(
                ["ps aux | grep '[r]unhouse.servers.http.slurm.partition_server'"],
                stream_logs=True,
            )
            if "partition_server" in status_codes[0][1]:
                return True

        return False
