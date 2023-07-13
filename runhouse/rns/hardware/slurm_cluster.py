import ast
import inspect
import logging
import pickle
from typing import Callable

from sky.utils import command_runner

from runhouse.rh_config import obj_store
from runhouse.rns.hardware.cluster import Cluster

logger = logging.getLogger(__name__)


class SlurmCluster(Cluster):
    RESOURCE_TYPE = "cluster"
    DEFAULT_TIMEOUT_MIN = 1

    def __init__(
        self,
        name: str,
        ip: str,
        partition: str = None,
        log_folder: str = None,
        ssh_creds: dict = None,
        job_params: dict = None,
        **kwargs,  # We have this here to ignore extra arguments when calling from from_config
    ):
        """
        .. note::
            To build a slurm cluster, please use the factory method :func:`slurm_cluster`.
        """
        self.partition = partition
        self.job_params = job_params or {}

        # %j is replaced by the Job ID at runtime
        self.log_folder = log_folder or f"{obj_store.LOGS_DIR}/%j"

        # Note: dryrun is ignored here, since we don't want to start a ray instance on the cluster
        super().__init__(name=name, dryrun=True, ips=[ip], ssh_creds=ssh_creds)

    @property
    def config_for_rns(self):
        config = super().config_for_rns
        config.update(
            {
                "partition": self.partition,
                "log_folder": self.log_folder,
                **self.job_params,
            }
        )
        return config

    @staticmethod
    def from_config(config: dict, dryrun=True, **kwargs):
        cluster_config = {**config, **kwargs}
        return SlurmCluster(**cluster_config)

    def submit_job(self, fn: Callable, *args, **kwargs) -> "submitit.Job":
        """Main API for submitting a job. This will submit a job via SSH.

        Args:
            fn (Callable): Function to submit to the cluster.
            *args: Optional args for the function.
            **kwargs: Optional kwargs for the function.

        Returns:
            Submitted Job object.
        """
        return self._submit_via_ssh(fn, *args, **kwargs)

    def create_ssh_tunnel(
        self,
        local_port: int,
        remote_port: int = None,
        node_ip: str = None,
        ssh_proxy_command: str = None,
    ):
        """Create an SSH tunnel to a node on the cluster.
        If no node IP is specified, will create a tunnel to the ip specified to the SlurmCluster object (presumably the
        jump server or login node).

        Returns the tunnel and the local port.

        Args:
            local_port (int): Local port to use for the tunnel.
            remote_port (int, optional): Remote port to use for the tunnel.
            node_ip (str, optional): IP of the node to create the tunnel.
                If not provided, will create a tunnel to the jump server / login node.
            ssh_proxy_command (str, optional): Proxy command to use for SSH connections to the cluster.
                Useful for communicating with clusters without public IPs (ex: sending
                jobs to a jump server responsible for submitting jobs to the node associated with the requested compute)
        """
        if node_ip:
            # TODO [JL] use proxy to create tunnel with the node IP
            logger.info(f"Creating SSH tunnel cluster node with IP: {node_ip}")
        else:
            logger.info(
                f"Creating SSH tunnel to jump server or login node with IP: {self.address}"
            )
            ssh_tunnel, local_port = self.ssh_tunnel(
                local_port=local_port, remote_port=remote_port
            )
        if ssh_tunnel is None:
            raise ValueError("Failed to create SSH tunnel")

        logger.info(f"Created SSH tunnel with IP: {self.address}")
        return ssh_tunnel, local_port

    def sync_data_to_cluster(self, source: str, target: str):
        """Sync data from local machine to the cluster.
        Note: We assume data stored on the jump server will be automatically synced with the requested resources'
        node where the job will be run.
        """
        runner = command_runner.SSHCommandRunner(ip=self.address, **self.ssh_creds())

        # Up: indicates that we are syncing from local to the cluster
        runner.rsync(source=source, target=target, up=True)

    def status(self, job_id: int) -> str:
        """Get the status of a job."""
        return self._get_job_output(job_id, output_type="job_state")

    def result(self, job_id: int) -> str:
        """Get the result of a job. Returns the result as a string."""
        return self._get_job_output(job_id, output_type="result")

    def stderr(self, job_id: int) -> str:
        """Get the stderr of a job."""
        return self._get_job_output(job_id, output_type="stderr")

    def stdout(self, job_id: int) -> str:
        """Get the stdout of a job."""
        return self._get_job_output(job_id, output_type="stdout")

    # -------------------------------------

    def _get_job_output(self, job_id: int, output_type: str):
        try:
            job_cmd = [
                "import submitit",
                f"job = submitit.Job(folder='{self.log_folder}', job_id='{job_id}')",
            ]

            ret = self.run_python(job_cmd + [f"print(job.{output_type}())"])

            resp = ret[0][1]
            if ret[0][0] != 0:
                raise Exception(f"Failed to get job {output_type}: {resp}")

            return resp.strip()

        except Exception as e:
            raise e

    def _submit_via_ssh(self, fn, *args, **kwargs):
        """Send the submitit code to the cluster, to be run via SSH. Return the job ID."""
        # TODO [JL] support running commands in addition to functions (submitit.helpers.CommandFunction)
        # https://github.com/facebookincubator/submitit/blob/main/docs/examples.md#working-with-commands

        if not self.ssh_creds() and not self.ips:
            raise ValueError(
                "When using SSH must provide values for: `ssh_creds` and `ips`."
            )

        job_params = {**self.job_params, **{"slurm_partition": self.partition}}
        executor_params_cmd = (
            f"executor.update_parameters(**{job_params})"
            if self.partition
            else "executor.update_parameters()"
        )

        func_name, func_params, func_return = self._inspect_fn(fn)
        ret = self.run_python(
            [
                "import submitit",
                "import pickle",
                f"executor = submitit.AutoExecutor(folder='{self.log_folder}')",
                executor_params_cmd,
                f"{func_name} = lambda {func_params}: {func_return}",
                f"job = executor.submit({func_name}, *{args}, **{kwargs})",
                "print(pickle.dumps(job))",
            ]
        )

        resp = ret[0][1]
        if ret[0][0] != 0:
            raise Exception(f"Failed to submit job: {ret[0]}")

        job = pickle.loads(ast.literal_eval(resp))

        logger.info(
            f"Submitted job (id={job.job_id}) to Slurm. Logs saved on server {self.address} "
            f"to folder: {obj_store.LOGS_DIR}/{job.job_id}"
        )

        return job

    @staticmethod
    def _inspect_fn(fn: Callable):
        """Grab the function's name, params, and return str repr. This is necessary since we are using submitit to
        run the function via SSH, and we need to define the function to run as part of the job that we submit inside
        the SSH command."""
        # Get function name
        func_name = fn.__name__

        # Get function parameters
        sig = inspect.signature(fn)
        func_params = ", ".join([str(p) for p in sig.parameters.values()])

        # Get function source code
        source_lines, _ = inspect.getsourcelines(fn)
        source = "".join(source_lines).strip()

        # Parse source code to extract return value
        tree = ast.parse(source)
        return_expr = None
        for node in ast.walk(tree):
            if isinstance(node, ast.Return):
                return_expr = node.value
                break

        func_return = ast.unparse(return_expr) if return_expr else None

        return func_name, func_params, func_return
