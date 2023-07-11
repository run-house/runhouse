import ast
import inspect
import logging
from typing import Callable

from runhouse.rh_config import obj_store
from runhouse.rns.hardware.cluster import Cluster

logger = logging.getLogger(__name__)


class SlurmCluster(Cluster):
    RESOURCE_TYPE = "cluster"
    DEFAULT_TIMEOUT_MIN = 1

    def __init__(
        self,
        name: str,
        partition: str = None,
        log_folder: str = None,
        ssh_creds: dict = None,
        ips: list = None,
        cluster_params: dict = None,
        proxy_command: str = None,
        **kwargs,  # We have this here to ignore extra arguments when calling from from_config
    ):
        """
        .. note::
            To build a slurm cluster, please use the factory method :func:`slurm_cluster`.
        """
        self.partition = partition
        self.ips = ips
        self.cluster_params = cluster_params or {}
        self.proxy_command = proxy_command

        # %j is replaced by the job id at runtime
        self.log_folder = log_folder or f"{obj_store.LOGS_DIR}/%j"

        # Note: dryrun is ignored here, since we don't want to start a ray instance on the cluster
        super().__init__(name=name, dryrun=True, ips=self.ips, ssh_creds=ssh_creds)

    @staticmethod
    def from_config(config: dict, dryrun=True, **kwargs):
        cluster_config = {**config, **kwargs}
        return SlurmCluster(**cluster_config)

    @property
    def config_for_rns(self):
        config = super().config_for_rns
        config.update(
            {
                "partition": self.partition,
                "log_folder": self.log_folder,
                "proxy_command": self.proxy_command,
                **self.cluster_params,
            }
        )
        return config

    def submit_job(self, fn: Callable, *args, **kwargs) -> int:
        """Main API for submitting a job. This will submit a job via SSH

        Args:
            fn (Callable): Function to submit to the cluster.
            *args: Optional args for the function.
            **kwargs: Optional kwargs for the function.

        Returns:
            Submitted Job ID
        """
        return self._submit_via_ssh(fn, *args, **kwargs)

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

    def _submit_via_ssh(self, fn: Callable, *args, **kwargs):
        """Submit a job via SSH. Returns the job ID."""
        # https://github.com/facebookincubator/submitit

        if not self.ssh_creds() and not self.ips:
            raise ValueError(
                "When using SSH must provide values for: `ssh_creds` and `ips`."
            )

        job_id = self._submitit(fn, *args, **kwargs)
        logger.info(
            f"Submitted job (id={job_id}) to Slurm. Logs saved on cluster to folder: {obj_store.LOGS_DIR}/{job_id}"
        )

        return job_id

    def _submitit(self, fn, *args, **kwargs):
        """Send the submitit code to the cluster, to be run via SSH. Return the job ID."""
        # TODO [JL] support running commands in addition to functions (submitit.helpers.CommandFunction)
        # https://github.com/facebookincubator/submitit/blob/main/docs/examples.md#working-with-commands

        job_params = {**self.cluster_params, **{"slurm_partition": self.partition}}
        executor_params_cmd = (
            f"executor.update_parameters(**{job_params})"
            if self.partition
            else "executor.update_parameters()"
        )

        func_name, func_params, func_return = self._inspect_fn(fn)
        ret = self.run_python(
            [
                "import submitit",
                f"executor = submitit.AutoExecutor(folder='{self.log_folder}')",
                executor_params_cmd,
                f"{func_name} = lambda {func_params}: {func_return}",
                f"job = executor.submit({func_name}, *{args}, **{kwargs})",
                "print(job.job_id)",
            ],
            ssh_proxy_command=self.proxy_command,
        )

        resp = ret[0][1]
        if ret[0][0] != 0:
            raise Exception(f"Failed to submit job: {ret[0]}")

        job_id = int(resp.strip())
        return job_id

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
