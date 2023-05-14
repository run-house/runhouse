import ast
import inspect
import json
import logging
from typing import Callable

import requests

from runhouse.rns.api_utils.utils import load_resp_content
from runhouse.rns.hardware.cluster import Cluster

logger = logging.getLogger(__name__)


class SlurmCluster(Cluster):
    RESOURCE_TYPE = "cluster"
    DEFAULT_API_VERSION = "v0.0.36"

    def __init__(
        self,
        name: str,
        partition: str = None,
        log_folder: str = None,
        ssh_creds: dict = None,
        ips: list = None,
        api_url: str = None,
        api_auth_user: str = None,
        api_jwt_token: str = None,
        api_version: str = None,
        **kwargs,  # We have this here to ignore extra arguments when calling from from_config
    ):
        """
        Args:
            name (str): Name for the Cluster, to re-use later on within Runhouse.
            partition (str, optional): Name of specific partition for the Slurm scheduler to use.
                *Optional when using SSH.*
            log_folder (str, optional): Name of folder to store logs for the jobs.

                *Required when using SSH.*
            ssh_creds (dict, optional): Required for submitting a job via SSH and for accessing the
                node(s) running the job.
                If using the REST API, SSH credentials are not required for submitting jobs.

                *Required when using SSH.*
            ips: (str, optional): List of IP addresses for the nodes on the cluster.

                *Required when using SSH.*
            api_url (str, optional): URL of the existing Cluster (ex: ``http://*.**.**.**:6820``).

                *Required when using the REST API.*
            api_auth_user (str, optional): Username to authenticate with when making requests to the cluster.

                *Required when using the REST API.*
            api_jwt_token (str, optional): the REST API (slurmrestd) authenticates
                via `JWT <https://slurm.schedmd.com/jwt.html>`_

                *Required when using the REST API.*
            api_version (str, optional): REST API version to use. This will depend on the
                version of Slurm installed on the cluster. (Default: ``v0.0.36``)

                *Required when using the REST API.*

        .. note::
            To build a slurm cluster, please use the factory method :func:`cluster`.
        """
        # SSH attributes
        self.partition = partition
        self.log_folder = log_folder
        self.ips = ips

        # REST API attributes
        self.api_url = api_url
        self.api_auth_user = api_auth_user
        self.api_jwt_token = api_jwt_token
        self.api_version = api_version or self.DEFAULT_API_VERSION

        super().__init__(name=name, dryrun=True, ips=self.ips, ssh_creds=ssh_creds)

    @staticmethod
    def from_config(config: dict, dryrun=True, **kwargs):
        # Note: Dryrun is ignored here, since we don't want to start a ray instance on the cluster
        config = {**config, **kwargs}
        return SlurmCluster(**config)

    @property
    def config_for_rns(self):
        config = super().config_for_rns
        if self.ssh_creds():
            # Add the SSH specific attributes if relevant
            config.update({"partition": self.partition, "log_folder": self.log_folder})

        if self.api_url:
            # Add the REST API specific attributes if relevant
            config.update(
                {
                    "api_url": self.api_url,
                    "api_auth_user": self.api_auth_user,
                    "api_jwt_token": self.api_jwt_token,
                    "api_version": self.api_version,
                }
            )

        return config

    def submit_job(
        self, fn: Callable = None, payload: dict = None, *args, **kwargs
    ) -> int:
        """Main API for submitting a job. This will either submit a job via SSH or via the REST API, depending
        on the arguments provided.

        Args:
            payload (dict, optional): Dictionary with all the parameters necessary for the slurmrestd API daemon.
            fn (Callable, optional): Function to submit to the cluster.
            *args: Optional args for the function.
            **kwargs: Optional kwargs for the function.

        Returns:
            Submitted Job ID
        """

        if fn is not None:
            return self._submit_via_ssh(fn, *args, **kwargs)
        elif payload is not None:
            return self._submit_via_rest(payload)
        else:
            raise ValueError(
                "Must provide either a payload or a function to submit a job."
            )

    def get_job_status(self, job_id: int) -> str:
        """Get the status of a job."""
        # TODO [JL]
        raise NotImplementedError

    def delete_job(self, job_id: int) -> str:
        """Delete a job."""
        # TODO [JL]
        raise NotImplementedError

    # ----------------- SSH -----------------
    def _submit_via_ssh(self, fn: Callable, *args, **kwargs) -> int:
        """Submit a job via SSH. Returns the job id."""
        # https://github.com/facebookincubator/submitit

        if any(v is None for v in [self.log_folder, self.ssh_creds(), self.ips]):
            raise ValueError(
                "When using SSH must provide values for: `log_folder`, `ssh_creds` and `ips`."
            )

        job_id = self._submitit(fn, *args, **kwargs)
        logger.info(
            f"Submitted job with id: {job_id} to Slurm via SSH. Logs saved on cluster to folder: `{self.log_folder}`"
        )

        return job_id

    def _submitit(self, fn, *args, **kwargs):
        """Send the submitit code to the cluster, to be run via SSH"""
        # TODO [JL] support running commands in addition to functions (submitit.helpers.CommandFunction)
        # https://github.com/facebookincubator/submitit/blob/main/docs/examples.md#working-with-commands

        executor_params = (
            f"executor.update_parameters(slurm_partition='{self.partition}')"
            if self.partition
            else "executor.update_parameters()"
        )

        func_name, func_params, func_return = self._inspect_fn(fn)
        ret = self.run_python(
            [
                "import submitit",
                f"executor = submitit.AutoExecutor(folder='{self.log_folder}')",
                executor_params,
                f"{func_name} = lambda {func_params}: {func_return}",
                f"job = executor.submit({func_name}, *{args}, **{kwargs})",
                "print(job.job_id)",
            ]
        )

        resp = ret[0][1]
        if ret[0][0] != 0:
            raise Exception(f"Failed to submit job: {resp}")

        job_id = int(resp.strip())
        return job_id

    @staticmethod
    def _inspect_fn(fn):
        """Grab the function's name, params, and return str repr. This is necessary since we are using submitit to
        run the function via SSH, and we need to define the function inside the SSH command."""
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

        if return_expr is None:
            func_return = None
        else:
            func_return = ast.unparse(return_expr)

        return func_name, func_params, func_return

    # ----------------- REST API (slurmrestd) -----------------
    def _submit_via_rest(self, payload: dict) -> int:
        """Submit a job to the Slurm Cluster via the REST API. Add the IP of the node where the job is
        submitted to the list of IPs. Returns the job id.

        API docs: https://slurm.schedmd.com/rest_api.html#slurmV0038SubmitJob

        Sample payload:

        .. code-block::

            {
                "job": {
                "name": "test",
                "ntasks":2,
                "nodes": 2,
                "current_working_directory": "/home/ubuntu/test",
                "standard_input": "/dev/null",
                "standard_output": "/home/ubuntu/test/test.out",
                "standard_error": "/home/ubuntu/test/test_error.out",
                "environment": {
                    "PATH": "/bin:/usr/bin/:/usr/local/bin/",
                    "LD_LIBRARY_PATH": "/lib/:/lib64/:/usr/local/lib"}
                },
                "script": "#!/bin/bash\necho 'I am from the REST API'"
            }
        """
        resp = requests.post(
            f"{self.api_url}/slurm/{self.api_version}/job/submit",
            headers=self._slurmrestd_headers(),
            data=json.dumps(payload),
        )
        if resp.status_code != 200:
            raise Exception(f"Error submitting job: {resp.text}")

        resp_data = load_resp_content(resp)
        job_id = resp_data.get("job_id")
        if not job_id:
            raise ValueError(f"Error running job: {resp_data.get('errors')}")

        logger.info(f"Submitted job with id: {job_id} to Slurm via REST")

        return job_id

    def _slurmrestd_headers(self):
        """Headers required for all requests to the REST API"""
        # https://slurm.schedmd.com/slurmrestd.html
        return {
            "X-SLURM-USER-NAME": self.api_auth_user,
            "X-SLURM-USER-TOKEN": self.api_jwt_token,
            "Content-Type": "application/json",
        }
