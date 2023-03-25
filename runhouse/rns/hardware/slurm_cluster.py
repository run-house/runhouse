import json
import logging

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
        url: str,
        auth_user: str,
        jwt_token: str,
        ssh_creds: dict = None,
        ips: list = None,
        api_version: str = None,
        **kwargs,  # We have this here to ignore extra arguments when calling from from_config
    ):
        """
        Slurm Cluster, another form of a BYO (bring-your-own) cluster.

        Args:
            name (str): Name for the Cluster, to re-use later on within Runhouse.
            url (str): URL of the existing Cluster (ex: ``http://*.**.**.**:6820``)
            auth_user (str): Username to authenticate with when making requests to the REST API
            jwt_token (str): the REST API (slurmrestd) authenticates via JWT (https://slurm.schedmd.com/jwt.html)
            ssh_creds (dict, optional): Once a job has been submitted, the node(s) involved can be accessed via SSH.

            .. note::
                While SSH credentials are not required for submitting jobs, they are necessary for accessing the node
                after the job has been submitted. (just as we allow for other cluster types)

            ips: (str, optional): Set of IP addresses for the cluster. If none are provided will
                extract them via the Slurm API (Default: ``None``).
            api_version (str, optional): REST API version to use. This will depend on the
                version of Slurm installed on the cluster. (Default: ``v0.0.36``)

        .. note::
            To build a cluster, please use the factory method :func:`cluster`.
        """

        self.url = url
        self.auth_user = auth_user
        self.jwt_token = jwt_token
        self.api_version = api_version or self.DEFAULT_API_VERSION
        self.ips = ips or self.cluster_ips()

        super().__init__(name=name, dryrun=True, ips=self.ips, ssh_creds=ssh_creds)

    @staticmethod
    def from_config(config: dict, dryrun=True, **kwargs):
        # Note: Dryrun is ignored here, since we don't want to start a ray instance on the cluster
        config = {**config, **kwargs}
        return SlurmCluster(**config)

    @property
    def config_for_rns(self):
        config = super().config_for_rns

        # Also store the ssh keys for the cluster in RNS
        config.update(
            {
                "url": self.url,
                "auth_user": self.auth_user,
                "jwt_token": self.jwt_token,
                "api_version": self.api_version,
            }
        )
        return config

    def cluster_ips(self) -> list:
        resp = requests.get(
            f"{self.url}/slurm/{self.api_version}/nodes",
            headers=self.slurmrestd_headers,
        )
        if resp.status_code != 200:
            raise Exception(f"Error getting cluster IPs: {resp.text}")
        resp_data = load_resp_content(resp)
        nodes = resp_data.get("nodes")
        return [node.get("address") for node in nodes]

    def ip_from_node_name(self, node_name: str) -> str:
        # https://slurm.schedmd.com/rest_api.html#slurmV0038GetClusterNodes
        resp = requests.get(
            f"{self.url}/slurm/{self.api_version}/node/{node_name}",
            headers=self.slurmrestd_headers,
        )
        if resp.status_code != 200:
            raise Exception(f"Error getting cluster nodes: {resp.text}")
        resp_data = load_resp_content(resp)
        try:
            nodes = resp_data.get("nodes")
            return nodes[0].get("address")
        except Exception as e:
            raise Exception(f"Failed to get node IP address: {e}")

    @property
    def slurmrestd_headers(self):
        """Headers required for all requests to the REST API"""
        # https://slurm.schedmd.com/slurmrestd.html
        return {
            "X-SLURM-USER-NAME": self.auth_user,
            "X-SLURM-USER-TOKEN": self.jwt_token,
            "Content-Type": "application/json",
        }

    def submit_job(self, payload: dict) -> int:
        """Submit a job to the Slurm Cluster. Add the IP of the node where the job is submitted to the list of IPs.
        API docs: https://slurm.schedmd.com/rest_api.html#slurmV0038SubmitJob
        Sample payload:
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
            f"{self.url}/slurm/{self.api_version}/job/submit",
            headers=self.slurmrestd_headers,
            data=json.dumps(payload),
        )
        if resp.status_code != 200:
            raise Exception(f"Error submitting job: {resp.text}")

        resp_data = load_resp_content(resp)
        job_id = resp_data.get("job_id")
        if not job_id:
            raise ValueError("Job ID not found - failed to submit job")

        node_ip: str = self.node_ip_for_submitted_job(job_id=job_id)
        logger.info(f"Submitted job {job_id} to cluster node {node_ip}")

        if node_ip not in self.ips:
            self.ips.append(node_ip)

        return node_ip

    def node_ip_for_submitted_job(self, job_id: int) -> str:
        resp = requests.get(
            f"{self.url}/slurm/{self.api_version}/job/{job_id}",
            headers=self.slurmrestd_headers,
        )
        if resp.status_code != 200:
            raise Exception(f"Error getting job info: {resp.text}")

        resp_data = load_resp_content(resp)
        job_node = resp_data.get("jobs")
        if not job_node:
            raise ValueError(f"Node for job {job_id} not found")

        node_name = job_node[0]["nodes"]
        return self.ip_from_node_name(node_name=node_name)

    def get_job_status(self, job_id: int) -> str:
        """Get the status of a job."""
        # TODO [JL]
        raise NotImplementedError

    def delete_job(self, job_id: int) -> str:
        """Delete a job."""
        # TODO [JL]
        raise NotImplementedError
