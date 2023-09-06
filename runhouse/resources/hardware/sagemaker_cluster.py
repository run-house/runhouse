import configparser
import contextlib
import logging
import os
import pkgutil
import pty
import re
import select
import shlex
import socket
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path
from typing import Dict, Tuple, Union

import pkg_resources

try:
    import boto3
    import sagemaker
    from sagemaker.estimator import EstimatorBase
    from sagemaker.mxnet import MXNet
    from sagemaker.pytorch import PyTorch
    from sagemaker.tensorflow import TensorFlow
    from sagemaker.xgboost import XGBoost
except ImportError:
    pass

from sshtunnel import BaseSSHTunnelForwarderError, SSHTunnelForwarder

from runhouse.globals import configs, open_cluster_tunnels, rns_client
from runhouse.rns.utils.api import is_jsonable, resolve_absolute_path

from .cluster import Cluster
from .utils import SkySSHRunner

logger = logging.getLogger(__name__)


class SageMakerCluster(Cluster):
    DEFAULT_HOST = "localhost"
    DEFAULT_INSTANCE_TYPE = "ml.m5.large"
    DEFAULT_REGION = "us-east-1"
    DEFAULT_USER = "root"
    ECR_URL = "763104351884.dkr.ecr.us-east-1.amazonaws.com"

    # Default path for any estimator source code copied onto the cluster
    ESTIMATOR_SRC_CODE_PATH = "/opt/ml/code"
    ESTIMATOR_LOG_FILE = "sm_cluster.out"

    DEFAULT_SSH_PORT = 11022
    DEFAULT_HTTP_PORT = 50052
    DEFAULT_CONNECTION_WAIT_TIME = 60  # seconds

    def __init__(
        self,
        name: str,
        role: str = None,
        profile: str = None,
        ssh_key_path: str = None,
        instance_id: str = None,
        instance_type: str = None,
        instance_count: int = None,
        image_uri: str = None,
        autostop_mins: int = None,
        connection_wait_time: int = None,
        estimator: Union["EstimatorBase", Dict] = None,
        job_name: str = None,
        dryrun=False,
        **kwargs,
    ):
        """
        The Runhouse SageMaker cluster abstraction. This is where you can use SageMaker as a compute backend, just as
        you would an on-demand cluster (i.e. cloud VMs) or a BYO (i.e. on-prem) cluster. Additionally supports running
        dedicated training jobs using SageMaker Estimators.

        .. note::
            To build a cluster, please use the factory method :func:`sagemaker_cluster`.
        """
        self._connection_wait_time = connection_wait_time
        self._instance_type = instance_type
        self._instance_count = instance_count
        self._ssh_key_path = ssh_key_path

        # SSHEstimatorWrapper to facilitate the SSH connection to the cluster
        self._ssh_wrapper = None

        # Keep track of ports forwarded for the SSH tunnel
        self._local_port = self.DEFAULT_HTTP_PORT
        self._ssh_port = self.DEFAULT_SSH_PORT

        # Relevant if estimator is provided
        self._estimator_entry_point = kwargs.get("estimator_entry_point")
        self._estimator_source_dir = kwargs.get("estimator_source_dir")
        self._estimator_framework = kwargs.get("estimator_framework")

        self.instance_id = instance_id
        self.job_name = job_name

        self.autostop_mins = (
            autostop_mins
            if autostop_mins is not None
            else configs.get("default_autostop")
        )

        self.estimator = self._load_estimator(estimator)

        # Default sagemaker session - to be overwritten by the profile if provided explicitly or thru a role
        self._sagemaker_session = self._set_sagemaker_session()

        # Role ARN is required for using the SageMaker APIs - can be provided explicitly or extracted from the profile
        self.role, self.profile = self._set_role_and_profile(role, profile)

        self.image_uri = self._set_image_uri(image_uri)

        # Note: Setting instance ID as cluster IP for compatibility with Cluster parent class methods
        super().__init__(name=name, ips=[self.instance_id], ssh_creds={}, dryrun=dryrun)

    @property
    def config_for_rns(self):
        config = super().config_for_rns
        config.update(
            {
                "instance_id": self.instance_id,
                "role": self.role,
                "profile": self.profile,
                "ssh_key_path": self.ssh_key_path,
                "job_name": self.job_name,
                "instance_type": self.instance_type,
                "instance_count": self.instance_count,
                "image_uri": self.image_uri,
                "autostop_mins": self.autostop_mins,
                "connection_wait_time": self.connection_wait_time,
            }
        )

        # If running a dedicated job on the cluster, add the estimator config
        if self.estimator and (
            self._estimator_source_dir and self._estimator_entry_point
        ):
            config.update(
                {
                    "estimator_entry_point": self._estimator_entry_point,
                    "estimator_source_dir": self._estimator_source_dir,
                }
            )

            if isinstance(self.estimator, EstimatorBase):
                # Serialize the estimator before saving it down in the config
                selected_attrs = {
                    key: value
                    for key, value in self.estimator.__dict__.items()
                    if is_jsonable(value)
                }
                # Estimator types: mxnet, tensorflow, keras, pytorch, onnx, xgboost
                self._estimator_framework = type(self.estimator).__name__
                config.update(
                    {
                        "estimator": selected_attrs,
                        "estimator_framework": self._estimator_framework,
                    }
                )

            if isinstance(self.estimator, dict):
                config.update({"estimator": self.estimator})

        return config

    @property
    def hosts_path(self):
        return os.path.expanduser("~/.ssh/known_hosts")

    @property
    def ssh_config_file(self):
        return os.path.expanduser("~/.ssh/config")

    @property
    def ssh_key_path(self):
        """Relative path to the private SSH key used to connect to the cluster."""
        if self._ssh_key_path:
            return self._relative_ssh_path(self._ssh_key_path)

        # Default path
        return "~/.ssh/sagemaker-ssh-gw"

    @ssh_key_path.setter
    def ssh_key_path(self, ssh_key_path):
        self._ssh_key_path = ssh_key_path

    @property
    def instance_count(self):
        if self._instance_count:
            return self._instance_count
        elif self.estimator:
            return self.estimator.instance_count
        else:
            return 1

    @instance_count.setter
    def instance_count(self, instance_count):
        self._instance_count = instance_count

    @property
    def connection_wait_time(self):
        """Amount of time the SSH helper will wait inside SageMaker before it continues normal execution"""
        if self._connection_wait_time is not None:
            return self._connection_wait_time
        elif self.estimator and (
            self._estimator_source_dir and self._estimator_entry_point
        ):
            # Allow for connecting to the instance before the job starts (e.g. training)
            return self.DEFAULT_CONNECTION_WAIT_TIME
        else:
            # For inference and others, always up and running
            return 0

    @connection_wait_time.setter
    def connection_wait_time(self, connection_wait_time):
        self._connection_wait_time = connection_wait_time

    @property
    def instance_type(self):
        if self._instance_type:
            return self._instance_type
        elif self.estimator:
            return self.estimator.instance_type
        else:
            return self.DEFAULT_INSTANCE_TYPE

    @instance_type.setter
    def instance_type(self, instance_type):
        self._instance_type = instance_type

    @property
    def default_region(self):
        try:
            region = self._sagemaker_session.boto_region_name
            if region is None:
                return self.DEFAULT_REGION
            return region
        except:
            return self.DEFAULT_REGION

    @property
    def s3_keys_path(self):
        """Path to public key stored for the cluster on S3. When initializing the cluster, the public key
        is copied by default to this location."""
        return f"s3://{self.default_bucket}/ssh-authorized-keys/"

    @property
    def default_bucket(self):
        """Default bucket to use for storing the cluster's public SSH key."""
        return self._sagemaker_session.default_bucket()

    @property
    def _extra_ssh_args(self):
        """Extra SSH arguments to be used when connecting to the cluster."""
        # Note - port 12345 can be used for Python Debug Server: "-R localhost:12345:localhost:12345"
        # https://github.com/aws-samples/sagemaker-ssh-helper#remote-debugging-with-pycharm-debug-server-over-ssh
        return (
            f"-L localhost:{self._local_port}:localhost:{self._local_port} "
            f"-L localhost:{self._ssh_port}:localhost:22"
        )

    @property
    def _env_activate_cmd(self):
        """Prefix for commands run on the cluster. Ensure we are running all commands in the conda environment
        and not the system default python."""
        # TODO [JL] Can SageMaker handle this for us?
        return "source /opt/conda/bin/activate"

    @property
    def _abs_ssh_key_path(self):
        return resolve_absolute_path(self.ssh_key_path)

    @classmethod
    def _relative_ssh_path(cls, ssh_path: str):
        """Convert to rel path if not already one."""
        if ssh_path.startswith("~"):
            return ssh_path

        # Convert to a relative path
        relative_path = os.path.relpath(ssh_path, os.path.expanduser("~"))
        relative_path = relative_path.replace("\\", "/")

        if not relative_path.startswith("~"):
            relative_path = f"~/{relative_path}"

        return relative_path

    # -------------------------------------------------------
    # Cluster State & Lifecycle Methods
    # -------------------------------------------------------
    def check_server(self, restart_server=True):
        if self.on_this_cluster():
            return

        if not self.instance_id:
            raise ValueError(f"SageMaker cluster {self.name} has no instance ID")

        if not self.client:
            cluster_config = self.config_for_rns

            try:
                self.connect_server_client()
                logger.info(
                    f"Checking server {self.name} with instance ID: {self.instance_id}"
                )

                self.client.check_server(cluster_config=cluster_config)
                logger.info(f"Server {self.instance_id} is up.")
            except:
                if restart_server:
                    logger.info(
                        f"Server {self.instance_id} is up, but the HTTP server may not be up."
                    )
                    self.run(
                        [
                            "sudo apt-get install screen -y "
                            "&& sudo apt-get install rsync -y"
                        ]
                    )
                    # Restart the server inside the base conda env
                    self.restart_server(
                        resync_rh=True,
                        restart_ray=True,
                        env_activate_cmd=self._env_activate_cmd,
                    )
                    logger.info(f"Checking server {self.instance_id} again.")

                    self.client.check_server(cluster_config=cluster_config)
                else:
                    raise ValueError(
                        f"Could not connect to SageMaker instance {self.instance_id}"
                    )

    def up(self):
        """Up the cluster.

        Example:
            >>> rh.sagemaker_cluster("sagemaker-cluster").up()
        """
        logger.info("Preparing to launch a new SageMaker cluster")
        self._launch_new_cluster()

    def up_if_not(self):
        """Bring up the cluster if it is not up. No-op if cluster is already up.

        Example:
            >>> rh.sagemaker_cluster("sagemaker-cluster").up_if_not()
        """
        if not self.is_up():
            self.address = None
            self.job_name = None
            self.instance_id = None
            self.up()
        return self

    def is_up(self) -> bool:
        """Check if the cluster is up.

        Example:
            >>> rh.sagemaker_cluster("sagemaker-cluster").is_up()
        """
        try:
            resp = self.status()
            status = resp.get("TrainingJobStatus")
            # Up if the instance is in progress
            return status == "InProgress"
        except:
            return False

    def teardown(self):
        """Teardown the SageMaker instance.

        Example:
            >>> rh.sagemaker_cluster(name="sagemaker-cluster").teardown()
        """
        self._stop_instance(delete_configs=False)

    def teardown_and_delete(self):
        """Teardown the SageMaker instance and delete from RNS configs.

        Example:
            >>> rh.sagemaker_cluster(name="sagemaker-cluster").teardown_and_delete()
        """
        self._stop_instance()

    def keep_warm(self, autostop_mins: int = -1):
        """Keep the cluster warm for given number of minutes after inactivity.

        Args:
            autostop_mins (int): Amount of time (in minutes) to keep the cluster warm after inactivity.
                If set to ``-1``, keep cluster warm indefinitely. (Default: ``-1``)
        """
        self._update_autostop(autostop_mins)

    def __getstate__(self):
        """Delete non-serializable elements (e.g. sagemaker session object) before pickling."""
        state = self.__dict__.copy()
        state["_sagemaker_session"] = None
        state["client"] = None
        state["_rpc_tunnel"] = None
        return state

    @contextlib.contextmanager
    def pause_autostop(self):
        """Context manager to temporarily pause autostop."""
        self._update_autostop(autostop_mins=-1)
        yield
        self._update_autostop(self.autostop_mins)

    def status(self) -> dict:
        """
        Get status of SageMaker cluster.

        Example:
            >>> status = rh.sagemaker_cluster("sagemaker-cluster").status()
        """
        try:
            return self._sagemaker_session.describe_training_job(self.job_name)
        except:
            return {}

    # -------------------------------------------------------
    # SSH APIs
    # -------------------------------------------------------
    def ssh_tunnel(
        self, local_port, remote_port=None, num_ports_to_try: int = 0, retry=True
    ) -> Tuple[SSHTunnelForwarder, int]:
        try:
            remote_bind_addresses = ("127.0.0.1", local_port)
            local_bind_addresses = ("", local_port)

            ssh_tunnel = SSHTunnelForwarder(
                (self.DEFAULT_HOST, self._ssh_port),
                ssh_username=self.DEFAULT_USER,
                ssh_pkey=self._abs_ssh_key_path,
                remote_bind_address=remote_bind_addresses,
                local_bind_address=local_bind_addresses,
                set_keepalive=1800,
            )

            # Start the SSH tunnel
            ssh_tunnel.start()

            # Update the SSH config for the cluster with the connected SSH port
            self._add_or_update_ssh_config_entry()

            logger.info("SSH connection has been successfully created with the cluster")

        except BaseSSHTunnelForwarderError as e:
            if not retry:
                # Failed to create the SSH tunnel object even after successfully refreshing the SSM session
                raise e

            # Refresh the SSM session, which should bind the HTTP and SSH ports to localhost which are forwarded
            # to the cluster
            self._refresh_ssm_session_with_cluster(num_ports_to_try)

            # Retry creating the SSH tunnel once the session has been refreshed
            return self.ssh_tunnel(
                local_port, remote_port, num_ports_to_try, retry=False
            )

        return ssh_tunnel, local_port

    def ssh(self, interactive: bool = True):
        """SSH into the cluster.

        Args:
            interactive (bool): Whether to start an interactive shell or not (Default: ``True``).

        Example:
            >>> rh.sagemaker_cluster(name="sagemaker-cluster").ssh()
        """

        if self.instance_id not in open_cluster_tunnels:
            # Make sure SSM session and SSH tunnels are up before running the command
            self.connect_server_client()

        if not interactive:
            logger.info(
                f"Created SSH tunnel with the cluster. To SSH into the cluster, run: `ssh {self.name}`"
            )
            return

        head_fd, worker_fd = pty.openpty()
        ssh_process = subprocess.Popen(
            ["ssh", "-o", "StrictHostKeyChecking=no", self.name],
            stdin=worker_fd,
            stdout=worker_fd,
            stderr=worker_fd,
            universal_newlines=True,
        )

        # Close the worker_fd in the parent process as it's not needed there
        os.close(worker_fd)

        # Wait for the SSH process to initialize
        select.select([head_fd], [], [])

        # Interact with the SSH process through the head_fd
        try:
            while True:
                if head_fd in select.select([head_fd], [], [], 0)[0]:
                    output = os.read(head_fd, 1024).decode()
                    print(output, end="")

                if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                    user_input = sys.stdin.readline()
                    try:
                        os.write(head_fd, user_input.encode())
                    except OSError:
                        pass

                    # terminate the SSH process gracefully
                    if user_input.strip() == "exit":
                        break
        except Exception as e:
            raise e
        finally:
            # Close the head_fd and terminate the SSH process when done
            os.close(head_fd)
            ssh_process.terminate()

    def _run_commands_with_ssh(
        self,
        commands: list,
        cmd_prefix: str,
        stream_logs: bool,
        port_forward: int = None,
        require_outputs: bool = True,
    ):
        return_codes = []
        for command in commands:
            if command.startswith("rsync"):
                try:
                    result = subprocess.run(
                        command,
                        shell=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                    )
                    return_codes.append(
                        (result.returncode, result.stdout, result.stderr)
                    )
                except subprocess.CalledProcessError as e:
                    return_codes.append((255, "", str(e)))
            else:
                if self.instance_id not in open_cluster_tunnels:
                    # Make sure tunnel is up before running commands (e.g. calling ``restart_server`` right after
                    # loading the cluster with dryrun)
                    self.connect_server_client()

                # Host can be replaced with name (as reflected in the ~/.ssh/config file)
                runner = SkySSHRunner(
                    self.name,
                    ssh_user=self.DEFAULT_USER,
                    ssh_private_key=self._abs_ssh_key_path,
                )
                command = f"{cmd_prefix} {command}" if cmd_prefix else command
                logger.info(f"Running command on {self.name}: {command}")
                return_code, stdout, stderr = runner.run(
                    command,
                    require_outputs=require_outputs,
                    stream_logs=stream_logs,
                    port_forward=port_forward,
                )

                if (
                    return_code != 0
                    and "dpkg: error processing package install-info" in stdout
                ):
                    # **NOTE**: there may be issues with some SageMaker GPUs post installation script which
                    # leads to an error which looks something like: "installed install-info package post-installation
                    # script subprocess returned error exit status 2"
                    # https://askubuntu.com/questions/1034961/cant-upgrade-error-etc-environment-source-not-found-and-error-processin
                    # /etc/environment file may also be corrupt, replacing with an empty file allows
                    # subsequent python commands to run
                    self._run_commands_with_ssh(
                        commands=[
                            "cd /var/lib/dpkg/info && sudo rm *.postinst "
                            "&& sudo mv /etc/environment /etc/environment_broken "
                            "&& sudo touch /etc/environment "
                            f"&& {command}"
                        ],
                        cmd_prefix=cmd_prefix,
                        stream_logs=stream_logs,
                    )

                return_codes.append((return_code, stdout, stderr))

        return return_codes

    # -------------------------------------------------------
    # Cluster Provisioning & Launching
    # -------------------------------------------------------
    def _create_launch_estimator(self):
        """Create the estimator object used for launching the cluster. If a custom estimator is provided, use that.
        Otherwise, use a Runhouse default estimator to launch.
        **Note If an estimator is provided, Runhouse will override the entry point and source dir to use the
        default Runhouse entry point and source dir. This is to ensure that the connection to the cluster
        can be maintained even if the job fails, after it has completed, or if autostop is enabled.**"""
        # Note: these entry points must point to the existing local files
        default_entry_point = "launch_instance.py"
        default_source_dir = pkg_resources.resource_filename(
            "runhouse", "scripts/sagemaker_cluster"
        )
        if self.estimator:
            # Save the original entry point and source dir to be used on the cluster for running the estimator
            self._estimator_entry_point = self.estimator.entry_point
            self._estimator_source_dir = self.estimator.source_dir

            # Update the estimator with the Runhouse custom entry point and source dir
            # When the job is initialized, it will run through the Runhouse entry point, which will manage the
            # running of the custom estimator
            self.estimator.entry_point = default_entry_point
            self.estimator.source_dir = default_source_dir

            return self.estimator

        else:
            # No estimator provided, use the Runhouse custom estimator (using PyTorch by default)
            estimator_dict = {
                "instance_count": self.instance_count,
                "role": self.role,
                "image_uri": self.image_uri,
                "framework_version": "2.0.1",
                "py_version": "py310",
                "entry_point": default_entry_point,
                "source_dir": default_source_dir,
                "instance_type": self.instance_type,
                # https://docs.aws.amazon.com/sagemaker/latest/dg/train-warm-pools.html
                "keep_alive_period_in_seconds": 3600,
            }

            return PyTorch(**estimator_dict)

    def _launch_new_cluster(self):
        self.estimator = self._create_launch_estimator()

        logger.info(
            f"Launching a new SageMaker cluster (instance count={self.instance_count}) on instance "
            f"type: {self.instance_type}"
        )

        self._create_new_instance()

        # If no name provided, use the autogenerated name
        self.job_name = self.estimator.latest_training_job.name

        self.instance_id = self._cluster_instance_id()

        # For compatibility with parent Cluster class methods which use an address
        self.address = self.instance_id

        logger.info(f"New SageMaker instance started with ID: {self.instance_id}")

        # Remove stale entries from the known hosts file
        self._filter_known_hosts()

        logger.info("Creating session with cluster via SSM")
        self._create_ssm_session_with_cluster()

        self.check_server()

        if self._estimator_source_dir and self._estimator_entry_point:
            # Copy the provided estimator's code to the cluster - Runhouse will then manage running the job in order
            # to preserve control over the cluster's autostop
            self._sync_estimator_to_cluster()
            logger.info(
                f"Logs for the estimator can be viewed on the cluster in "
                f"path: {self.ESTIMATOR_SRC_CODE_PATH}/{self.ESTIMATOR_LOG_FILE}"
            )

        logger.info(
            f"Connection with {self.name} has been created. You can SSH onto "
            f"the cluster with the CLI using: ``ssh {self.name}``"
        )

    def _create_new_instance(self):
        from sagemaker_ssh_helper.wrapper import SSHEstimatorWrapper

        # Make sure the SSHEstimatorWrapper is being used by the estimator, this is necessary for
        # enabling the SSH tunnel to the cluster
        # https://github.com/aws-samples/sagemaker-ssh-helper/tree/main#step-2-modify-your-start-training-job-code
        ssh_dependency_dir = SSHEstimatorWrapper.dependency_dir()
        if ssh_dependency_dir not in self.estimator.dependencies:
            self.estimator.dependencies.append(ssh_dependency_dir)

        # Create the SSH wrapper & run the job
        self._ssh_wrapper = SSHEstimatorWrapper.create(
            self.estimator, connection_wait_time_seconds=self.connection_wait_time
        )

        self._start_instance()

    def _create_ssm_session_with_cluster(self, num_ports_to_try: int = 5):
        """Create a session with the cluster. Runs a bash script containing a series of commands which use existing
        SSH keys or generate new ones needed to authorize the connection with the cluster via the AWS SSM.
        These commands are run when the cluster is initially provisioned, or for subsequent connections if the session
        is longer active. Once finished, the SSH port and HTTP port will be bound to processes on
        localhost which are forwarded to the cluster."""
        # https://github.com/aws-samples/sagemaker-ssh-helper/tree/main#forwarding-tcp-ports-over-ssh-tunnel
        base_command = self._load_base_command_for_ssm_session()

        # Ports to forward from local host to the cluster
        extra_ssh_args = self._extra_ssh_args
        command = f"{base_command} {extra_ssh_args}"

        connected = False
        while not connected:
            extra_ssh_args = self._extra_ssh_args
            command = f"{base_command} {extra_ssh_args}"

            try:
                if num_ports_to_try == 0:
                    raise ConnectionError(
                        f"Failed to create SSM session and connect to {self.name} after repeated attempts."
                        f"Make sure SSH keys exist in local path: {self._abs_ssh_key_path}. The public key must "
                        f"also match the key stored remotely in s3 bucket: {self.s3_keys_path}"
                    )

                logger.info(f"Running command: {command}")

                # Define an event to signal completion of the SSH tunnel setup
                tunnel_setup_complete = threading.Event()

                # Manually allocate a pseudo-terminal to prevent a "pseudo-terminal not allocated" error
                head_fd, worker_fd = pty.openpty()

                def ssm_session_and_port_forwarding():
                    # Execute the command with the pseudo-terminal in a separate thread
                    process = subprocess.Popen(
                        command,
                        shell=True,
                        stdout=subprocess.PIPE,
                        stdin=subprocess.PIPE,
                    )

                    # Close the worker file descriptor as we don't need it
                    os.close(worker_fd)

                    # Close the master file descriptor after reading the output
                    os.close(head_fd)

                    # Wait for the process to complete and collect its return code
                    process.wait()

                    # Signal that the tunnel setup is complete
                    tunnel_setup_complete.set()

                tunnel_thread = threading.Thread(target=ssm_session_and_port_forwarding)
                tunnel_thread.daemon = True  # Set the thread as a daemon, so it won't block the main thread

                # Start the SSH tunnel thread
                tunnel_thread.start()

                # Give time for the SSM session to start, SSH keys to be copied onto the cluster, and the SSH port
                # forwarding command to run
                tunnel_setup_complete.wait(timeout=30)

                if not self._ports_are_in_use():
                    # Command should bind SSH port and HTTP port on localhost, if this is not the case try re-running
                    # the bash script with a different set of ports
                    # E.g. ❯ lsof -i:11022,50052
                    # COMMAND   PID   USER   FD   TYPE             DEVICE SIZE/OFF NODE NAME
                    # ssh     97115 myuser    3u  IPv4 0xcf81f230786cc9fd      0t0  TCP localhost:50052 (LISTEN)
                    # ssh     97115 myuser    6u  IPv4 0xcf81f230786eff6d      0t0  TCP localhost:11022 (LISTEN)
                    raise ConnectionError

                # Update the SSH config for the cluster with the connected SSH port
                self._add_or_update_ssh_config_entry()

                connected = True

                logger.info(
                    f"Created SSM session using ports {self._ssh_port} and {self._local_port}. "
                    f"All active sessions can be viewed with: ``aws ssm describe-sessions --state Active``"
                )

            except (ConnectionError, subprocess.CalledProcessError):
                # Try re-running with updated the ports - possible the ports are already in use
                self._local_port += 1
                self._ssh_port += 1
                num_ports_to_try -= 1
                pass

    def _start_instance(self):
        """Call the SageMaker CreateTrainingJob API to start the training job on the cluster."""
        # TODO [JL] Note: Keeping private until re-running training jobs on the same cluster is supported
        if not self.estimator:
            logger.warning("No estimator found, cannot run job.")
            return

        try:
            self.estimator.fit(wait=False, job_name=self.job_name)
        except Exception as e:
            raise e

    def _load_base_command_for_ssm_session(self) -> str:
        """Bash script command for creating the SSM session and uploading the SSH keys to the cluster. Will try
        reusing existing keys locally if they exist, otherwise will generate new ones."""
        ssh_key_path = self._abs_ssh_key_path

        # Run script which creates a new SSM session with the cluster (and optionally generates new SSH keys)
        script_path = pkg_resources.resource_filename(
            "runhouse", "scripts/sagemaker_cluster/start-ssm-proxy-connection.sh"
        )
        os.chmod(script_path, 0o755)

        authorized_key_path = self.s3_keys_path

        base_command = (
            f'bash {script_path} "{self.instance_id}" '
            f'"{authorized_key_path}" "{ssh_key_path}" "{self.default_region}"'
        )

        if Path(ssh_key_path).exists():
            # Run script to create SSM session and connect to the cluster using existing SSH keys
            # Script copies the existing public key to the cluster
            logger.info(
                f"Using existing SSH keys to connect to cluster from path: {ssh_key_path}"
            )
            return f"{base_command} False"
        else:
            logger.warning(
                f"No keys found in path: {ssh_key_path}. Generating new keys and uploading the new public key to s3 "
                f"bucket: {authorized_key_path}"
            )
            return f"{base_command} True"

    def _refresh_ssm_session_with_cluster(self, num_ports_to_try: int = 5):
        """Reconnect to the cluster via the AWS SSM. This bypasses the step of creating a new SSH key which was already
        done when upping the cluster. Note: this assumes the session has previously been created, which we do when
        the cluster has been upped.

        To view all sessions: ``aws ssm describe-sessions --state Active``
        """
        # https://github.com/aws-samples/sagemaker-ssh-helper/blob/main/sagemaker_ssh_helper/sm-connect-ssh-proxy
        script_path = pkg_resources.resource_filename(
            "runhouse", "scripts/sagemaker_cluster/refresh-ssm-session.sh"
        )
        os.chmod(script_path, 0o755)

        # Remove stale entries from the known hosts file - this is import for avoiding collisions when
        # subsequent clusters are created as the IP address is added to the file as: [localhost]:11022
        self._filter_known_hosts()

        num_ports_initial = num_ports_to_try
        connected = False
        ssh_key_path = self._abs_ssh_key_path

        while not connected:
            command = [
                script_path,
                self.instance_id,
                ssh_key_path,
                self.default_region,
            ] + shlex.split(self._extra_ssh_args)

            if num_ports_to_try == 0:
                raise ConnectionError(
                    f"Failed to create connection with {self.name} after {num_ports_initial} attempts "
                    f"(cluster status=`{self.status().get('TrainingJobStatus')}`). Make sure the public key in "
                    f"local path: {ssh_key_path} matches the key stored in bucket: {self.s3_keys_path}, "
                    f"that a connection to another SageMaker cluster is not already active, "
                    f"and that AWS CLI V2 is installed. If the problem persists, try running the command "
                    f"to create the session manually: `bash {' '.join(command)}`"
                )

            try:
                subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    stdin=subprocess.PIPE,
                )

                # Give enough time for the aws ssm + ssh port forwarding commands in the script to complete
                # Better to wait a few more seconds than to restart the HTTP server and Ray on the cluster unnecessarily
                time.sleep(8)

                if not self._ports_are_in_use():
                    # Command should bind SSH port and HTTP port on localhost, if this is not the case try re-running
                    # with different ports
                    raise socket.error

                connected = True
                logger.info(f"Successfully refreshed SSM session with {self.name}")

            except socket.error:
                # If the refresh didn't work try connecting with a different port - could be that the port
                # is already taken
                self._local_port += 1
                self._ssh_port += 1
                num_ports_to_try -= 1
                pass

    # -------------------------------------------------------
    # Helpers
    # -------------------------------------------------------
    def _rsync(self, source: str, dest: str, up: bool, contents: bool = False):
        source = source + "/" if not source.endswith("/") else source
        dest = dest + "/" if not dest.endswith("/") else dest

        command = (
            f"rsync -rvh --exclude='.git' --exclude='venv*/' --exclude='dist/' --exclude='docs/' "
            f"--exclude='__pycache__/' --exclude='.*' "
            f"--include='.rh/' -e 'ssh -o StrictHostKeyChecking=no "
            f"-i {self._abs_ssh_key_path} -p {self._ssh_port}' {source} root@localhost:{dest}"
        )

        logger.info(f"Syncing {source} to: {dest} on cluster")
        return_codes = self.run([command])
        if return_codes[0][0] != 0:
            logger.error(f"rsync to SageMaker cluster failed: {return_codes[0][1]}")

    def _cluster_instance_id(self):
        try:
            return self._ssh_wrapper.get_instance_ids()[0]
        except Exception as e:
            raise e

    def _set_role_and_profile(self, role: str = None, profile: str = None):
        """Set the SageMaker role and profile used for launching and connecting to the SageMaker instance.
        Role can be provided as an ARN or a name. If provided, will search for the profile containing this role
        in local AWS configs. If no profile is provided, try loading from the environment variable ``AWS_PROFILE``,
        otherwise default to using the ``default`` profile."""
        if self.estimator:
            # If using an estimator must provide a name or full ARN
            # https://sagemaker.readthedocs.io/en/stable/api/training/estimators.html
            role = self.estimator.role

        if role and role.startswith("arn:aws"):
            profile = profile or self._filter_aws_configs_for_profile(role)
            return role, profile

        try:
            # https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-roles.html
            aws_profile = profile or os.environ.get("AWS_PROFILE")
            if aws_profile is None:
                aws_profile = "default"
                logger.warning(
                    f"No profile provided or environment variable set to `AWS_PROFILE`, using the {aws_profile} profile"
                )

            logger.info(f"Loading SageMaker execution role from: {aws_profile}")

            # Load the execution role using the profile provided
            boto_session = boto3.Session(profile_name=aws_profile)

            # Overwrite the default sagemaker session to with a session using the profile provided
            self._sagemaker_session = self._set_sagemaker_session(
                boto_session=boto_session
            )

            role = sagemaker.get_execution_role(
                sagemaker_session=self._sagemaker_session
            )
            profile = aws_profile or self._filter_aws_configs_for_profile(role)

            return role, profile

        except Exception as e:
            raise e

    def _filter_aws_configs_for_profile(self, role_arn: str) -> Union[str, None]:
        """Find the profile associated with a particular role ARN. If no profile is found, return None."""
        try:
            aws_dir = os.path.expanduser("~/.aws")
            credentials_path = os.path.join(aws_dir, "credentials")
            config_path = os.path.join(aws_dir, "config")

            profiles_with_role = set()

            for path in [credentials_path, config_path]:
                config = configparser.ConfigParser()
                config.read(path)

                for section in config.sections():
                    if "role_arn" in config[section]:
                        if config[section]["role_arn"] == role_arn:
                            # Add just the name of the profile (not the full section heading)
                            profiles_with_role.add(section.split(" ")[-1])

            if not profiles_with_role:
                return None

            # TODO [JL] multiple profiles with the same role?
            return list(profiles_with_role)[0]

        except Exception as e:
            logger.warning(f"Could not find a profile for role {role_arn}: {e}")
            return None

    def _set_image_uri(self, image_uri: str = None) -> str:
        """Set the docker image URI used for launching the SageMaker instance. If no image URI is provided, use
        a default image based on the instance type."""
        if image_uri:
            return image_uri

        if self.estimator:
            return self.estimator.image_uri

        return self._base_image_uri()

    def _set_sagemaker_session(self, boto_session: boto3.Session = None):
        """Create a SageMaker session required for using the SageMaker APIs. If none is provided
        create one using the default region."""
        if self.estimator:
            return self.estimator.sagemaker_session

        boto_session = boto_session or boto3.Session(region_name=self.default_region)
        return sagemaker.Session(boto_session=boto_session)

    def _stop_instance(self, delete_configs=True):
        """Stop the SageMaker instance. Optionally remove its config from RNS."""
        try:
            self._sagemaker_session.stop_training_job(job_name=self.job_name)

            if not self.is_up():
                raise Exception(f"Failed to stop cluster {self.name}")

            logger.info(f"Successfully stopped cluster {self.name}")

            # Remove stale host key(s) from known hosts
            self._filter_known_hosts()

            if delete_configs:
                # Delete from RNS
                rns_client.delete_configs(resource=self)

                # Delete entry from ~/.ssh/config
                self._delete_ssh_config_entry()
                logger.info(f"Deleted cluster {self.name} from configs")

        except Exception as e:
            raise e

    def _sync_runhouse_to_cluster(self, _install_url=None, env=None):
        if not self.instance_id:
            raise ValueError(f"No instance ID set for cluster {self.name}. Is it up?")

        # Sync the local ~/.rh directory to the cluster
        self._rsync(
            source=os.path.expanduser("~/.rh"),
            dest="~/.rh",
            up=True,
            contents=True,
        )
        logger.info("Synced ~/.rh folder to the cluster")

        local_rh_package_path = Path(pkgutil.get_loader("runhouse").path).parent

        # **Note** temp patch to handle PyYAML errors: https://github.com/yaml/pyyaml/issues/724
        base_rh_install_cmd = (
            f"{self._env_activate_cmd} && python3 -m pip install 'cython<3.0.0'"
        )

        # Check if runhouse is installed from source and has setup.py
        if (
            not _install_url
            and local_rh_package_path.parent.name == "runhouse"
            and (local_rh_package_path.parent / "setup.py").exists()
        ):
            # Package is installed in editable mode
            local_rh_package_path = local_rh_package_path.parent
            dest_path = f"~/{local_rh_package_path.name}"

            self._rsync(
                source=str(local_rh_package_path),
                dest=dest_path,
                up=True,
                contents=True,
            )

            rh_install_cmd = (
                f"{base_rh_install_cmd} && python3 -m pip install ./runhouse[sagemaker]"
            )
        else:
            if not _install_url:
                import runhouse

                _install_url = f"runhouse[sagemaker]=={runhouse.__version__}"
            rh_install_cmd = (
                f"{base_rh_install_cmd} && python3 -m pip install {_install_url}"
            )

        status_codes = self.run([rh_install_cmd])

        if status_codes[0][0] != 0:
            raise ValueError(
                f"Error installing runhouse on cluster: {status_codes[0][1]}"
            )

    def _load_estimator(
        self, estimator: Union[Dict, "EstimatorBase", None]
    ) -> Union[None, "EstimatorBase"]:
        """Build an Estimator object from config"""
        if estimator is None:
            return None

        if isinstance(estimator, EstimatorBase):
            return estimator

        if not isinstance(estimator, dict):
            raise TypeError(
                f"Unsupported estimator type. Expected dictionary or EstimatorBase, got {type(estimator)}"
            )
        # Re-build the estimator object from its config
        estimator_framework = self._estimator_framework
        if estimator_framework == "PyTorch":
            return PyTorch(**estimator)
        elif estimator_framework == "TensorFlow":
            return TensorFlow(**estimator)
        elif estimator_framework == "MXNet":
            return MXNet(**estimator)
        elif estimator_framework == "XGBoost":
            return XGBoost(**estimator)
        else:
            raise NotImplementedError(
                f"Unsupported estimator framework {estimator_framework}"
            )

    def _sync_estimator_to_cluster(self):
        """If providing a custom estimator sync over the estimator's source directory to the cluster"""
        from runhouse import folder

        estimator_folder = folder(
            path=os.path.expanduser(self._estimator_source_dir)
        ).to(self, path=self.ESTIMATOR_SRC_CODE_PATH)
        logger.info(
            f"Synced estimator source directory to the cluster in path: {estimator_folder.path}"
        )

    def _base_image_uri(self):
        """Pick a default image for the cluster based on its instance type"""
        # https://github.com/aws/deep-learning-containers/blob/master/available_images.md
        # TODO [JL] Add flexibility for py version & framework_version
        gpu_instance_types = ["p2", "p3", "p4", "g3", "g4", "g5"]

        image_type = (
            "gpu"
            if any(prefix in self.instance_type for prefix in gpu_instance_types)
            else "cpu"
        )

        cuda_version = "cu118-" if image_type == "gpu" else ""
        image_url = (
            f"{self.ECR_URL}/pytorch-training:2.0.1-{image_type}-py310-{cuda_version}"
            f"ubuntu20.04-sagemaker"
        )

        return image_url

    def _update_autostop(self, autostop_mins: int = None):
        cluster_config = self.config_for_rns
        cluster_config["autostop_mins"] = autostop_mins or -1
        if not self.client:
            self.connect_server_client()
        # Update the config on the server with the new autostop time
        self.client.check_server(cluster_config=cluster_config)

    # -------------------------------------------------------
    # Port Management
    # -------------------------------------------------------
    def _ports_are_in_use(self) -> bool:
        """Check if the ports used for port forwarding from localhost to the cluster are in use."""
        try:
            self._bind_ports_to_localhost()
            # Ports are not in use
            return False
        except OSError:
            # At least one of the ports is in use
            return True

    def _bind_ports_to_localhost(self):
        """Try binding the SSH and HTTP ports to localhost to check if they are in use."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s1:
            s1.bind(("localhost", self._ssh_port))
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s2:
            s2.bind(("localhost", self._local_port))

    # -------------------------------------------------------
    # SSH config
    # -------------------------------------------------------
    def _filter_known_hosts(self):
        """To prevent host key collisions in the ~/.ssh/known_hosts file, remove any existing entries of localhost.
        Since the connection to the cluster is made via localhost, remove stale entries"""
        known_hosts = self.hosts_path
        if not Path(known_hosts).exists():
            # e.g. in a collab or notebook environment
            return

        valid_hosts = []
        with open(known_hosts, "r") as f:
            for line in f:
                if not line.strip().startswith(f"[{self.DEFAULT_HOST}]"):
                    valid_hosts.append(line)

        with open(known_hosts, "w") as f:
            f.writelines(valid_hosts)

    def _ssh_config_entry(
        self,
        port: int,
        name: str = None,
        hostname: str = None,
        identity_file: str = None,
        user: str = None,
    ):
        return textwrap.dedent(
            f"""
            # Added by Runhouse for SageMaker SSH Support
            Host {name or self.name}
              HostName {hostname or self.DEFAULT_HOST}
              IdentityFile {identity_file or self._abs_ssh_key_path}
              Port {port}
              User {user or self.DEFAULT_USER}
        """
        )

    def _add_or_update_ssh_config_entry(self):
        """Update the SSH config to allow for accessing the cluster via: ssh <cluster name>"""
        connected_ssh_port = self._ssh_port
        config_file = self.ssh_config_file

        with open(config_file, "r") as f:
            existing_config = f.read()

        pattern = re.compile(
            rf"^\s*Host\s+{re.escape(self.name)}\s*$.*?(?=^\s*Host\s+|\Z)",
            re.MULTILINE | re.DOTALL,
        )

        entry_match = pattern.search(existing_config)

        if entry_match:
            # If entry already exists update the port with the connected SSH port (may have changed from previous
            # connection attempt)
            existing_entry = entry_match.group()
            updated_entry = re.sub(
                r"(?<=Port )\d+", str(connected_ssh_port), existing_entry
            )
            updated_config = existing_config.replace(existing_entry, updated_entry)
            with open(config_file, "w") as f:
                f.write(updated_config)
        else:
            # Otherwise, add the new entry to the config file
            new_entry = self._ssh_config_entry(port=connected_ssh_port)
            with open(config_file, "a") as f:
                f.write(new_entry)

    def _delete_ssh_config_entry(self):
        """Remove the SSH config entry for the cluster."""
        config_file = self.ssh_config_file

        with open(config_file) as f:
            lines = f.readlines()

        # Find the start and end lines of the entry to be deleted
        start_line = None
        for i, line in enumerate(lines):
            if self.job_name in line:
                start_line = i
                break

        if not start_line:
            return

        # Find the end of the entry (next empty line)
        end_line = start_line
        while end_line < len(lines) and lines[end_line].strip():
            end_line += 1

        del lines[start_line:end_line]

        with open(config_file, "w") as f:
            f.writelines(lines)
