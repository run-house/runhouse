import contextlib
import json
import logging
import os
import pkgutil
import pty
import re
import select
import subprocess
import sys
import textwrap
import threading
import warnings
from pathlib import Path
from typing import Dict, Tuple, Union

import paramiko
import sagemaker
from sagemaker.estimator import EstimatorBase
from sagemaker.mxnet import MXNet
from sagemaker.pytorch import PyTorch
from sagemaker.tensorflow import TensorFlow
from sagemaker.xgboost import XGBoost
from sshtunnel import SSHTunnelForwarder

from runhouse.rh_config import configs, open_cluster_tunnels, rns_client
from runhouse.rns.hardware.cluster import Cluster

logger = logging.getLogger(__name__)


class SageMakerCluster(Cluster):
    DEFAULT_HOST = "localhost"
    DEFAULT_INSTANCE_TYPE = "ml.m5.large"
    DEFAULT_USER = "root"
    # Default path for source code copied to SageMaker instance
    ESTIMATOR_SRC_CODE_PATH = "/opt/ml/code"
    DEFAULT_REGION = "us-east-1"

    DEFAULT_SSH_PORT = 11022
    DEFAULT_HTTP_PORT = 50052
    DEFAULT_CONNECTION_WAIT_TIME = 60  # seconds

    def __init__(
        self,
        name,
        role: str = None,
        instance_type: str = None,
        instance_count: int = None,
        autostop_mins: int = None,
        connection_wait_time: int = None,
        estimator: Union[EstimatorBase, Dict] = None,
        job_name: str = None,
        dryrun=False,
        **kwargs,  # We have this here to ignore extra arguments when calling from from_config
    ):
        """
        The Runhouse SageMaker cluster abstraction. This is where you can run training jobs, or simply spin up a
        new SageMaker cluster as a compute backend which behaves in the same way as an OnDemand or BYO cluster.

        .. note::
            To build a cluster, please use the factory method :func:`sagemaker_cluster`.
        """
        self._connection_wait_time = connection_wait_time
        self._instance_type = instance_type
        self._instance_count = instance_count
        self._ssh_wrapper = None

        # Relevant if estimator is provided
        self._estimator_entry_point = kwargs.get("estimator_entry_point")
        self._estimator_source_dir = kwargs.get("estimator_source_dir")
        self._estimator_framework = kwargs.get("estimator_framework")

        self.instance_id = kwargs.get("instance_id")
        self.job_name = job_name

        self.autostop_mins = (
            autostop_mins
            if autostop_mins is not None
            else configs.get("default_autostop")
        )

        self.estimator = self._load_estimator(estimator)
        self.role = self._set_role(role)

        # Setting instance ID as cluster IP for compatibility with Cluster parent class methods
        super().__init__(
            name=name, ips=[self.instance_id], ssh_creds=None, dryrun=dryrun
        )

        if not dryrun and not self.is_up():
            logger.info("Preparing to launch a new SageMaker cluster")
            self._prepare_and_launch_new_cluster()

    @property
    def config_for_rns(self):
        config = super().config_for_rns
        config.update(
            {
                "instance_id": self.instance_id,
                "role": self.role,
                "job_name": self.job_name,
                "instance_type": self.instance_type,
                "instance_count": self.instance_count,
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
                    if self.is_serializable(value)
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
    def ssh_key_path(self):
        return os.path.expanduser("~/.ssh/sagemaker-ssh-gw")

    @property
    def hosts_path(self):
        return os.path.expanduser("~/.ssh/known_hosts")

    @property
    def ssh_config_file(self):
        return os.path.expanduser("~/.ssh/config")

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
        # TODO [JL] Run CLI command instead? something like: ``aws configure get region``
        import boto3

        try:
            region = boto3.session.Session().region_name
            if region is None:
                return self.DEFAULT_REGION
            return region
        except:
            return self.DEFAULT_REGION

    @staticmethod
    def is_serializable(obj):
        try:
            json.dumps(obj)
            return True
        except (TypeError, OverflowError):
            return False

    def check_server(self, restart_server=True):
        if self.on_this_cluster():
            return

        if not self.instance_id:
            raise ValueError("SageMaker cluster has no instance ID")

        if not self.client:
            cluster_config = self.config_for_rns

            try:
                self.address = (
                    self.instance_id
                )  # For compatibility with parent method (``connect_server_client``)
                self.connect_server_client()
                logger.info(
                    f"Checking server {self.name} with instance ID: {self.instance_id}"
                )

                self.client.check_server(cluster_config=cluster_config)
                logger.info(f"Server {self.instance_id} is up.")
            except Exception as e:
                logger.warning(f"Could not connect to {self.instance_id}: {e}")
                if restart_server:
                    logger.info(
                        f"Server {self.instance_id} is up, but the HTTP server may not be up."
                    )

                    # Make sure screen and a compatible version of protobuf for the sagemaker-ssh helper library
                    # are installed before installing runhouse and restarting the server
                    # TODO [JL] do we still need this?
                    self.run(
                        [
                            "sudo apt-get install screen -y",
                            "pip install protobuf==3.20.3",
                        ]
                    )
                    self.restart_server(resync_rh=True, restart_ray=True)
                    logger.info(f"Checking server {self.instance_id} again.")
                    self.client.check_server(cluster_config=cluster_config)
                else:
                    raise ValueError(
                        f"Could not connect to SageMaker instance {self.instance_id}"
                    )

    def is_up(self) -> bool:
        """Check if the cluster is up.

        Example:
            >>> rh.sagemaker_cluster("sagemaker-cluster").is_up()
        """
        import boto3

        sagemaker_client = boto3.client("sagemaker", region_name=self.default_region)
        try:
            response = sagemaker_client.describe_training_job(
                TrainingJobName=self.job_name
            )
            status = response["TrainingJobStatus"]
            # Up if the instance is in progress
            return status == "InProgress"

        except:
            return False

    def up(self):
        return_codes = self.restart_server(restart_ray=True)
        if return_codes[0][0] != 0:
            raise ValueError(
                f"Could not restart or connect to server {self.name}: {return_codes[0]}"
            )

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
            autostop_mins (int): Amount of time (in min) to keep the cluster warm after inactivity.
            If set to -1, keep cluster warm indefinitely. (Default: `-1`)
        """
        self._update_autostop(autostop_mins)

    @contextlib.contextmanager
    def pause_autostop(self):
        """Context manager to temporarily pause autostop."""
        self._update_autostop(autostop_mins=-1)
        yield
        self._update_autostop(self.autostop_mins)

    def _run_training_job(self):
        """Call the SageMaker CreateTrainingJob API to start the training job on the cluster."""
        # TODO [JL] Keeping private until re-running training jobs on the same cluster is supported
        if not self.estimator:
            logger.warning("No estimator found, cannot run training job.")
            return

        self.estimator.fit(wait=False, job_name=self.job_name)

    def ssh_tunnel(
        self, local_port, remote_port=None, num_ports_to_try: int = 0
    ) -> Tuple[SSHTunnelForwarder, int]:
        # https://github.com/aws-samples/sagemaker-ssh-helper/tree/main#forwarding-tcp-ports-over-ssh-tunnel
        ssh_port = self.DEFAULT_SSH_PORT
        connected = False
        ssh_tunnel = None

        while not connected:
            try:
                if local_port > local_port + num_ports_to_try:
                    raise Exception(
                        f"Failed to create SSH tunnel after {num_ports_to_try} attempts"
                    )
                command = (
                    f'sm-local-start-ssh "{self.instance_id}" -R localhost:12345:localhost:12345 '
                    f"-L localhost:{local_port}:localhost:{local_port} "
                    f"-L localhost:{ssh_port}:localhost:22"
                )

                # Define an event to signal completion of the SSH tunnel setup
                tunnel_setup_complete = threading.Event()

                # Manually allocate a pseudo-terminal to prevent a "pseudo-terminal not allocated" error
                master_fd, slave_fd = pty.openpty()

                def setup_ssh_tunnel():
                    # Execute the command with the pseudo-terminal in a separate thread
                    process = subprocess.Popen(
                        command,
                        shell=True,
                        stdin=slave_fd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        close_fds=True,
                        preexec_fn=os.setsid,
                    )

                    # Close the slave file descriptor as we don't need it
                    os.close(slave_fd)

                    # Close the master file descriptor after reading the output
                    os.close(master_fd)

                    # Wait for the process to complete and collect its return code
                    process.wait()

                    # Signal that the tunnel setup is complete
                    tunnel_setup_complete.set()

                tunnel_thread = threading.Thread(target=setup_ssh_tunnel)
                tunnel_thread.daemon = True  # Set the thread as a daemon, so it won't block the main thread

                # Start the SSH tunnel thread
                tunnel_thread.start()

                # Wait until the connection to the instance has been established before forming the SSHTunnelForwarder
                # https://github.com/aws-samples/sagemaker-ssh-helper/blob/main/sagemaker_ssh_helper/sm-local-start-ssh
                tunnel_setup_complete.wait(timeout=30)

                # Continue with the creation of the SSHTunnelForwarder object
                ssh_host = self.DEFAULT_HOST
                ssh_username = self.DEFAULT_USER
                ssh_key_path = self.ssh_key_path
                remote_bind_addresses = ("127.0.0.1", remote_port or local_port)
                local_bind_addresses = ("", local_port)

                ssh_tunnel = SSHTunnelForwarder(
                    (ssh_host, ssh_port),
                    ssh_username=ssh_username,
                    ssh_pkey=ssh_key_path,
                    remote_bind_address=remote_bind_addresses,
                    local_bind_address=local_bind_addresses,
                    set_keepalive=1,
                )

                # Start the SSH tunnel
                ssh_tunnel.start()

                logger.info(
                    "SSH connection has been successfully created with the cluster"
                )
                logger.info(
                    "Note: You can also SSH directly onto the cluster via the CLI: "
                    f"``ssh {self.name}``"
                )
                connected = True
            except Exception as e:
                logger.warning(e)
                # try connecting with a different port - most likely the issue is the port is already taken
                local_port += 1
                ssh_port += 1
                pass

        return ssh_tunnel, local_port

    def ssh(self):
        """SSH into the cluster.
        **Note**: Via the CLI can SSH via ``ssh cluster-name``

        Example:
            >>> rh.sagemaker_cluster(name="sagemaker-cluster").ssh()
        """
        master_fd, slave_fd = pty.openpty()
        ssh_process = subprocess.Popen(
            ["ssh", self.name],
            stdin=slave_fd,
            stdout=slave_fd,
            stderr=slave_fd,
            universal_newlines=True,
        )

        # Close the slave_fd in the parent process as it's not needed there
        os.close(slave_fd)

        # Wait for the SSH process to initialize
        select.select([master_fd], [], [])

        # Interact with the SSH process through the master_fd
        try:
            while True:
                if master_fd in select.select([master_fd], [], [], 0)[0]:
                    output = os.read(master_fd, 1024).decode()
                    print(output, end="")

                if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                    user_input = sys.stdin.readline()
                    try:
                        os.write(master_fd, user_input.encode())
                    except OSError:
                        pass

                    # terminate the SSH process gracefully
                    if user_input.strip() == "exit":
                        break
        except Exception as e:
            raise e
        finally:
            # Close the master_fd and terminate the SSH process when done
            os.close(master_fd)
            ssh_process.terminate()

    def _create_launch_estimator(self):
        """Create the estimator object used for launching the cluster. If a custom estimator is provided, use that.
        Otherwise, use a Runhouse default estimator to launch.
        **Note If an estimator is provided, Runhouse will override the entry point and source dir to use the
        default Runhouse entry point and source dir. This is to ensure that the connection to the cluster
        can be maintained even if the job fails, or if autostop is enabled.**"""
        default_entry_point = "launch_instance.py"
        default_source_dir = os.path.join(
            os.path.abspath(
                os.path.join(os.path.abspath(os.path.dirname(__file__)), "..", "..")
            ),
            "scripts",
            "sagemaker_cluster",
        )
        if self.estimator:
            # Save this to be used on the cluster for running the estimator
            self._estimator_entry_point = self.estimator.entry_point
            self._estimator_source_dir = self.estimator.source_dir

            # Update the estimator with the Runhouse custom entry point and source dir
            self.estimator.entry_point = default_entry_point
            self.estimator.source_dir = default_source_dir

            return self.estimator

        else:
            # No estimator provided, use the Runhouse custom estimator (using PyTorch by default)
            estimator_dict = {
                "instance_count": self.instance_count,
                "framework_version": "1.9.1",
                "py_version": "py38",
                "role": self.role,
                "entry_point": default_entry_point,
                "source_dir": default_source_dir,
                "instance_type": self.instance_type,
                # https://docs.aws.amazon.com/sagemaker/latest/dg/train-warm-pools.html
                "keep_alive_period_in_seconds": 3600,
            }

            return PyTorch(**estimator_dict)

    def _prepare_and_launch_new_cluster(self):
        self.estimator = self._create_launch_estimator()

        logger.info(
            f"Launching a new SageMaker cluster (instance count={self.instance_count}) on type: {self.instance_type}"
        )

        self._launch_new_instance()

        self.job_name = self._ssh_wrapper.training_job_name()

        # Set the instance ID of the new SageMaker instance
        self.instance_id = self._get_instance_id()
        logger.info(f"New SageMaker instance started with ID: {self.instance_id}")

        self.check_server(restart_server=True)

        if self._estimator_source_dir and self._estimator_entry_point:
            # Copy the estimator's code to the cluster - Runhouse will then manage running the job
            self._sync_estimator_to_cluster()

        # Add the cluster name to the local SSH config to enable the <ssh cluster_name> command
        self._add_ssh_config_entry()

        logger.info(
            f"Connection with {self.name} has been created. You can now SSH onto "
            f"the cluster with the CLI using: ``ssh {self.name}``"
        )

    def _launch_new_instance(self):
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

        self._run_training_job()

    def _rsync(self, source: str, dest: str, up: bool, contents: bool = False):
        command = (
            f"rsync -rvh --exclude='.git' "
            f"--exclude='venv*/' --exclude='dist/' --exclude='docs/' --exclude='__pycache__/' --exclude='.*' "
            f"--include='.rh/' -e 'ssh -o StrictHostKeyChecking=no -i {self.ssh_key_path} -p {self.DEFAULT_SSH_PORT}' "
            f"--delete {source} root@localhost:{dest}"
        )

        return_codes = self.run([command])
        if return_codes[0][0] != 0:
            logger.error(f"rsync to SageMaker cluster failed: {return_codes[0][1]}")

    def _get_instance_id(self):
        try:
            return self._ssh_wrapper.get_instance_ids()[0]
        except Exception as e:
            raise e

    def _set_role(self, role: str = None):
        """Set the role required for launching and connecting to the SageMaker instance. If no role is provided
        explicitly or via the estimator, try to get the default SageMaker role configured in the local environment."""
        if role:
            return role

        if self.estimator:
            return self.estimator.role

        try:
            return sagemaker.get_execution_role()
        except Exception as e:
            raise e

    def _stop_instance(self, delete_configs=True):
        """Stop the SageMaker instance. Optionally remove its config from RNS"""
        import boto3

        try:
            sagemaker_client = boto3.client("sagemaker")
            resp = sagemaker_client.stop_training_job(TrainingJobName=self.job_name)
            resp_metadata = resp["ResponseMetadata"]

            if resp_metadata["HTTPStatusCode"] != 200:
                raise Exception(f"Failed to stop cluster: {resp_metadata}")

            logger.info(f"Successfully stopped cluster {self.instance_id}")
            if delete_configs:
                # Delete from RNS
                rns_client.delete_configs(resource=self)

                # Delete from ~/.ssh/config
                self._delete_ssh_config_entry()

                logger.info(f"Deleted SageMaker cluster {self.name}")

        except Exception as e:
            raise e

    def _sync_runhouse_to_cluster(self, _install_url=None, env=None):
        if not self.instance_id:
            raise ValueError(f"No instance ID set for cluster {self.name}. Is it up?")

        # Sync the local ~/.rh directory to the cluster
        self._rsync(
            source=f"{os.path.expanduser('~/.rh')}/",
            dest="~/.rh",
            up=True,
            contents=True,
        )
        logger.info("Synced .rh config to the cluster")

        local_rh_package_path = Path(pkgutil.get_loader("runhouse").path).parent

        # Check if runhouse is installed from source and has setup.py
        if (
            not _install_url
            and local_rh_package_path.parent.name == "runhouse"
            and (local_rh_package_path.parent / "setup.py").exists()
        ):
            # Package is installed in editable mode
            local_rh_package_path = local_rh_package_path.parent
            dest_path = "~/"

            self._rsync(
                source=str(local_rh_package_path),
                dest=dest_path,
                up=True,
                contents=True,
            )
            rh_install_cmd = "pip install ./runhouse"
        else:
            if not _install_url:
                import runhouse

                _install_url = f"runhouse=={runhouse.__version__}"
            rh_install_cmd = f"pip install {_install_url}"

        install_cmd = (
            f"{env._activate_cmd} && {rh_install_cmd}" if env else rh_install_cmd
        )
        status_codes = self.run([install_cmd])

        if status_codes[0][0] != 0:
            raise ValueError(f"Error installing runhouse on cluster <{self.name}>")

    def _run_commands_with_ssh(
        self,
        commands: list,
        cmd_prefix: str,
        stream_logs: bool,
        port_forward: int = None,
        require_outputs: bool = True,
    ):
        return_codes = []
        ssh_client = paramiko.SSHClient()

        try:
            # Automatically add the server's host key to the known_hosts file
            ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh_client.load_system_host_keys()

            try:
                connected_ssh_port = self._connected_ssh_port()
            except ValueError:
                connected_ssh_port = self.DEFAULT_SSH_PORT

            # Connect to the instance
            ssh_client.connect(
                hostname=self.DEFAULT_HOST,
                port=connected_ssh_port,
                username=self.DEFAULT_USER,
                key_filename=self.ssh_key_path,
                allow_agent=False,
                look_for_keys=False,
            )

            for command in commands:
                if command.startswith("rsync"):
                    # Use subprocess to run rsync locally and specify the ssh command as an argument
                    # Since rsync is not a simple shell command, we can't use the SSH client
                    result = subprocess.run(
                        command,
                        shell=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                    )
                    stdout = result.stdout
                    stderr = result.stderr
                    return_code = result.returncode
                else:
                    stdin, stdout, stderr = ssh_client.exec_command(command)
                    stdout = stdout.read().decode()
                    stderr = stderr.read().decode()

                    return_code = (
                        1 if ("Traceback" in stderr and "ERROR" in stderr) else 0
                    )

                return_codes.append((return_code, stdout, stderr))

            return return_codes

        except paramiko.BadHostKeyException as e:
            # Handle error along the lines of: "Host key for server 'localhost' does not match: got 'X', expected 'Y'"
            # Delete the old keys from previous SageMaker clusters from the known hosts file and retry connecting
            logger.warning(e)
            self._filter_known_hosts()
            self._run_commands_with_ssh(
                commands=commands,
                cmd_prefix=cmd_prefix,
                stream_logs=stream_logs,
                port_forward=port_forward,
                require_outputs=require_outputs,
            )
        except (
            paramiko.AuthenticationException,
            paramiko.SSHException,
            Exception,
        ) as e:
            msg = (
                "Error occurred: {}\n\n"
                "Note: If you are experiencing connection errors, make sure you are using "
                "the latest version of paramiko and the AWS CLI:\n``pip install -U paramiko``\n"
                "AWS CLI: https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html\n\n"
                f"To check if the SSH tunnel with the instance is still up: ``ssh {self.name}``"
            ).format(str(e))
            warnings.warn(msg)

        finally:
            # Remember to close the SSH connection when you're done
            ssh_client.close()

    def _connected_ssh_port(self) -> int:
        """Get opened SSH port for connecting to the SageMaker cluster."""
        open_tunnels: tuple = open_cluster_tunnels.get(self.instance_id)
        if open_tunnels is None:
            raise ValueError(
                f"No tunnels are currently open for the cluster with name {self.name} "
                f"and instance id: {self.instance_id}."
            )

        connected_ssh_port: int = open_tunnels[0].ssh_port
        return connected_ssh_port

    def _load_estimator(self, estimator: Union[Dict, EstimatorBase, None]):
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

    def _update_autostop(self, autostop_mins: int = None):
        cluster_config = self.config_for_rns
        cluster_config["autostop_mins"] = autostop_mins or -1
        if not self.client:
            self.connect_server_client()
        self.client.check_server(cluster_config=cluster_config)

    def _filter_known_hosts(self):
        known_hosts = self.hosts_path
        valid_hosts = []
        with open(known_hosts, "r") as f:
            for line in f:
                if not line.strip().startswith(f"[{self.DEFAULT_HOST}]"):
                    valid_hosts.append(line)

        with open(known_hosts, "w") as f:
            f.writelines(valid_hosts)

    def _add_ssh_config_entry(self):
        """Update the SSH config to allow for accessing the cluster via: ssh <cluster name>"""
        connected_ssh_port = self._connected_ssh_port()

        config_file = self.ssh_config_file
        identity_file = self.ssh_key_path

        # Create the new entry
        new_entry = textwrap.dedent(
            f"""
            # Added by Runhouse for SageMaker SSH Support (Job Name: {self.job_name})
            Host {self.name}
              HostName {self.DEFAULT_HOST}
              IdentityFile {identity_file}
              Port {connected_ssh_port}
              User {self.DEFAULT_USER}
        """
        )

        with open(config_file, "r") as f:
            existing_config = f.read()

        pattern = rf"\(Job Name: {re.escape(self.job_name)}\)"
        match = re.search(pattern, existing_config)
        if match:
            return

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
