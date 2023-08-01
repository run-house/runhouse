import ast
import contextlib
import logging
import os
import pkgutil
import pty
import subprocess
import threading
import warnings
from pathlib import Path
from typing import Dict, Tuple, Union

import paramiko
import sagemaker
from sagemaker.estimator import EstimatorBase
from sagemaker.pytorch import PyTorch
from sshtunnel import SSHTunnelForwarder

from runhouse.rh_config import configs, rns_client
from runhouse.rns.hardware.cluster import Cluster
from runhouse.rns.utils.hardware import _current_cluster

logger = logging.getLogger(__name__)


class SageMakerCluster(Cluster):
    DEFAULT_HOST = "localhost"
    DEFAULT_INSTANCE_TYPE = "ml.m5.large"
    DEFAULT_USER = "root"
    SSH_KEY_PATH = "~/.ssh/sagemaker-ssh-gw"

    DEFAULT_SSH_PORT = 11022
    DEFAULT_HTTP_PORT = 50052
    DEFAULT_CONNECTION_WAIT_TIME = 600  # seconds

    def __init__(
        self,
        name,
        role: str = None,
        instance_type: str = None,
        instance_count: int = None,
        autostop_mins: int = None,
        connection_wait_time: int = None,
        estimator: Union[EstimatorBase, Dict] = None,
        dryrun=False,
        **kwargs,  # We have this here to ignore extra arguments when calling from from_config
    ):
        """
        The Runhouse SageMaker cluster abstraction. This is where you can run jobs (e.g. training, inference,
        batch transform, etc.), or simply spin up a new SageMaker instance which behaves in a similar way to
        OnDemand or BYO clusters.

        .. note::
            To build a cluster, please use the factory method :func:`sagemaker_cluster`.
        """
        self._role = role
        self._connection_wait_time = connection_wait_time
        self._instance_type = instance_type
        self._ssh_wrapper = None
        self._job_type = None

        self.instance_count = instance_count or 1
        self.instance_id = kwargs.get("instance_id")
        self.job_name = kwargs.get("job_name")

        self.autostop_mins = (
            autostop_mins
            if autostop_mins is not None
            else configs.get("default_autostop")
        )

        self.estimator = self._load_estimator(estimator)

        # Setting instance ID as cluster IP for compatibility with Cluster parent class methods
        super().__init__(
            name=name, ips=[self.instance_id], ssh_creds=None, dryrun=dryrun
        )

        if dryrun:
            return

        if self.estimator is None:
            # Create an estimator for the purpose of launching the instance
            self.estimator = self._create_mock_estimator()
        else:
            # Make sure the estimator's source file has the SSH SageMaker init
            # TODO [JL] this should be handled by Runhouse internally
            self._add_ssh_helper_init_to_estimator()

            # Running a dedicated training job
            self._job_type = self.estimator.JOB_CLASS_NAME

        if not self.instance_id:
            logger.info(
                f"Launching a new SageMaker instance on type: {self.instance_type}"
            )
            self._launch_new_instance()

            self.job_name = self._ssh_wrapper.training_job_name()
            logger.info(f"New SageMaker instance started with name: {self.job_name}")

            # Set the instance ID of the new SageMaker instance
            self.instance_id = self._get_instance_id()
            logger.info(f"New SageMaker instance started with ID: {self.instance_id}")

        self.check_server(restart_server=True)

        # Add the job name for this instance to the local SSH config
        self._add_ssh_config_entry()

        logger.info(
            "SSH connection has been created, you can now SSH into "
            f"the instance with the CLI using: ssh {self.name}"
        )

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
        if self.estimator and self._job_type:
            if isinstance(self.estimator, dict):
                config.update({"estimator": self.estimator})
                return config

            if type(self.estimator).__name__ == "PyTorch":
                estimator_attrs = [
                    "framework_version",
                    "py_version",
                    "instance_count",
                    "instance_type",
                    "role",
                    "image_uri",
                    "entry_point",
                    "source_dir",
                    "output_path",
                    "volume_size",
                ]
                selected_attrs = {
                    attr: getattr(self.estimator, attr, None)
                    for attr in estimator_attrs
                }
                # Estimator types: mxnet, tensorflow, keras, pytorch, onnx, xgboost
                config.update({"estimator": selected_attrs})
                config["estimator"]["framework"] = "PyTorch"

        return config

    @property
    def ssh_key_path(self):
        return os.path.expanduser(self.SSH_KEY_PATH)

    @property
    def ssh_config_file(self):
        return os.path.expanduser("~/.ssh/config")

    @property
    def role(self):
        if self._role:
            return self._role
        arn_role = os.getenv("ROLE_ARN")
        if arn_role:
            return arn_role
        try:
            execution_role_arn = sagemaker.get_execution_role()
            return execution_role_arn
        except Exception as e:
            raise e

    @role.setter
    def role(self, value):
        self._role = value

    @property
    def connection_wait_time(self):
        """Amount of time the SSH helper will wait inside SageMaker before it continues normal execution"""
        if self._connection_wait_time is not None:
            return self._connection_wait_time
        elif self._job_type == "training-job" or self.estimator:
            # Allow for connecting to the instance before the job starts (e.g. training)
            return self.DEFAULT_CONNECTION_WAIT_TIME
        else:
            # For inference and others, always up and running
            return 0

    @connection_wait_time.setter
    def connection_wait_time(self, value):
        self._connection_wait_time = value

    @property
    def instance_type(self):
        if self._instance_type:
            return self._instance_type
        elif self.estimator:
            return self.estimator.instance_type
        else:
            return self.DEFAULT_INSTANCE_TYPE

    @instance_type.setter
    def instance_type(self, value):
        self._instance_type = value

    def check_server(self, restart_server=True):
        if self.name == _current_cluster("name"):
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
                logger.info(f"Checking server with instance ID: {self.instance_id}")
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
                        f"Could not connect to SageMaker instance <{self.instance_id}>"
                    )

    def is_up(self) -> bool:
        """Check if the cluster is up.

        Example:
            >>> rh.cluster("sagemaker-cluster").is_up()
        """
        import boto3

        sagemaker_client = boto3.client("sagemaker")
        try:
            response = sagemaker_client.describe_training_job(
                TrainingJobName=self.job_name
            )
            status = response["TrainingJobStatus"]
            # Up if the instance is in progress
            return status == "InProgress"

        except sagemaker_client.exceptions.ResourceNotFound as e:
            raise e

    def teardown(self):
        """Teardown the SageMaker instance.

        Example:
            >>> rh.cluster("sagemaker-cluster").teardown()
        """
        self._stop_instance(delete_configs=False)

    def teardown_and_delete(self):
        """Teardown the SageMaker instance and delete from RNS configs.

        Example:
            >>> rh.cluster("sagemaker-cluster").teardown_and_delete()
        """
        self._stop_instance()

    def keep_warm(self, autostop_mins: int = -1):
        """Keep the cluster warm for given number of minutes after inactivity.

        Args:
            autostop_mins (int): Amount of time (in min) to keep the cluster warm after inactivity.
            If set to -1, keep cluster warm indefinitely. (Default: `-1`)
        """
        raise NotImplementedError

    @contextlib.contextmanager
    def pause_autostop(self):
        """Context manager to temporarily pause autostop."""
        raise NotImplementedError

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

                # Execute the command with the pseudo-terminal in a separate thread
                def run_ssh_tunnel():
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

                tunnel_thread = threading.Thread(target=run_ssh_tunnel)
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
                    (ssh_host, self.DEFAULT_SSH_PORT),
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
                num_ports_to_try -= 1
                pass

        return ssh_tunnel, local_port

    def ssh(self):
        """SSH into the cluster

        Example:
            >>> rh.cluster("sagemaker-cluster").ssh()
        """
        self.check_server(restart_server=True)
        # Alternative SSH command (if job name is not added to ~/.ssh/config):
        # ssh -i ~/.ssh/sagemaker-ssh-gw -p 11022 root@localhost
        subprocess.run(f"ssh {self.name}".split(" "))

    def _launch_new_instance(self):
        from sagemaker_ssh_helper.wrapper import SSHEstimatorWrapper

        logger.info("Setting up training job on new SageMaker instance")
        # Make sure the SSHEstimatorWrapper is being used by the estimator, this is necessary for
        # enabling the SSH tunnel to the SageMaker instance
        # https://github.com/aws-samples/sagemaker-ssh-helper/tree/main#step-2-modify-your-start-training-job-code
        dependency_dir = SSHEstimatorWrapper.dependency_dir()
        if dependency_dir not in self.estimator.dependencies:
            self.estimator.dependencies.append(dependency_dir)

        # Create the SSH wrapper & run the job
        self._ssh_wrapper = SSHEstimatorWrapper.create(
            self.estimator, connection_wait_time_seconds=self.connection_wait_time
        )

        if self.autostop_mins:
            # TODO [JL] incorporate autostop when launching
            logger.info(f"Launching instance with autostop: {self.autostop_mins}")

        # Call the SageMaker CreateTrainingJob API to start training
        self.estimator.fit(wait=False)

    def _rsync(self, source: str, dest: str, up: bool, contents: bool = False):
        command = (
            f"rsync -rvh --exclude='.git' "
            f"--exclude='.*' --exclude='venv*/' --exclude='dist/' --exclude='docs/' --exclude='__pycache__/' "
            f"--exclude='.*' -e 'ssh -o StrictHostKeyChecking=no -i {self.ssh_key_path} -p {self.DEFAULT_SSH_PORT}' "
            f"--delete {source} root@localhost:{dest}"
        )

        return_codes = self.run([command])
        if return_codes[0][0] != 0:
            logger.error(f"rsync to SageMaker cluster failed: {return_codes[0][1]}")

    def _stop_instance(self, delete_configs=True):
        """Stop the SageMaker instance. Optionally remove its config from RNS"""
        import boto3

        sagemaker_client = boto3.client("sagemaker")
        resp = sagemaker_client.stop_training_job(TrainingJobName=self.job_name)
        resp_metadata = resp["ResponseMetadata"]

        if resp_metadata["HTTPStatusCode"] == 200:
            logger.info(f"Successfully stopped cluster {self.instance_id}")
            if delete_configs:
                # Delete from RNS
                rns_client.delete_configs(resource=self)

                # Delete from ~/.ssh/config
                self._delete_ssh_config_entry()

                logger.info(f"Deleted SageMaker cluster {self.name}")
        else:
            raise Exception(f"Failed to stop cluster: {resp_metadata}")

    def _sync_runhouse_to_cluster(self, _install_url=None, env=None):
        if not self.instance_id:
            raise ValueError(f"No instance ID set for cluster {self.name}. Is it up?")

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

            # Connect to the instance
            ssh_client.connect(
                hostname=self.DEFAULT_HOST,
                port=self.DEFAULT_SSH_PORT,
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

        except (
            paramiko.AuthenticationException,
            paramiko.SSHException,
            Exception,
        ) as e:
            msg = (
                "Error occurred: {}\n\n"
                "Note: If you are experiencing connection errors, make sure you are using "
                "the latest version of paramiko and the AWS CLI:\npip install -U paramiko\n"
                "AWS CLI: https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html\n\n"
                "To check if the SSH tunnel with the instance is still up:\n"
                f"ssh {self.name}\n\n"
            ).format(str(e))
            warnings.warn(msg)

        finally:
            # Remember to close the SSH connection when you're done
            ssh_client.close()

    def _logfile_path(self, logfile):
        return f"/root/.rh/{logfile}"

    def _load_estimator(self, estimator: Union[Dict, EstimatorBase, None]):
        if estimator is None:
            return None

        if isinstance(estimator, EstimatorBase):
            return estimator

        if not isinstance(estimator, dict):
            raise TypeError(
                f"Unsupported estimator type. Expected dictionary or EstimatorBase, got {type(estimator)}"
            )

        estimator_framework = estimator.pop("framework", None)
        if estimator_framework is None:
            raise ValueError("Framework not specified in the config file.")

        elif estimator_framework == "PyTorch":
            # Re-build the estimator object from the saved params
            return PyTorch(**estimator)
        else:
            raise NotImplementedError("Currently only PyTorch Estimator is supported.")

    def _get_instance_id(self):
        try:
            return self._ssh_wrapper.get_instance_ids()[0]
        except Exception as e:
            raise e

    def _create_mock_estimator(self):
        """Create an estimator required for launching the instance.
        **Note: this is only meant to serve as a mock estimator object, not used for an actual training job
        but solely for the purpose of spinning up the SageMaker instance.**"""
        estimator_dict = {
            "instance_count": self.instance_count,
            "framework_version": "1.9.1",
            "py_version": "py38",
            "role": self.role,
            "entry_point": "launch_instance.py",
            "source_dir": os.path.abspath(
                os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    "../../scripts/sagemaker_cluster",
                )
            ),
            "instance_type": self.instance_type,
        }
        return PyTorch(**estimator_dict)

    def _add_ssh_helper_init_to_estimator(self):
        # https://github.com/aws-samples/sagemaker-ssh-helper#step-3-modify-your-training-script
        job_file_path = os.path.join(
            self.estimator.source_dir, self.estimator.entry_point
        )

        # Define the import line to be added
        ssh_install_cmd = "sagemaker_ssh_helper.setup_and_start_ssh()"
        import_line = f"import sagemaker_ssh_helper\n{ssh_install_cmd}\n\n"

        # Read the content of the Python file
        with open(job_file_path, "r") as f:
            code = f.read()

        if ssh_install_cmd in code:
            return

        tree = ast.parse(code)

        # Find the last import statement in the AST
        last_import = -1
        for node in ast.walk(tree):
            if isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
                last_import = max(last_import, node.end_lineno)

        if last_import == -1:
            # If there are no import statements, insert the code at the beginning of the file
            code = import_line + code
        else:
            lines = code.splitlines()
            insert_line = min(last_import + 1, len(lines))

            # Insert the code after the last import statement
            code = "\n".join(lines[:insert_line] + [import_line] + lines[insert_line:])

        with open(job_file_path, "w") as f:
            f.write(code)

    def _add_ssh_config_entry(self):
        """Update the SSH config to allow for accessing the cluster via: ssh <cluster name>"""
        config_file = self.ssh_config_file
        identity_file = self.ssh_key_path

        # Create the new entry
        new_entry = f"""\n# Added by Runhouse for SageMaker SSH Support\nHost {self.name}\n  HostName {self.DEFAULT_HOST}\n  IdentityFile {identity_file}\n  Port {self.DEFAULT_SSH_PORT}\n  User {self.DEFAULT_USER}\n"""  # noqa

        with open(config_file, "r") as f:
            existing_content = f.read()

        if new_entry in existing_content:
            return

        with open(config_file, "a") as f:
            f.write(new_entry)

    def _delete_ssh_config_entry(self):
        """Remove the SSH config entry from th for the cluster."""
        config_file = self.ssh_config_file

        with open(config_file) as f:
            lines = f.readlines()

        # Find the start and end lines of the entry to be deleted
        start_line = None
        for i, line in enumerate(lines):
            if line.strip().startswith("Host ") and self.name in line:
                start_line = i
                break

        if start_line is not None:
            end_line = len(lines)
            for i in range(start_line + 1, len(lines)):
                if lines[i].strip().startswith("Host "):
                    end_line = i
                    break

            # Remove the entry from the lines list (start at -1 to also delete the comment)
            del lines[start_line - 1 : end_line]

            with open(config_file, "w") as f:
                f.writelines(lines)
