import ast
import logging
import os
import pkgutil
import pty
import subprocess
import threading
import warnings
from pathlib import Path
from typing import Dict, Union

import paramiko
from sagemaker.pytorch import PyTorch
from sagemaker_ssh_helper.wrapper import SSHEstimatorWrapper
from sshtunnel import SSHTunnelForwarder

from runhouse.rh_config import rns_client
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
    CONNECTION_WAIT_TIME = 600  # seconds

    def __init__(
        self,
        name,
        arn_role: str,
        instance_type: str = None,
        estimator: Union["sagemaker.estimator.EstimatorBase", Dict] = None,
        connection_wait_time: int = None,
        dryrun=False,
        **kwargs,  # We have this here to ignore extra arguments when calling from from_config
    ):
        """
        The Runhouse SageMaker cluster abstraction. This is where you can run jobs (e.g. training, inference,
        batch transform, etc.), or simply spin up a new SageMaker instance which behaves in a similar way to
        OnDemand or BYO clusters.

        .. note::
            To build a cluster, please use the factory method :func:`cluster`.
        """
        self.arn_role = arn_role
        self.estimator = estimator
        self.instance_id = kwargs.get("instance_id")
        self.job_name = kwargs.get("job_name")
        self.job_type = kwargs.get("job_type")

        self.connection_wait_time = self.set_connection_wait(connection_wait_time)
        self.ssh_process = None

        if isinstance(self.estimator, dict):
            # Convert to an estimator object to use for creating and running the job
            self.estimator = self._reload_estimator_from_config(self.estimator)

        self.instance_type = self.set_instance_type(instance_type)

        super().__init__(
            name=name, ips=[self.instance_id], ssh_creds=None, dryrun=dryrun
        )

        if not dryrun:
            if self.estimator is None:
                # The instance will be launched using an estimator as a SageMaker training job,
                # but will not actually run a training script (this is just as a way to launch the instance)
                self.estimator = self._create_mock_estimator()
            else:
                # TODO [JL] support inference jobs
                # Make sure the estimator's source file has the SSH SageMaker init
                self._add_ssh_helper_init_to_estimator()

                # Running a dedicated training job
                self.job_type = self.estimator.JOB_CLASS_NAME

            if not self.instance_id:
                logger.info("Launching a new SageMaker instance")
                self._launch_new_instance()

            self.check_server(restart_server=True)  # TODO make ``restart`` dynamic?

            # Add the job name for this instance to the local SSH config
            self._add_ssh_config_entry(self.job_name)

            logger.info(
                "SSH connection has been created, you can now SSH into "
                f"the instance with the CLI using:\nssh {self.job_name}"
            )

    @property
    def config_for_rns(self):
        config = super().config_for_rns
        config.update(
            {
                "instance_id": self.instance_id,
                "arn_role": self.arn_role,
                "job_name": self.job_name,
                "instance_type": self.instance_type,
                "job_type": self.job_type,
                "connection_wait_time": self.connection_wait_time,
            }
        )
        if self.estimator and self.job_type:
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

    def set_instance_type(self, instance_type: str = None):
        if instance_type:
            return instance_type
        elif self.estimator:
            return self.estimator.instance_type
        else:
            return self.DEFAULT_INSTANCE_TYPE

    def set_connection_wait(self, connection_wait_time: int = None) -> int:
        """Amount of time the SSH helper will wait inside SageMaker before it continues normal execution"""
        if connection_wait_time:
            return connection_wait_time
        elif self.job_type == "training-job":
            # Allow for connecting to the instance before training starts
            return self.CONNECTION_WAIT_TIME
        else:
            # For inference and others, always up and running
            return 0

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
        # Stop the SageMaker training job
        self._stop_instance()

    def ssh_tunnel(self, local_port, remote_port=None, num_ports_to_try: int = 0):
        # https://github.com/aws-samples/sagemaker-ssh-helper/tree/main#forwarding-tcp-ports-over-ssh-tunnel
        # To test working connection: ssh -i ~/.ssh/sagemaker-ssh-gw -p 11022 root@localhost
        command = (
            f'sm-local-start-ssh "{self.instance_id}" -R localhost:12345:localhost:12345 '
            f"-L localhost:{self.DEFAULT_HTTP_PORT}:localhost:{self.DEFAULT_HTTP_PORT} "
            f"-L localhost:{self.DEFAULT_SSH_PORT}:localhost:22"
        )

        # Define an event to signal completion of the SSH tunnel setup
        tunnel_setup_complete = threading.Event()

        logger.info("Creating SSH tunnel with instance...")
        try:
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
                logger.info(f"SSH tunnel process with PID={process.pid} completed.")

                # Signal that the tunnel setup is complete
                tunnel_setup_complete.set()

            tunnel_thread = threading.Thread(target=run_ssh_tunnel)
            tunnel_thread.daemon = (
                True  # Set the thread as a daemon, so it won't block the main thread
            )

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
                "SSH connection has been successfully created with the instance"
            )

            logger.info(
                "**Note**: You can also create an active SSH tunnel via the CLI:\n"
                f"ssh -i ~/.ssh/sagemaker-ssh-gw -p {self.DEFAULT_SSH_PORT} root@localhost"
            )
        except Exception as e:
            logger.error(
                f"Failed to create SSH connection with SageMaker instance {self.instance_id}: {e}"
            )

        finally:
            return ssh_tunnel, self.DEFAULT_HTTP_PORT

    def _launch_new_instance(self):
        logger.info("Setting up training job on new SageMaker instance")
        # Make sure the SSHEstimatorWrapper is being used by the estimator, this is necessary for
        # enabling the SSH tunnel to the SageMaker instance
        # https://github.com/aws-samples/sagemaker-ssh-helper/tree/main#step-2-modify-your-start-training-job-code
        dependency_dir = SSHEstimatorWrapper.dependency_dir()
        if dependency_dir not in self.estimator.dependencies:
            self.estimator.dependencies.append(dependency_dir)

        # Create the SSH wrapper & run the job
        self.ssh_wrapper = SSHEstimatorWrapper.create(
            self.estimator, connection_wait_time_seconds=self.connection_wait_time
        )

        # Call the SageMaker CreateTrainingJob API to start training
        self.estimator.fit(wait=False)
        self.job_name = self.ssh_wrapper.training_job_name()

        # Set the instance ID of the new SageMaker instance
        self.instance_id = self._get_instance_id()
        logger.info(f"New SageMaker instance started with ID: {self.instance_id}")

    def _rsync(self, source: str, dest: str, up: bool, contents: bool = False):
        command = (
            f"rsync -rvh --exclude='.git' "
            f"--exclude='.*' --exclude='venv*/' --exclude='dist/' --exclude='docs/' --exclude='__pycache__/' "
            f"--exclude='.*' -e 'ssh -i {self.ssh_key_path} -p {self.DEFAULT_SSH_PORT}' "
            f"--delete {source} root@localhost:{dest}"
        )

        return_codes = self.run([command])

        if return_codes[0][0] != 0:
            logger.error(f"rsync to SageMaker instance failed: {return_codes[0][1]}")

    def _stop_instance(self, delete_configs=True):
        """Stop the SageMaker instance. Optionally remove its config from RNS"""
        import boto3

        sagemaker_client = boto3.client("sagemaker")
        resp = sagemaker_client.stop_training_job(TrainingJobName=self.job_name)
        if resp["ResponseMetadata"]["HTTPStatusCode"] == 200:
            if delete_configs:
                rns_client.delete_configs(resource=self)
                logger.info(f"Deleted SageMaker cluster {self.name} from config")
            logger.info(f"Successfully stopped instance {self.instance_id}")
        else:
            raise Exception(f"Failed to stop instance: {resp['ResponseMetadata']}")

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
            dest_path = f"~/{local_rh_package_path.name}"

            self._rsync(
                source=str(local_rh_package_path),
                dest=dest_path,
                up=True,
                contents=True,
            )
            # Default path to runhouse on the SageMaker instance
            rh_install_cmd = "pip install /root/runhouse/runhouse"
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

                    # TODO [JL] need a better way to check for errors
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
            # TODO [JL] make this nicer
            msg = (
                "Error occurred: {}\n\n"
                "Note: If you are experiencing connection errors, make sure you are using "
                "the latest version of paramiko and the AWS CLI:\npip install -U paramiko\n"
                "AWS CLI: https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html\n\n"
                "To check if the SSH tunnel with the SageMaker instance is still up:\n"
                "ssh -i ~/.ssh/sagemaker-ssh-gw -p 11022 root@localhost\n\n"
                "If the tunnel is not up, you can create a new one via the CLI by running:\n"
                f'sm-local-start-ssh "{self.instance_id}" -R localhost:12345:localhost:12345 '
                f"-L localhost:{self.DEFAULT_HTTP_PORT}:localhost:{self.DEFAULT_HTTP_PORT} "
                f"-L localhost:{self.DEFAULT_SSH_PORT}:localhost:22\n\n"
                f"Or in python with: ``my_cluster.ssh_tunnel()``"
            ).format(str(e))
            warnings.warn(msg)

        finally:
            # Remember to close the SSH connection when you're done
            ssh_client.close()

    def _logfile_path(self, logfile):
        return f"/root/.rh/{logfile}"

    def _reload_estimator_from_config(self, config: dict):
        estimator_framework = config.pop("framework", None)
        if estimator_framework is None:
            raise ValueError("Framework not specified in the config file.")

        if estimator_framework == "PyTorch":
            # Re-build the estimator object from the saved params
            return PyTorch(**config)
        else:
            raise NotImplementedError("Currently only PyTorch Estimator is supported.")

    def _get_instance_id(self):
        try:
            return self.ssh_wrapper.get_instance_ids()[0]
        except Exception as e:
            raise e

    def _create_mock_estimator(self):
        """Create an estimator required for launching the instance.
        **Note: this is only meant to serve as a mock estimator object, not used for an actual training job
        but solely for the purpose of spinning up the SageMaker instance.**"""
        estimator_dict = {
            "instance_count": 1,
            "framework_version": "1.9.1",
            "py_version": "py38",
            "role": self.arn_role,
            "entry_point": "launch_instance.py",
            "source_dir": os.path.abspath(
                os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    "../../scripts/sagemaker",
                )
            ),
            "instance_type": self.instance_type,
        }
        return PyTorch(**estimator_dict)

    def _add_ssh_config_entry(self, job_name):
        """Update the ~/.ssh/config to allow for accessing the cluster via: ssh <job name>"""
        config_file = os.path.expanduser("~/.ssh/config")
        identity_file = self.ssh_key_path

        # Create the new entry
        new_entry = f"""\n# Added by Runhouse for SageMaker SSH Support\nHost {job_name}\n  HostName {self.DEFAULT_HOST}\n  IdentityFile {identity_file}\n  Port {self.DEFAULT_SSH_PORT}\n  User {self.DEFAULT_USER}\n"""  # noqa

        # Append the new entry to the existing content
        with open(config_file, "a") as f:
            f.write(new_entry)

    def _add_ssh_helper_init_to_estimator(self):
        # https://github.com/aws-samples/sagemaker-ssh-helper#step-3-modify-your-training-script
        # TODO [JL] We should try to avoid this (adding to HTTP server init is not solving it)
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
