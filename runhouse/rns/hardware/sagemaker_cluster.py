import logging
import os
from typing import Dict, Union

import paramiko
import sagemaker
from sagemaker_ssh_helper.wrapper import SSHEstimatorWrapper

from runhouse.rns.hardware.cluster import Cluster

logger = logging.getLogger(__name__)


class SageMakerCluster(Cluster):
    DEFAULT_HOST = "localhost"
    DEFAULT_PORT = 11022
    DEFAULT_USER = "root"
    SSH_KEY_PATH = "~/.ssh/sagemaker-ssh-gw"
    CONNECTION_WAIT_TIME = 600  # seconds

    def __init__(
        self,
        name,
        arn_role: str,
        estimator: Union[sagemaker.estimator.EstimatorBase, Dict] = None,
        model: Union[sagemaker.model.Model, Dict] = None,
        connection_wait_time: int = None,
        dryrun=False,
        **kwargs,  # We have this here to ignore extra arguments when calling from from_config
    ):
        """
        The Runhouse SageMaker cluster abstraction. This is where you can run jobs (e.g. training, inference,
         batch transform), or access the underlying compute instance directly via SSH.

        .. note::
            To build a cluster, please use the factory method :func:`cluster`.
        """
        self.arn_role = arn_role
        self.instance_id = kwargs.get("instance_id")
        self.job_name = kwargs.get("job_name")
        self.job_type = kwargs.get("job_type")
        self.ssh_process = None
        self.connection_wait_time = self.set_connection_wait(connection_wait_time)

        if not dryrun and (estimator or model):
            if estimator:
                logger.info("Setting up SageMaker training job")
                self.job_type = estimator.JOB_CLASS_NAME
                if isinstance(estimator, dict):
                    # Convert to an estimator object to use for creating and running the job
                    estimator = self._reload_estimator_from_config(estimator)

                # Make sure the SSHEstimatorWrapper is being used by the estimator, this is necessary for
                # enabling the SSH tunnel to the SageMaker instance
                # https://github.com/aws-samples/sagemaker-ssh-helper/tree/main#step-2-modify-your-start-training-job-code
                dependency_dir = SSHEstimatorWrapper.dependency_dir()
                if dependency_dir not in estimator.dependencies:
                    estimator.dependencies.append(dependency_dir)

                # Create the SSH wrapper & run the job
                self.ssh_wrapper = SSHEstimatorWrapper.create(
                    estimator, connection_wait_time_seconds=self.connection_wait_time
                )

                # Call the SageMaker CreateTrainingJob API to start training
                estimator.fit(wait=False)
                self.job_name = self.ssh_wrapper.training_job_name()

            elif model:
                # TODO [JL]
                """
                self.job_type = model.JOB_CLASS_NAME

                if isinstance(model, dict):
                    model = self._reload_model_from_config(model)

                self.ssh_wrapper = SSHModelWrapper.create(model,
                                                          connection_wait_time_seconds=self.connection_wait_time)
                predictor = estimator.deploy(
                    initial_instance_count=1,
                    instance_type='ml.m5.xlarge',
                    endpoint_name="new_endpoint",
                    wait=True
                )
                """
                raise NotImplementedError("Inference jobs are not yet supported.")

            self._add_ssh_config_entry(self.job_name)

            logger.info("Launching new job..")
            self.instance_id = self.ssh_wrapper.get_instance_ids()[0]
            logger.info(f"Successfully created a new job with name: {self.job_name}")

            # https://github.com/aws-samples/sagemaker-ssh-helper/tree/main#forwarding-tcp-ports-over-ssh-tunnel
            logger.info(
                "To create an active SSH tunnel with the SageMaker instance via the CLI:\n"
                f'sm-local-start-ssh "{self.instance_id}" -R localhost:12345:localhost:12345 '
                "-L localhost:11022:localhost:22"
            )

            # TODO [JL] create the tunnel in a separate process instead of having user manage it themselves?
            # # Start the tunnel in a separate process to maintain the connection to the instance
            # ssh_process = multiprocessing.Process(target=self.create_ssh_tunnel)
            # ssh_process.start()
            # # save the process object to be able to kill it later
            # self.ssh_process = ssh_process
            # logger.info(
            #     "To kill the connection:\n my_cluster.stop_ssh_tunnel()"
            # )

            # Once ports are open and tunnel is created, can connect using the ``ssh <host>`` command
            logger.info(
                "Once the connection has been created, in a separate terminal session you can SSH into "
                f"the instance using:\nssh {self.job_name}"
            )

        self.estimator = estimator

        super().__init__(name=name, ips=None, ssh_creds=None, dryrun=dryrun)

    @property
    def config_for_rns(self):
        config = super().config_for_rns
        config.update(
            {
                "instance_id": self.instance_id,
                "arn_role": self.arn_role,
                "job_name": self.job_name,
                "job_type": self.job_type,
            }
        )
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
                attr: getattr(self.estimator, attr, None) for attr in estimator_attrs
            }
            # mxnet, tensorflow, keras, pytorch, onnx, xgboost
            config.update({"estimator": selected_attrs})
            config["estimator"]["framework"] = "PyTorch"
        else:
            raise NotImplementedError("Currently only PyTorch Estimator is supported.")

        return config

    @property
    def ssh_key_path(self):
        return os.path.expanduser(self.SSH_KEY_PATH)

    def set_connection_wait(self, connection_wait_time: int = None) -> int:
        """Amount of time the SSH helper will wait inside SageMaker before it continues normal execution"""
        if connection_wait_time:
            return connection_wait_time

        if self.job_type == "training-job":
            # Allow for connecting to the instance before training starts
            return self.CONNECTION_WAIT_TIME

        # For inference and others, always up and running
        return 0

    def stop_ssh_tunnel(self):
        """Kills the open connection with the SageMaker instance."""
        if self.ssh_process is not None:
            self.ssh_process.terminate()
            self.ssh_process.join()
            self.ssh_process = None

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
            ssh_client.load_system_host_keys()

            # Connect to the instance
            ssh_client.connect(
                hostname=self.DEFAULT_HOST,
                port=self.DEFAULT_PORT,
                username=self.DEFAULT_USER,
                key_filename=self.ssh_key_path,
                allow_agent=False,
                look_for_keys=False,
            )
            for command in commands:
                stdin, stdout, stderr = ssh_client.exec_command(command)
                stdout = stdout.read().decode()
                stderr = stderr.read().decode()

                return_code = 0 if not stderr else 1
                return_codes.append((return_code, stdout, stderr))

            return return_codes

        except (
            paramiko.AuthenticationException,
            paramiko.SSHException,
            Exception,
        ) as e:
            msg = (
                "Error occurred: {}\n\n"
                "Note: If you are experiencing connection errors, please make sure you are using "
                "the latest version of paramiko:\n pip install -U paramiko\n"
                "Also confirm the SSH tunnel with the instance is still up by running:\n"
                "ssh -i ~/.ssh/sagemaker-ssh-gw -p 11022 root@localhost\n\n"
                "If the tunnel is not up, you can re-connect by running:\n"
                f'sm-local-start-ssh "{self.instance_id}" -R localhost:12345:localhost:12345 '
                f"-L localhost:11022:localhost:22"
            ).format(str(e))

            raise ValueError(msg)

        finally:
            # Remember to close the SSH connection when you're done
            ssh_client.close()

    def _reload_estimator_from_config(self, config: dict):
        estimator_framework = config.pop("framework", None)
        if estimator_framework is None:
            raise ValueError("No framework saved for estimator")

        if estimator_framework == "PyTorch":
            from sagemaker.pytorch import PyTorch

            # Re-build the estimator object from the saved params
            return PyTorch(**config)

        raise NotImplementedError("Currently only PyTorch Estimator is supported.")

    def create_ssh_tunnel(self):
        import subprocess
        import time

        command = (
            f'sm-local-start-ssh "{self.instance_id}" -R localhost:12345:localhost:12345 '
            "-L localhost:11022:localhost:22"
        )

        # Launch the process in the background
        process = subprocess.Popen(command, shell=True)

        # Sleep indefinitely to keep the process running in the background
        while True:
            try:
                time.sleep(1)
            except KeyboardInterrupt:
                # If you want to stop the process with Ctrl+C
                process.terminate()
                break

    def _add_ssh_config_entry(
        self,
        job_name,
        hostname="localhost",
        identity_file=None,
        port=11022,
        user="root",
    ):
        """Update the ~/.ssh/config to allow for ``ssh <host>"""
        config_file = os.path.expanduser("~/.ssh/config")
        identity_file = (
            os.path.expanduser(identity_file) if identity_file else self.ssh_key_path
        )

        # Create the new entry
        new_entry = f"""\n# Added by Runhouse for SageMaker SSH Support\nHost {job_name}\n  HostName {hostname}\n  IdentityFile {identity_file}\n  Port {port}\n  User {user}\n"""  # noqa

        # Append the new entry to the existing content
        with open(config_file, "a") as f:
            f.write(new_entry)
