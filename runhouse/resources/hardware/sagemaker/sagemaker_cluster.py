import configparser
import contextlib
import getpass
import importlib
import logging
import os
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
import warnings
from pathlib import Path
from typing import Dict, Optional, Union

try:
    import boto3
    import paramiko
    import sagemaker
    from sagemaker.estimator import EstimatorBase
    from sagemaker.mxnet import MXNet
    from sagemaker.pytorch import PyTorch
    from sagemaker.tensorflow import TensorFlow
    from sagemaker.xgboost import XGBoost
except ImportError:
    pass

from runhouse.constants import LOCAL_HOSTS

from runhouse.globals import configs, rns_client
from runhouse.resources.hardware.cluster import Cluster
from runhouse.resources.hardware.utils import ServerConnectionType
from runhouse.rns.utils.api import is_jsonable, relative_ssh_path, resolve_absolute_path
from runhouse.rns.utils.names import _generate_default_name

logger = logging.getLogger(__name__)

####################################################################################################
# Caching mechanisms for SSHTunnelForwarder
####################################################################################################

ssh_tunnel_cache = {}


def get_open_ssh_tunnel(address: str, ssh_port: int) -> Optional["SSHTunnelForwarder"]:
    if (address, ssh_port) in ssh_tunnel_cache:

        ssh_tunnel = ssh_tunnel_cache[(address, ssh_port)]
        # Initializes tunnel_is_up dictionary
        ssh_tunnel.check_tunnels()

        if (
            ssh_tunnel.is_active
            and ssh_tunnel.tunnel_is_up[ssh_tunnel.local_bind_address]
        ):
            return ssh_tunnel

        else:
            # If the tunnel is no longer active or up, pop it from the global cache
            ssh_tunnel_cache.pop((address, ssh_port))
    else:
        return None


def cache_open_ssh_tunnel(
    address: str,
    ssh_port: str,
    ssh_tunnel: "SSHTunnelForwarder",
):
    ssh_tunnel_cache[(address, ssh_port)] = ssh_tunnel


class SageMakerCluster(Cluster):
    DEFAULT_SERVER_HOST = "localhost"
    DEFAULT_INSTANCE_TYPE = "ml.m5.large"
    DEFAULT_REGION = "us-east-1"
    DEFAULT_USER = "root"
    # https://github.com/aws/deep-learning-containers/blob/master/available_images.md
    BASE_ECR_URL = "763104351884.dkr.ecr.us-east-1.amazonaws.com"

    # Default path for any estimator source code copied onto the cluster
    ESTIMATOR_SRC_CODE_PATH = "/opt/ml/code"
    ESTIMATOR_LOG_FILE = "sm_cluster.out"
    SSH_KEY_FILE_NAME = "sagemaker-ssh-gw"

    DEFAULT_SSH_PORT = 11022
    DEFAULT_CONNECTION_WAIT_TIME = 60  # seconds

    def __init__(
        self,
        name: str,
        role: str = None,
        profile: str = None,
        region: str = None,
        ssh_key_path: str = None,
        instance_id: str = None,
        instance_type: str = None,
        num_instances: int = None,
        image_uri: str = None,
        autostop_mins: int = None,
        connection_wait_time: int = None,
        estimator: Union["EstimatorBase", Dict] = None,
        job_name: str = None,
        server_host: str = None,
        server_port: int = None,
        domain: str = None,
        server_connection_type: str = None,
        ssl_keyfile: str = None,
        ssl_certfile: str = None,
        den_auth: bool = False,
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
        super().__init__(
            name=name,
            ssh_creds=kwargs.pop("ssh_creds", {}),
            ssh_port=kwargs.pop("ssh_port", self.DEFAULT_SSH_PORT),
            server_host=server_host,
            server_port=server_port,
            server_connection_type=server_connection_type,
            ssl_certfile=ssl_certfile,
            ssl_keyfile=ssl_keyfile,
            domain=domain,
            den_auth=den_auth,
            dryrun=dryrun,
            **kwargs,
        )
        self._connection_wait_time = connection_wait_time
        self._instance_type = instance_type
        self._num_instances = num_instances
        self._ssh_key_path = ssh_key_path

        # SSHEstimatorWrapper to facilitate the SSH connection to the cluster
        self._ssh_wrapper = None

        # Note: Relevant only if an estimator is explicitly provided
        self._estimator_entry_point = kwargs.get("estimator_entry_point")
        self._estimator_source_dir = kwargs.get("estimator_source_dir")
        self._estimator_framework = kwargs.get("estimator_framework")

        self.job_name = job_name

        # Set initial region - may be overwritten depending on the profile used
        self.region = region or self.DEFAULT_REGION

        # Set a default sessions initially - may overwrite depending on the profile loaded below
        self._set_boto_session()
        self._set_sagemaker_session()

        # Either use the user-provided instance_id, or look it up from the job_name
        self.instance_id = instance_id or (
            self._cluster_instance_id() if self.job_name else None
        )

        self._autostop_mins = (
            autostop_mins
            if autostop_mins is not None
            else configs.get("default_autostop")
        )

        self.estimator = self._load_estimator(estimator)

        self.role, self.profile = (
            self._load_role_and_profile(role, profile)
            if not dryrun
            else (role, profile)
        )
        logger.info(
            f"Using SageMaker execution role: `{self.role}` and profile: `{self.profile}`"
        )

        self.image_uri = self._load_image_uri(image_uri)

        # Note: Setting instance ID as cluster IP for compatibility with Cluster parent class methods
        self.address = self.instance_id

    def config(self, condensed=True):
        config = super().config(condensed)
        config.update(
            {
                "instance_id": self.instance_id,
                "role": self.role,
                "region": self.region,
                "profile": self.profile,
                "ssh_key_path": self.ssh_key_path,
                "job_name": self.job_name,
                "instance_type": self.instance_type,
                "num_instances": self.num_instances,
                "image_uri": self.image_uri,
                "autostop_mins": self._autostop_mins,
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
                    "estimator_source_dir": str(self._estimator_source_dir),
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
        return Path("~/.ssh/known_hosts").expanduser()

    @property
    def ssh_config_file(self):
        return Path("~/.ssh/config").expanduser()

    @property
    def ssh_key_path(self):
        """Relative path to the private SSH key used to connect to the cluster."""
        if self._ssh_key_path:
            return relative_ssh_path(self._ssh_key_path)

        # Default relative path
        return f"~/.ssh/{self.SSH_KEY_FILE_NAME}"

    @ssh_key_path.setter
    def ssh_key_path(self, ssh_key_path):
        self._ssh_key_path = ssh_key_path

    @property
    def num_instances(self):
        if self._num_instances:
            return self._num_instances
        elif self.estimator:
            return self.estimator.instance_count
        else:
            return 1

    @num_instances.setter
    def num_instances(self, num_instances):
        self._num_instances = num_instances

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
    def default_bucket(self):
        """Default bucket to use for storing the cluster's authorized public keys."""
        return self._sagemaker_session.default_bucket()

    @property
    def _use_https(self) -> bool:
        # Note: Since always connecting via SSM no need for HTTPS
        return False

    @property
    def _extra_ssh_args(self):
        """Extra SSH arguments to be used when connecting to the cluster."""
        # Note - port 12345 can be used for Python Debug Server: "-R localhost:12345:localhost:12345"
        # https://github.com/aws-samples/sagemaker-ssh-helper#remote-debugging-with-pycharm-debug-server-over-ssh
        return f"-L localhost:{self.ssh_port}:localhost:22"

    @property
    def _s3_keys_path(self):
        """Path to public key stored for the cluster on S3. When initializing the cluster, the public key
        is copied by default to an authorized keys file in this location."""
        return f"s3://{self.default_bucket}/ssh-authorized-keys/"

    @property
    def _ssh_public_key_path(self):
        return f"{self._abs_ssh_key_path}.pub"

    @property
    def _abs_ssh_key_path(self):
        return resolve_absolute_path(self.ssh_key_path)

    @property
    def _ssh_key_comment(self):
        """Username and hostname to be used as the comment for the public key."""
        return f"{getpass.getuser()}@{socket.gethostname()}"

    @property
    def _s3_client(self):
        if self._boto_session is None:
            self._set_boto_session()

        return self._boto_session.client("s3")

    def _get_env_activate_cmd(self, env=None):
        """Prefix for commands run on the cluster. Ensure we are running all commands in the conda environment
        and not the system default python."""
        # TODO [JL] Can SageMaker handle this for us?
        if env:
            from runhouse.resources.envs import _get_env_from

            return _get_env_from(env)._activate_cmd
        return "source /opt/conda/bin/activate"

    def _set_boto_session(self, profile_name: str = None):
        self._boto_session = boto3.Session(
            region_name=self.region, profile_name=profile_name
        )

    def _set_sagemaker_session(self):
        """Create a SageMaker session required for using the SageMaker APIs."""
        self._sagemaker_session = sagemaker.Session(boto_session=self._boto_session)

    def set_connection_defaults(self):
        if (
            "aws-cli/2."
            not in subprocess.run(
                ["aws", "--version"], capture_output=True, text=True
            ).stdout
        ):
            raise RuntimeError(
                "SageMaker SDK requires AWS CLI v2. You may also need to run `pip uninstall awscli` to ensure "
                "the right version is being used. For more info: https://www.run.house/docs/api/python/cluster#id9"
            )

        if self.ssh_key_path:
            self.ssh_key_path = relative_ssh_path(self.ssh_key_path)
        else:
            self.ssh_key_path = None

        if (
            self.server_connection_type is not None
            and self.server_connection_type != ServerConnectionType.AWS_SSM
        ):
            raise ValueError(
                "SageMaker Cluster currently requires a server connection type of `aws_ssm`."
            )
        self.server_connection_type = ServerConnectionType.AWS_SSM.value

        if self.server_host and self.server_host not in LOCAL_HOSTS:
            raise ValueError(
                "SageMaker Cluster currently requires a server host of `localhost` or `127.0.0.1`"
            )

    # -------------------------------------------------------
    # Cluster State & Lifecycle Methods
    # -------------------------------------------------------
    def restart_server(
        self,
        _rh_install_url: str = None,
        resync_rh: bool = True,
        restart_ray: bool = True,
        env: Union[str, "Env"] = None,
        restart_proxy: bool = False,
    ):
        """Restart the RPC server on the SageMaker instance.

        Args:
            resync_rh (bool): Whether to resync runhouse. (Default: ``True``)
            restart_ray (bool): Whether to restart Ray. (Default: ``True``)
            env (str or Env): Env to restart the server from. If not provided
                will use default env on the cluster.
            restart_proxy (bool): Whether to restart nginx on the cluster, if configured. (Default: ``False``)
        Example:
            >>> rh.sagemaker_cluster("sagemaker-cluster").restart_server()
        """
        return super().restart_server(
            _rh_install_url, resync_rh, restart_ray, env, restart_proxy
        )

    def check_server(self, restart_server=True):
        if self.on_this_cluster():
            return

        if not self.instance_id or not self.is_up():
            logger.info(f"Cluster {self.name} is not up, bringing it up now.")
            self.up_if_not()

        if not self.client:
            try:
                self.connect_server_client()
                logger.info(
                    f"Checking server {self.name} with instance ID: {self.instance_id}"
                )

                self.client.check_server()
                logger.info(f"Server {self.instance_id} is up.")
            except:
                if restart_server:
                    logger.info(
                        f"Server {self.instance_id} is up, but the API server may not be up."
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
                        env=None,
                    )
                    logger.info(f"Checking server {self.instance_id} again.")

                    self.client.check_server()
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

        return self

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
            resp: dict = self.status()
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

        return self

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
        self._update_autostop(self._autostop_mins)

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
    ) -> "SSHTunnelForwarder":
        from sshtunnel import BaseSSHTunnelForwarderError, SSHTunnelForwarder

        tunnel = get_open_ssh_tunnel(self.address, self.ssh_port)
        if tunnel and tunnel.local_bind_port == local_port:
            logger.info(
                f"SSH tunnel on ports {local_port, remote_port} already created with the cluster"
            )
            return tunnel

        try:
            remote_bind_addresses = ("127.0.0.1", local_port)
            local_bind_addresses = ("", local_port)

            ssh_tunnel = SSHTunnelForwarder(
                self.DEFAULT_SERVER_HOST,
                ssh_username=self.DEFAULT_USER,
                ssh_pkey=self._abs_ssh_key_path,
                ssh_port=self.ssh_port,
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
                raise BaseSSHTunnelForwarderError(
                    f"{e} Make sure ports {self.server_port} and {self.ssh_port} are "
                    f"not already in use."
                )

            # Refresh the SSM session, which should bind the HTTP and SSH ports to localhost which are forwarded
            # to the cluster
            self._refresh_ssm_session_with_cluster(num_ports_to_try)

            # Retry creating the SSH tunnel once the session has been refreshed
            return self.ssh_tunnel(
                local_port, remote_port, num_ports_to_try, retry=False
            )

        cache_open_ssh_tunnel(self.address, self.ssh_port, ssh_tunnel)
        return ssh_tunnel

    def ssh(self, interactive: bool = True):
        """SSH into the cluster.

        Args:
            interactive (bool): Whether to start an interactive shell or not (Default: ``True``).

        Example:
            >>> rh.sagemaker_cluster(name="sagemaker-cluster").ssh()
        """

        if (self.address, self.ssh_port) not in ssh_tunnel_cache:
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
        node: str = None,
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
                # Host can be replaced with name (as reflected in the ~/.ssh/config file)
                from runhouse.resources.hardware.sky_ssh_runner import SkySSHRunner

                runner = SkySSHRunner(
                    self.name,
                    port=self.ssh_port,
                    ssh_user=self.DEFAULT_USER,
                    ssh_private_key=self._abs_ssh_key_path,
                    ssh_control_name=f"{self.name}:{self.ssh_port}",
                )
                command = f"{cmd_prefix} {command}" if cmd_prefix else command
                logger.debug(f"Running command on {self.name}: {command}")
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
        full_module_name = f"resources/hardware/sagemaker/{default_entry_point}"

        entry_point_path = self._get_path_for_module(full_module_name)
        source_dir_path = os.path.dirname(entry_point_path)

        # Set default_entry_point and default_source_dir
        default_source_dir = source_dir_path

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
                "instance_count": self.num_instances,
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
            f"Launching a new SageMaker cluster (num instances={self.num_instances}) on instance "
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

        connected = False
        while not connected:
            command = f"{base_command} {self._extra_ssh_args}"

            try:
                if num_ports_to_try == 0:
                    raise ConnectionError(
                        f"Failed to create SSM session and connect to {self.name} after repeated attempts."
                        f"Make sure SSH keys exist in local path: {self._abs_ssh_key_path}"
                    )

                logger.debug(f"Running command: {command}")

                # Define an event to signal completion of the SSH tunnel setup
                tunnel_setup_complete = threading.Event()

                # Manually allocate a pseudo-terminal to prevent a "pseudo-terminal not allocated" error
                head_fd, worker_fd = pty.openpty()

                def run_ssm_session_cmd():
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

                tunnel_thread = threading.Thread(target=run_ssm_session_cmd)
                tunnel_thread.daemon = True  # Set the thread as a daemon, so it won't block the main thread

                # Start the SSH tunnel thread
                tunnel_thread.start()

                # Give time for the SSM session to start, SSH keys to be copied onto the cluster, and the SSH port
                # forwarding command to run
                tunnel_setup_complete.wait(timeout=30)

                if not self._ports_are_in_use():
                    # Command should bind SSH port and HTTP port on localhost, if this is not the case try re-running
                    # the bash script with a different set of ports
                    # E.g. ❯ lsof -i:11022,32300
                    # COMMAND   PID   USER   FD   TYPE             DEVICE SIZE/OFF NODE NAME
                    # ssh     97115 myuser    3u  IPv4 0xcf81f230786cc9fd      0t0  TCP localhost:32300 (LISTEN)
                    # ssh     97115 myuser    6u  IPv4 0xcf81f230786eff6d      0t0  TCP localhost:11022 (LISTEN)
                    raise ConnectionError

                # Update the SSH config for the cluster with the connected SSH port
                self._add_or_update_ssh_config_entry()

                connected = True

                logger.info(
                    f"Created SSM session using ports {self.ssh_port} and {self.server_port}. "
                    f"All active sessions can be viewed with: ``aws ssm describe-sessions --state Active``"
                )

            except (ConnectionError, subprocess.CalledProcessError):
                # Try re-running with updated the ports - possible the ports are already in use
                self.server_port += 1
                self.ssh_port += 1
                num_ports_to_try -= 1
                pass

    def _start_instance(self):
        """Call the SageMaker CreateTrainingJob API to start the training job on the cluster."""
        # TODO [JL] Note: Keeping private until re-running training jobs on the same cluster is supported
        if not self.estimator:
            logger.warning("No estimator found, cannot run job.")
            return

        # NOTE: underscores not allowed for training job name - must match: ^[a-zA-Z0-9](-*[a-zA-Z0-9]){0,62}
        self.estimator.fit(
            wait=False,
            job_name=self.job_name or _generate_default_name(self.name, sep="-"),
        )

    def _load_base_command_for_ssm_session(self) -> str:
        """Bash script command for creating the SSM session and uploading the SSH keys to the cluster. Will try
        reusing existing keys locally if they exist, otherwise will generate new ones locally and copy them to s3."""
        private_key_path = self._abs_ssh_key_path
        public_key_path = self._ssh_public_key_path

        resource_name = "resources/hardware/sagemaker/start-ssm-proxy-connection.sh"
        script_path = self._get_path_for_module(resource_name)

        os.chmod(script_path, 0o755)

        s3_key_path = self._s3_keys_path

        # bash script which creates an SSM session with the cluster
        base_command = (
            f'bash {script_path} "{self.instance_id}" '
            f'"{s3_key_path}" "{private_key_path}" "{self.region}"'
        )

        bucket = s3_key_path.split("/")[2]
        key = "/".join(s3_key_path.split("/")[3:])

        if Path(public_key_path).exists() and Path(private_key_path).exists():
            # If the key pair exists locally, make sure a matching public key also exists in s3
            with open(self._ssh_public_key_path, "r") as f:
                public_key = f.read()

            self._add_public_key_to_authorized_keys(bucket, key, public_key)

            return base_command

        # If no private + public keys exists generate a new key pair from scratch
        logger.warning(
            f"No private + public keypair found in local path: {private_key_path}. Generating a new key pair "
            "locally and uploading the new public key to s3"
        )
        self._create_new_ssh_key_pair(bucket, key)

        return base_command

    def _refresh_ssm_session_with_cluster(self, num_ports_to_try: int = 5):
        """Reconnect to the cluster via the AWS SSM. This bypasses the step of creating a new SSH key which was already
        done when upping the cluster. Note: this assumes the session has previously been created, which we do when
        the cluster has been upped.

        To view all sessions: ``aws ssm describe-sessions --state Active``
        """
        ssh_key_path = self._abs_ssh_key_path
        public_key_path = self._ssh_public_key_path

        if not Path(ssh_key_path).exists() and not Path(public_key_path).exists():
            logger.warning(
                f"SSH key pairs not found in paths: {ssh_key_path} and {public_key_path}"
            )
            self._create_ssm_session_with_cluster()

        # https://github.com/aws-samples/sagemaker-ssh-helper/blob/main/sagemaker_ssh_helper/sm-connect-ssh-proxy
        full_module_name = "resources/hardware/sagemaker/refresh-ssm-session.sh"
        script_path = self._get_path_for_module(full_module_name)

        os.chmod(script_path, 0o755)

        # Remove stale entries from the known hosts file - this is import for avoiding collisions when
        # subsequent clusters are created as the IP address is added to the file as: [localhost]:11022
        self._filter_known_hosts()

        num_attempts = num_ports_to_try
        connected = False

        while not connected:
            command = [
                script_path,
                self.instance_id,
                ssh_key_path,
                self.region,
            ] + shlex.split(self._extra_ssh_args)

            if num_ports_to_try == 0:
                raise ConnectionError(
                    f"Failed to create connection with {self.name} after {num_attempts} attempts "
                    f"(cluster status=`{self.status().get('TrainingJobStatus')}`). Make sure that another SageMaker "
                    f"cluster is not already active, that AWS CLI V2 is installed, and that the path has been properly "
                    f"added to your bash profile"
                    f"(https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-troubleshooting.html). "
                    f"If the error persists, try running the command to create the session "
                    f"manually: `bash {' '.join(command)}`"
                )

            try:
                subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    stdin=subprocess.PIPE,
                )

                # Give enough time for the aws ssm + ssh port forwarding commands in the script to complete
                # Better to wait a few more seconds than to restart the HTTP server on the cluster unnecessarily
                time.sleep(8)

                if not self._ports_are_in_use():
                    # Command should bind SSH port and HTTP port on localhost, if this is not the case try re-running
                    # with different ports
                    raise socket.error

                connected = True
                logger.info("Successfully refreshed SSM session")

            except socket.error:
                # If the refresh didn't work try connecting with a different port - could be that the port
                # is already taken
                self.server_port += 1
                self.ssh_port += 1
                num_ports_to_try -= 1
                pass

    # -------------------------------------------------------
    # SSH Keys Management
    # -------------------------------------------------------
    def _create_new_ssh_key_pair(self, bucket, key):
        """Create a new private / public key pairing needed for SSHing into the cluster."""
        private_key_path = self._abs_ssh_key_path
        ssh_key = paramiko.RSAKey.generate(bits=2048)

        ssh_key.write_private_key_file(private_key_path)
        os.chmod(private_key_path, 0o600)

        # Set the comment for the public key to include username and hostname
        comment = self._ssh_key_comment
        public_key = f"ssh-rsa {ssh_key.get_base64()} {comment}"

        # Update the public key locally
        self._write_public_key(public_key)

        # Update the public key in s3
        self._add_public_key_to_authorized_keys(bucket, key, public_key)

    def _add_public_key_to_authorized_keys(
        self, bucket: str, key: str, public_key: str
    ) -> None:
        """Add the public key to the authorized keys file stored in S3. This file will get copied onto the cluster's
        authorized keys file."""
        path_to_auth_keys = self._path_to_auth_keys(key)
        authorized_keys: str = self._load_authorized_keys(bucket, path_to_auth_keys)

        if not authorized_keys:
            # Create a new authorized keys file
            logger.info(
                f"No authorized keys file found in s3 path: {self._s3_keys_path}. Creating and uploading a new file."
            )
            self._upload_key_to_s3(bucket, path_to_auth_keys, public_key)
            return

        if public_key not in authorized_keys:
            # Add the public key to the existing authorized keys saved in s3
            authorized_keys += f"\n{public_key}\n"
            logger.info(
                f"Adding public key to authorized keys file saved for the cluster "
                f"in path: {self._s3_keys_path}"
            )
            self._upload_key_to_s3(bucket, path_to_auth_keys, authorized_keys)

    def _path_to_auth_keys(self, key):
        """Path to the authorized keys file stored in the s3 bucket for the role ARN associated with the cluster."""
        return key + f"{self.SSH_KEY_FILE_NAME}.pub"

    def _write_public_key(self, public_key: str):
        """Update the public key stored locally."""
        with open(self._ssh_public_key_path, "w") as f:
            f.write(public_key)

    def _upload_key_to_s3(self, bucket, key, body):
        """Save a public key to the authorized file in the default bucket for given SageMaker role."""
        self._s3_client.put_object(
            Bucket=bucket,
            Key=key,
            Body=body,
        )

    def _load_authorized_keys(self, bucket, auth_keys_file) -> Union[str, None]:
        """Load the authorized keys file for this AWS role stored in S3. If no file exists, return None."""
        try:
            response = self._s3_client.get_object(Bucket=bucket, Key=auth_keys_file)
            existing_pub_keys = response["Body"].read().decode("utf-8")
            return existing_pub_keys

        except self._s3_client.exceptions.NoSuchKey:
            # No authorized keys file exists in s3 for this role
            return None

        except Exception as e:
            raise e

    # -------------------------------------------------------
    # Cluster Helpers
    # -------------------------------------------------------
    def _rsync(self, source: str, dest: str, up: bool, contents: bool = False):
        source = source + "/" if not source.endswith("/") else source
        dest = dest + "/" if not dest.endswith("/") else dest

        command = (
            f"rsync -rvh --exclude='.git' --exclude='venv*/' --exclude='dist/' --exclude='docs/' "
            f"--exclude='__pycache__/' --exclude='.*' "
            f"--include='.rh/' -e 'ssh -o StrictHostKeyChecking=no "
            f"-i {self._abs_ssh_key_path} -p {self.ssh_port}' {source} root@localhost:{dest}"
        )

        logger.info(f"Syncing {source} to: {dest} on cluster")
        return_codes = self.run([command])
        if return_codes[0][0] != 0:
            logger.error(f"rsync to SageMaker cluster failed: {return_codes[0][1]}")

    def _cluster_instance_id(self):
        """Get the instance ID of the cluster. This is the ID of the instance running the training job generated
        by SageMaker."""
        if self._ssh_wrapper:
            # This is a hack to effectively do list.get(0, None)
            return next(iter(self._ssh_wrapper.get_instance_ids()), None)

        from sagemaker_ssh_helper.manager import SSMManager

        ssm_manager = SSMManager(region_name=self.region)
        ssm_manager.redo_attempts = 0
        instance_ids = ssm_manager.get_training_instance_ids(self.job_name)
        # This is a hack to effectively do list.get(0, None)
        return next(iter(instance_ids), None)

    def _get_path_for_module(self, resource_name: str) -> str:
        import importlib.resources as pkg_resources

        package_name = "runhouse"
        script_path = str(pkg_resources.files(package_name) / resource_name)
        return script_path

    def _load_role_and_profile(self, role, profile):
        """Load the SageMaker role and profile used for launching and connecting to the cluster.
        Role can be provided as an ARN or a name. If provided, will search for the profile containing this role
        in local AWS configs. If no profile is provided, try loading from the environment variable ``AWS_PROFILE``,
        otherwise default to using the ``default`` profile."""
        if self.estimator:
            # If using an estimator must provide a name or full ARN
            # https://sagemaker.readthedocs.io/en/stable/api/training/estimators.html
            role = self.estimator.role

        if role and role.startswith("arn:aws"):
            profile = profile or self._load_profile_and_region_for_role(role)

        else:
            # https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-roles.html
            profile = profile or os.environ.get("AWS_PROFILE")
            if profile is None:
                profile = "default"
                logger.warning(
                    f"No profile provided or environment variable set to `AWS_PROFILE`, using the {profile} profile"
                )

        try:
            # Update the sessions using the profile provided
            self._set_boto_session(profile_name=profile)
            self._set_sagemaker_session()

            # If no role explicitly provided, use sagemaker to get it via the profile
            role = role or sagemaker.get_execution_role(
                sagemaker_session=self._sagemaker_session
            )
        except Exception as e:
            if self.on_this_cluster():
                # If we're on the cluster, we may not have the profile or role saved locally, but should still be able
                # to create the cluster object (e.g. for rh.here).
                pass
            else:
                raise e

        return role, profile

    def _load_profile_and_region_for_role(self, role_arn: str) -> Union[str, None]:
        """Find the profile (and region) associated with a particular role ARN. If no profile is found, return None."""
        try:
            aws_dir = os.path.expanduser("~/.aws")
            credentials_path = os.path.join(aws_dir, "credentials")
            config_path = os.path.join(aws_dir, "config")

            profiles_with_role = set()

            for path in [credentials_path, config_path]:
                config = configparser.ConfigParser()
                config.read(path)

                for section in config.sections():
                    config_section = config[section]
                    config_role_arn = config_section.get("role_arn")
                    if config_role_arn == role_arn:
                        # Add just the name of the profile (not the full section heading)
                        profiles_with_role.add(section.split(" ")[-1])

                    # Update the region to use the one associated with this profile
                    profile_region = config_section.get("region")
                    if profile_region != self.region:
                        warnings.warn(
                            f"Updating region based on AWS config to: {profile_region}"
                        )
                        self.region = profile_region

            if not profiles_with_role:
                return None

            profiles = list(profiles_with_role)
            profile = profiles[0]

            if len(profiles) > 1:
                logger.warning(
                    f"Found multiple profiles associated with the same role. Using the first "
                    f"one ({profile})"
                )

            return profile

        except Exception as e:
            logger.warning(f"Could not find a profile for role {role_arn}: {e}")
            return None

    def _load_image_uri(self, image_uri: str = None) -> str:
        """Load the docker image URI used for launching the SageMaker instance. If no image URI is provided, use
        a default image based on the instance type."""
        if image_uri:
            return image_uri

        if self.estimator:
            return self.estimator.image_uri

        return self._base_image_uri()

    def _stop_instance(self, delete_configs=True):
        """Stop the SageMaker instance. Optionally remove its config from RNS."""
        self._sagemaker_session.stop_training_job(job_name=self.job_name)

        if self.is_up():
            raise Exception(f"Failed to stop instance {self.name}")

        logger.info(f"Successfully stopped instance {self.name}")

        # Remove stale host key(s) from known hosts
        self._filter_known_hosts()

        if delete_configs:
            # Delete from RNS
            rns_client.delete_configs(resource=self)
            logger.info(f"Deleted {self.name} from configs")

    def _sync_runhouse_to_cluster(self, node: str = None, _install_url=None, env=None):
        if not self.instance_id:
            raise ValueError(f"No instance ID set for cluster {self.name}. Is it up?")

        if not self.client:
            self.connect_server_client()

        # Sync the local ~/.rh directory to the cluster
        self._rsync(
            source=str(Path("~/.rh").expanduser()),
            dest="~/.rh",
            up=True,
            contents=True,
        )
        logger.info("Synced ~/.rh folder to the cluster")

        local_rh_package_path = Path(importlib.util.find_spec("runhouse").origin).parent
        # local_rh_package_path = Path(pkgutil.get_loader("runhouse").path).parent

        # **Note** temp patch to handle PyYAML errors: https://github.com/yaml/pyyaml/issues/724
        base_rh_install_cmd = f'{self._get_env_activate_cmd(env=None)} && python3 -m pip install "cython<3.0.0"'

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
        if "sagemaker_session" not in estimator:
            # Estimator requires an initialized sagemaker session
            # https://stackoverflow.com/questions/55869651/how-to-fix-aws-region-error-valueerror-must-setup-local-aws-configuration-with
            estimator["sagemaker_session"] = self._sagemaker_session

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
            path=Path(self._estimator_source_dir).expanduser()
        ).to(self, path=self.ESTIMATOR_SRC_CODE_PATH)
        logger.info(
            f"Synced estimator source directory to the cluster in path: {estimator_folder.path}"
        )

    def _base_image_uri(self):
        """Pick a default image for the cluster based on its instance type"""
        # TODO [JL] Add flexibility for py version & framework_version
        gpu_instance_types = ["p2", "p3", "p4", "g3", "g4", "g5"]

        image_type = (
            "gpu"
            if any(prefix in self.instance_type for prefix in gpu_instance_types)
            else "cpu"
        )

        cuda_version = "cu118-" if image_type == "gpu" else ""
        image_url = (
            f"{self.BASE_ECR_URL}/pytorch-training:2.0.1-{image_type}-py310-{cuda_version}"
            f"ubuntu20.04-sagemaker"
        )

        return image_url

    def _update_autostop(self, autostop_mins: int = None):
        cluster_config = self.config()
        cluster_config["autostop_mins"] = autostop_mins or -1
        if not self.client:
            self.connect_server_client()
        # Update the config on the server with the new autostop time
        self.client.check_server()

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
            s1.bind(("localhost", self.ssh_port))
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s2:
            s2.bind(("localhost", self.server_port))

    # -------------------------------------------------------
    # SSH config
    # -------------------------------------------------------
    def _filter_known_hosts(self):
        """To prevent host key collisions in the ~/.ssh/known_hosts file, remove any stale entries of localhost
        using the SSH port."""
        known_hosts = self.hosts_path
        if not known_hosts.exists():
            # e.g. in a collab or notebook environment
            return

        valid_hosts = []
        with open(known_hosts, "r") as f:
            for line in f:
                if not line.strip().startswith(
                    f"[{self.DEFAULT_SERVER_HOST}]:{self.ssh_port}"
                ):
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
              HostName {hostname or self.DEFAULT_SERVER_HOST}
              IdentityFile {identity_file or self._abs_ssh_key_path}
              Port {port}
              User {user or self.DEFAULT_USER}
        """
        )

    def _add_or_update_ssh_config_entry(self):
        """Update the SSH config to allow for accessing the cluster via: ssh <cluster name>"""
        connected_ssh_port = self.ssh_port
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
