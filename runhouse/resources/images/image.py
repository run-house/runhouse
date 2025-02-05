from enum import Enum
from typing import Any, Dict, List, Optional, Union

# Internal class to represent the image construction process
class ImageSetupStepType(Enum):
    """Enum for valid Image setup step types"""

    PACKAGES = "packages"
    CMD_RUN = "cmd_run"
    SETUP_CONDA_ENV = "setup_conda_env"
    RSYNC = "rsync"
    SYNC_SECRETS = "sync_secrets"
    SET_ENV_VARS = "set_env_vars"


class ImageSetupStep:
    def __init__(
        self,
        step_type: ImageSetupStepType,
        **kwargs: Dict[str, Any],
    ):
        """
        A component of the Runhouse Image, consisting of the step type (e.g. packages, set_env_vars),
        along with arguments to provide to the function corresponding to the step type.

        Args:
            step_type (ImageSetupStepType): Type of setup step used to provide the Image.
            kwargs (Dict[str, Any]): Please refer to the corresponding functions in ``Image`` to determine
                the correct keyword arguments to provide.
        """
        self.step_type = step_type
        self.kwargs = kwargs


class Image:
    def __init__(self, name: str = None, image_id: str = None):
        """
        Runhouse Image object, specifying cluster setup properties and steps.

        Args:
            name (str): Name to assign the Runhouse image.
            image_id (str): Machine image to use, if any. (Default: ``None``)

        Example:
            >>> my_image = (
            >>>     rh.Image(name="base_image")
            >>>     .setup_conda_env(
            >>>         conda_env_name="base_env",
            >>>         conda_config={"dependencies": ["python=3.11"], "name": "base_env"},
            >>>     )
            >>>     .install_packages(["numpy", "pandas"])
            >>>     .set_env_vars({"OMP_NUM_THREADS": 1})
            >>> )
        """

        self.name = name
        self.image_id = image_id

        self.setup_steps = []
        self.conda_env_name = None
        self.docker_secret = None

    @staticmethod
    def _setup_step_config(step: ImageSetupStep):
        """Get ImageSetupStep config"""
        config = {
            "step_type": step.step_type.value,
        }
        if step.step_type == ImageSetupStepType.SYNC_SECRETS:
            secrets = step.kwargs.get("providers")
            new_kwargs = []
            for secret in secrets:
                new_kwargs.append(
                    secret if isinstance(secret, str) else secret.config(values=False)
                )
            config["kwargs"] = new_kwargs
        else:
            config["kwargs"] = step.kwargs
        return config

    @staticmethod
    def _setup_step_from_config(step: Dict):
        """Convert setup step config (dict) to ImageSetupStep object"""
        step_type = step["step_type"]
        if step_type == "sync_secrets":
            from runhouse.resources.secrets.secret import Secret

            secrets = step["kwargs"]
            secret_list = []
            for secret in secrets:
                secret_list.append(
                    Secret.from_config(secret) if isinstance(secret, Dict) else secret
                )
            kwargs = {"providers": secret_list}
        else:
            kwargs = step["kwargs"]
        return ImageSetupStep(
            step_type=ImageSetupStepType(step_type),
            **kwargs,
        )

    def _save_sub_resources(self):
        secret_steps = [
            step
            for step in self.setup_steps
            if step.step_type == ImageSetupStepType.SYNC_SECRETS
        ]
        for step in secret_steps:
            from runhouse.resources.secrets.secret import Secret

            secrets = step.kwargs.get("providers")
            for secret in secrets:
                if isinstance(secret, Secret):
                    secret.save()

    def from_docker(self, image_id: str, docker_secret: Union["Secret", str] = None):
        """Set up and use an existing Docker image.

        Args:
            image_id (str): Docker image in the following format ``"<registry>/<image>:<tag>"``
            docker_secret (Secret or str, optional): Runhouse secret corresponding to information
                necessary to pull the image from a private registry, such as Docker Hub or cloud provider.
                See ``DockerRegistrySecret`` for more information.
        """
        if self.image_id:
            raise ValueError(
                "Setting both a machine image and docker image is not yet supported."
            )
        self.image_id = f"docker:{image_id}"
        self.docker_secret = docker_secret
        return self

    def config(self) -> Dict[str, Any]:
        config = {}
        if self.name:
            config["name"] = self.name
        if self.image_id:
            config["image_id"] = self.image_id
        if self.docker_secret:
            config["docker_secret"] = (
                self.docker_secret
                if isinstance(self.docker_secret, str)
                else self.docker_secret.config()
            )
        if self.setup_steps:
            config["setup_steps"] = [
                Image._setup_step_config(step) for step in self.setup_steps
            ]

        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]):
        img = Image(name=config.get("name"), image_id=config.get("image_id"))
        if config.get("setup_steps"):
            img.setup_steps = [
                Image._setup_step_from_config(step) for step in config["setup_steps"]
            ]

        docker_secret = config.get("docker_secret")
        if docker_secret:
            if isinstance(docker_secret, Dict):
                from runhouse.resources.secrets.secret import Secret

                docker_secret = Secret.from_config(docker_secret)
            img.docker_secret = docker_secret

        return img

    ########################################################
    # Steps to build the image
    ########################################################

    def install_packages(
        self, reqs: List[Union["Package", str]], conda_env_name: Optional[str] = None
    ):
        """Install the given packages.

        Args:
            reqs (List[Package or str]): List of packages to install on cluster and env.
            conda_env_name (str, optional): Name of conda env to install the package in, if relevant. If left empty,
                defaults to base environment. (Default: ``None``)
        """

        self.setup_steps.append(
            ImageSetupStep(
                step_type=ImageSetupStepType.PACKAGES,
                reqs=reqs,
                conda_env_name=conda_env_name or self.conda_env_name,
            )
        )
        return self

    def run_bash(self, command: str, conda_env_name: Optional[str] = None):
        """Run bash commands.

        Args:
            command (str): Commands to run on the cluster.
            conda_env_name (str, optional): Name of conda env to run the command in, if applicable. (Defaut: ``None``)
        """
        self.setup_steps.append(
            ImageSetupStep(
                step_type=ImageSetupStepType.CMD_RUN,
                command=command,
                conda_env_name=conda_env_name or self.conda_env_name,
            )
        )
        return self

    def setup_conda_env(self, conda_env_name: str, conda_config: Union[str, Dict]):
        """Setup Conda env

        Args:
            conda_env_name (str): Name of conda env to create.
            conda_config (str or Dict): Path or Dict referring to the conda yaml config to use to create the conda env.
        """
        self.setup_steps.append(
            ImageSetupStep(
                step_type=ImageSetupStepType.SETUP_CONDA_ENV,
                conda_env_name=conda_env_name,
                conda_config=conda_config,
            )
        )
        self.conda_env_name = conda_env_name
        return self

    def rsync(
        self, source: str, dest: str, contents: bool = False, filter_options: str = None
    ):
        """Sync the contents of the source directory into the destination.

        Args:
            source (str): The source path.
            dest (str): The target path.
            contents (bool, optional): Whether the contents of the source directory or the directory
                itself should be copied to destination.
                If ``True`` the contents of the source directory are copied to the destination, and the source
                directory itself is not created at the destination.
                If ``False`` the source directory along with its contents are copied ot the destination, creating
                an additional directory layer at the destination. (Default: ``False``).
            filter_options (str, optional): The filter options for rsync.
        """
        self.setup_steps.append(
            ImageSetupStep(
                step_type=ImageSetupStepType.RSYNC,
                source=source,
                dest=dest,
                contents=contents,
                filter_options=filter_options,
            )
        )

    def sync_secrets(self, providers: List[Union[str, "Secret"]]):
        """Send secrets for the given providers.

        Args:
            providers (List[str or Secret]): List of providers to send secrets for.
        """
        self.setup_steps.append(
            ImageSetupStep(
                step_type=ImageSetupStepType.SYNC_SECRETS,
                providers=providers,
            )
        )
        return self

    def set_env_vars(self, env_vars: Union[str, Dict]):
        """Set environment variables.

        Args:
            env_vars (str or Dict): Dict of environment variables and values to set, or string pointing
                to local ``.env`` file consisting of env vars to set.
        """
        self.setup_steps.append(
            ImageSetupStep(
                step_type=ImageSetupStepType.SET_ENV_VARS,
                env_vars=env_vars,
            )
        )
        return self
