from enum import Enum
from typing import Any, Dict, List, Optional, Union

# Internal class to represent the image construction process
class ImageSetupStepType(Enum):
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
        self.step_type = step_type
        self.kwargs = kwargs


class Image:
    def __init__(self, name: str, image_id: str = None):
        """
        Args:
            name (str): Name to assign the Runhouse image.
            image_id (str): Machine image to use, if any. (Default: ``None``)
        """
        self.name = name
        self.image_id = image_id

        self.setup_steps = []
        self.conda_env_name = None
        self.docker_secret = None

    def from_docker(self, image_id, docker_secret: Union["Secret", str] = None):
        if self.image_id:
            raise ValueError(
                "Setting both a machine image and docker image is not yet supported."
            )
        self.image_id = f"docker:{image_id}"
        self.docker_secret = docker_secret
        return self

    def config(self) -> Dict[str, Any]:
        config = {"name": self.name}
        if self.image_id:
            config["image_id"] = self.image_id
        if self.docker_secret:
            config["docker_secret"] = (
                self.docker_secret
                if isinstance(self.docker_secret, str)
                else self.docker_secret.config()
            )
        config["setup_steps"] = [
            {
                "step_type": step.step_type.value,
                "kwargs": step.kwargs,
            }
            for step in self.setup_steps
        ]

        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]):
        img = Image(name=config["name"], image_id=config.get("image_id"))
        img.setup_steps = [
            ImageSetupStep(
                step_type=ImageSetupStepType(step["step_type"]),
                **step["kwargs"],
            )
            for step in config["setup_steps"]
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

    def install_packages(self, reqs: List[str], conda_env_name: Optional[str] = None):
        self.setup_steps.append(
            ImageSetupStep(
                step_type=ImageSetupStepType.PACKAGES,
                reqs=reqs,
                conda_env_name=conda_env_name or self.conda_env_name,
            )
        )
        return self

    def run_bash(self, command: str, conda_env_name: Optional[str] = None):
        self.setup_steps.append(
            ImageSetupStep(
                step_type=ImageSetupStepType.CMD_RUN,
                command=command,
                conda_env_name=conda_env_name or self.conda_env_name,
            )
        )
        return self

    def setup_conda_env(self, conda_env_name: str, conda_yaml: Union[str, Dict]):
        self.setup_steps.append(
            ImageSetupStep(
                step_type=ImageSetupStepType.SETUP_CONDA_ENV,
                conda_env_name=conda_env_name,
                conda_yaml=conda_yaml,
            )
        )
        self.conda_env_name = conda_env_name
        return self

    def rsync(
        self, source: str, dest: str, contents: bool = False, filter_options: str = None
    ):
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
        self.setup_steps.append(
            ImageSetupStep(
                step_type=ImageSetupStepType.SYNC_SECRETS,
                providers=providers,
            )
        )
        return self

    def set_env_vars(self, env_vars: Union[str, Dict]):
        self.setup_steps.append(
            ImageSetupStep(
                step_type=ImageSetupStepType.SET_ENV_VARS,
                env_vars=env_vars,
            )
        )
        return self
