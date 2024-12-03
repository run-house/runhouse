from enum import Enum
from typing import Any, Dict, List, Optional, Union

# Internal class to represent the image construction process
class ImageSetupStepType(Enum):
    PACKAGES = "packages"
    CMD_RUN = "cmd_run"
    SETUP_CONDA_ENV = "setup_conda_env"


class ImageSetupStep:
    def __init__(
        self,
        step_type: ImageSetupStepType,
        **kwargs: Dict[str, Any],
    ):
        self.step_type = step_type
        self.kwargs = kwargs


class Image:
    def __init__(self, name: str):
        self.name = name
        self.setup_steps = []
        self.conda_env_name = None

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
