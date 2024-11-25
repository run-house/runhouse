from enum import Enum
from typing import Any, Dict, List, Optional, Union

# Internal class to represent the image construction process
class ImageSetupStepType(Enum):
    REQS = "reqs"
    CMD_RUN = "cmd_run"


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

    def install_reqs(self, reqs: List[str], conda_env_name: Optional[str] = None):
        self.setup_steps.append(
            ImageSetupStep(
                step_type=ImageSetupStepType.REQS,
                reqs=reqs,
                conda_env_name=conda_env_name,
            )
        )
        return self

    def run_bash(self, command: str, conda_env_name: Optional[str] = None):
        self.setup_steps.append(
            ImageSetupStep(
                step_type=ImageSetupStepType.CMD_RUN,
                command=command,
                conda_env_name=conda_env_name,
            )
        )
        return self

    def add_secrets(self, secrets: List[Union[str, "Secret"]]):
        self.setup_steps.append(
            ImageSetupStep(step_type=ImageSetupStepType.SECRETS, secrets=secrets)
        )
        return self
