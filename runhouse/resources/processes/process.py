from typing import Dict, Optional, Union

from runhouse.logger import get_logger

from runhouse.resources.hardware import _get_cluster_from, Cluster

logger = get_logger(__name__)


class Process:
    def __init__(
        self,
        name: str,
        compute: Optional[Dict] = {},
        env_vars: Union[Dict, str] = {},
        conda_env_name: Optional[str] = None,
    ):
        if name is None:
            raise ValueError("Process name must be provided.")

        self.name = name
        self._compute = compute
        self._env_vars = env_vars
        self._runtime_env = {"conda_env": conda_env_name} if conda_env_name else {}
        self._system = None

    def get_or_to(self, system: Union[str, Cluster]):
        system = _get_cluster_from(system)
        if not isinstance(system, Cluster):
            raise ValueError(f"Could not find system {system}")

        if self.name in system.list_processes():
            logger.info("Found worker in system, ignoring _compute and _env_vars.")
            process = Process(name=self.name)
            process._system = system
            return process

        return self.to(system)

    def to(self, system: Union[str, Cluster]):
        if self._system:
            raise ValueError("Process already sent to a system.")

        system = _get_cluster_from(system)
        if not system:
            raise ValueError(f"Could not find system {system}")

        system.create_process(
            name=self.name,
            compute=self._compute,
            runtime_env=self._runtime_env,
            env_vars=self._env_vars,
        )

        self._system = system
        return self


def process(*args, **kwargs):
    return Process(*args, **kwargs)
