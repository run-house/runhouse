from typing import Dict, Optional, Union

from runhouse.resources.envs.env import Env
from runhouse.resources.envs.env_factory import env as env_factory
from runhouse.resources.hardware.cluster import Cluster
from runhouse.resources.secrets.provider_secrets.provider_secret import ProviderSecret


class ApiKeySecret(ProviderSecret):
    """Secret class for providers consisting of a single API key, generally stored as an environment variable.

    .. note::
            To create an ApiKeySecret, please use the factory method :func:`provider_secret`
            and passing in the corresponding provider.
    """

    def write(
        self,
        file: bool = False,
        env: bool = False,
        path: str = None,
        env_vars: Dict = None,
        overwrite: bool = False,
    ):
        if not file or path:
            env = True
        super().write(
            file=file, env=env, path=path, env_vars=env_vars, overwrite=overwrite
        )

    def to(
        self,
        system: Union[str, Cluster],
        path: str = None,
        env: Union[str, Env] = None,
        values: bool = True,
        name: Optional[str] = None,
    ):
        if not (self.path or path or env):
            env = env_factory()
        return super().to(system=system, path=path, env=env, values=values, name=name)
